"""Movement system for expressive robot control.

This module provides a 100Hz control loop for managing robot movements,
combining sequential primary moves (dances, emotions, head movements) with
additive secondary moves (speech wobble, face tracking).

Architecture:
- Primary moves are queued and executed sequentially
- Secondary moves are additive offsets applied on top
- Single control point via set_target at 100Hz
- Automatic breathing animation when idle

Based on the movement systems from:
- pollen-robotics/reachy_mini_conversation_app
- eoai-dev/moltbot_body
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset, linear_pose_interpolation

logger = logging.getLogger(__name__)

# Configuration
CONTROL_LOOP_FREQUENCY_HZ = 100.0


def _env_flag(name: str, default: str = "on") -> bool:
    return os.getenv(name, default).strip().lower() not in ("off", "0", "false", "no")


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


# Body-follow face tracking (wireless base): when the head must yaw far to
# keep a tracked face in view, the base rotates underneath it so tracking
# continues past the head's range. When the face is lost and the head-only
# scan comes up empty, the base joins the search by slowly turning toward
# where the face was last seen.
#
# The base carries the whole robot, so it is driven as a *velocity* command
# through the same acceleration limiter as everything else: a base that
# snaps into motion throws the head sideways and rocks the chassis. Gains
# are deliberately unhurried -- the loop closes through the camera, which
# adds a few hundred ms of lag, and a hot gain on a laggy loop hunts.
BODY_FOLLOW_ENABLED = _env_flag("CLAWBODY_BODY_FOLLOW")
# Head yaw (degrees) that engages the base; hysteresis releases it near center
BODY_FOLLOW_START_DEG = _env_float("CLAWBODY_BODY_FOLLOW_START", 18.0)
BODY_FOLLOW_STOP_DEG = 6.0
# Proportional gain (rad/s of base speed per rad of head yaw) and speed cap
BODY_FOLLOW_GAIN = _env_float("CLAWBODY_BODY_FOLLOW_GAIN", 1.1)
BODY_FOLLOW_MAX_SPEED = float(np.deg2rad(
    _env_float("CLAWBODY_BODY_FOLLOW_MAX_SPEED", 30.0)))
BODY_SEARCH_ENABLED = _env_flag("CLAWBODY_BODY_SEARCH")
BODY_SEARCH_DELAY = 3.0  # seconds of head-only scanning before the base joins
BODY_SEARCH_SPEED = float(np.deg2rad(20.0))
# Enough for a wall-to-wall unwind (300 deg) even after a partial first leg
BODY_SEARCH_MAX_TURN = float(np.deg2rad(500.0))
# Pause body-follow briefly after an explicit turn_body/body_sway command
EXTERNAL_YAW_HOLDOFF = 1.5

# The daemon interprets head poses in the world frame; the neck (Stewart
# platform) can only realize ~+/-65 deg of yaw relative to the base, and an
# out-of-range command makes the daemon reject the ENTIRE target (silent
# freeze). All our patterns are body-relative, so we clamp the composed
# relative yaw with margin and then rotate by the base yaw before issuing.
NECK_YAW_LIMIT = float(np.deg2rad(50.0))
# Secondary offsets stack additively (face tracking + thinking + speech +
# move macros), and the platform will physically drive the head shell into
# the body if the sum pitches down too far (loud bump). Clamp the composed
# pose to a safe envelope. Sign convention (per the SDK's look_at geometry
# and the sleep pose): POSITIVE pitch = looking DOWN, the contact
# direction; negative = up, which is mechanically freer and needed to
# track standing people.
HEAD_PITCH_UP_LIMIT = float(np.deg2rad(-40.0))
HEAD_PITCH_DOWN_LIMIT = float(np.deg2rad(25.0))
HEAD_ROLL_LIMIT = float(np.deg2rad(25.0))
HEAD_Z_MIN, HEAD_Z_MAX = -0.020, 0.025  # metres
# The base is NOT continuous: the wireless body motor hits a hard stop
# around +/-157 deg (measured on hardware). Stay inside it with margin.
# Base +/-150 plus neck +/-50 still covers the full circle (+/-200 deg of
# gaze); the body search unwinds the long way around to cross the seam.
BODY_YAW_RANGE = float(np.deg2rad(150.0))

# --- Motion envelope ------------------------------------------------------
# Everything the robot does funnels through one 100Hz set_target, and the
# daemon servos to whatever pose it is handed as fast as the motors allow.
# A *step* in that pose is therefore a full-torque impulse: the head is the
# heaviest thing on the robot and it sits on top, so an impulse rocks the
# base and can tip the robot over. Bounding velocity alone is not enough --
# going from 0 to the speed cap in one tick is still an impulse -- so the
# final command is passed through a velocity- AND acceleration-limited
# follower. These are a safety envelope, sized so well-behaved motion never
# touches them; they exist to catch the pathological cases (a detection
# jump, a macro reversal, the daemon handing control back mid-pose).
HEAD_MAX_SPEED = float(np.deg2rad(_env_float("CLAWBODY_HEAD_MAX_SPEED", 140.0)))
HEAD_MAX_ACCEL = float(np.deg2rad(_env_float("CLAWBODY_HEAD_MAX_ACCEL", 900.0)))
HEAD_MAX_LIN_SPEED = _env_float("CLAWBODY_HEAD_MAX_LIN_SPEED", 0.12)  # m/s
HEAD_MAX_LIN_ACCEL = _env_float("CLAWBODY_HEAD_MAX_LIN_ACCEL", 0.9)  # m/s^2
# Base slew: the whole robot rotates, so it is the most tip-prone axis
BODY_YAW_MAX_SPEED = float(np.deg2rad(_env_float("CLAWBODY_BODY_MAX_SPEED", 70.0)))
BODY_YAW_MAX_ACCEL = float(np.deg2rad(_env_float("CLAWBODY_BODY_MAX_ACCEL", 110.0)))

# The daemon refuses set_target while it plays a recorded move of its own
# (emotions/dances go through its player), so when it hands control back
# the real head can be far from where this loop thinks it is. Commanding
# that difference in one tick is exactly the snap the limiter exists to
# prevent, so re-seed the limiter from the measured pose instead.
RESYNC_TOLERANCE = float(np.deg2rad(_env_float("CLAWBODY_RESYNC_DEG", 20.0)))
RESYNC_HOLD = 0.3  # seconds of sustained divergence before re-seeding
RESYNC_POLL = 0.1  # seconds between measured-pose reads

# Global scale on the expressive gestures (nod/shake/bounce/sway)
GESTURE_SCALE = float(np.clip(_env_float("CLAWBODY_GESTURE_SCALE", 1.0), 0.2, 1.5))

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]
SpeechOffsets = Tuple[float, float, float, float, float, float]


def _wrap_angles(values: NDArray) -> NDArray:
    """Wrap angles to (-pi, pi] elementwise."""
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _approach_speed(error: Any, v_max: float, a_max: float, dt: float) -> NDArray:
    """Fastest speed that still decelerates to a stop exactly on target.

    The textbook answer, `sqrt(2*a*e)`, is the continuous-time one: braking
    from it in discrete ticks travels an extra `v*dt/2`, so a follower using
    it arrives with velocity left over and slams into whatever it was
    approaching. Inverting the discrete braking distance
    `e = v^2/(2a) + v*dt/2` instead gives the form below.

    Near zero that form still asks for about `2e/dt` -- enough to overshoot
    on the final tick -- so it is capped by the speed that nulls the error
    in exactly one tick. The two cross over smoothly, leaving a profile
    that is monotone in `e`, finite-sloped at the origin (no buzzing
    against the acceleration limit once settled) and never overshoots.
    """
    error = np.asarray(error, dtype=float)
    magnitude = np.abs(error)
    half_tick = 0.5 * a_max * dt
    braked = np.sqrt(half_tick * half_tick + 2.0 * a_max * magnitude) - half_tick
    speed = np.sign(error) * np.minimum(magnitude / dt, braked)
    return np.clip(speed, -v_max, v_max)


class MotionLimiter:
    """Velocity- and acceleration-limited follower for a vector signal.

    Tracks a smooth input almost transparently -- the input's own velocity
    is fed forward, so in steady state the output is the input delayed by
    about one tick -- while any step or whip in the input is served at
    bounded speed and bounded acceleration.

    The catch-up term is the fastest approach speed that can still stop
    exactly on target (`sqrt(2*a*err)`), so recovering from a step never
    overshoots into an oscillation.
    """

    def __init__(
        self,
        size: int,
        v_max: float,
        a_max: float,
        dt: float,
        wrap: bool = False,
        ff_tau: float = 0.05,
    ) -> None:
        """Initialize the follower.

        Args:
            size: Number of independent axes
            v_max: Speed limit (units/s)
            a_max: Acceleration limit (units/s^2)
            dt: Control period in seconds
            wrap: Treat values as angles and use shortest-arc differences
            ff_tau: Smoothing constant for the input-velocity feed-forward.
                Long enough to turn a slower producer's staircase (the
                camera worker publishes at 25Hz into a 100Hz loop) into a
                ramp, short enough not to lag real motion.
        """
        self.pos = np.zeros(size, dtype=float)
        self.vel = np.zeros(size, dtype=float)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.dt = float(dt)
        self.wrap = bool(wrap)
        self._ff = np.zeros(size, dtype=float)
        self._ff_alpha = float(min(1.0, dt / max(ff_tau, dt)))
        self._prev_target: Optional[NDArray] = None

    def reset(self, pos: Any) -> None:
        """Re-seed the follower at `pos`, at rest."""
        self.pos = np.asarray(pos, dtype=float).copy()
        self.vel[:] = 0.0
        self._ff[:] = 0.0
        self._prev_target = self.pos.copy()

    def step(self, target: Any) -> NDArray:
        """Advance one tick toward `target` and return the limited output."""
        target = np.asarray(target, dtype=float)
        if self._prev_target is None:
            self.reset(target)
            return self.pos.copy()

        # Feed forward how fast the target itself is moving
        moved = target - self._prev_target
        if self.wrap:
            moved = _wrap_angles(moved)
        self._prev_target = target.copy()
        self._ff += self._ff_alpha * (moved / self.dt - self._ff)

        error = target - self.pos
        if self.wrap:
            error = _wrap_angles(error)
        catch_up = _approach_speed(error, self.v_max, self.a_max, self.dt)

        desired = np.clip(self._ff + catch_up, -self.v_max, self.v_max)
        max_dv = self.a_max * self.dt
        self.vel += np.clip(desired - self.vel, -max_dv, max_dv)
        self.pos = self.pos + self.vel * self.dt
        return self.pos.copy()

    def clamp_pos(self, low: Any, high: Any) -> None:
        """Clip the follower's position, zeroing velocity on clipped axes."""
        clamped = np.clip(self.pos, low, high)
        stopped = clamped != self.pos
        if np.any(stopped):
            self.pos = clamped
            self.vel[stopped] = 0.0

    def at_rest(self, tolerance: float) -> bool:
        """True while the output is barely moving."""
        return bool(np.max(np.abs(self.vel)) < tolerance)


class BreathingMove(Move):
    """Continuous breathing animation for idle state."""
    
    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ):
        """Initialize breathing move.
        
        Args:
            interpolation_start_pose: Current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions
            interpolation_duration: Time to blend to neutral (seconds)
        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration
        
        # Target neutral pose
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])
        
        # Breathing parameters. Antenna sway is kept small and slow: the
        # antenna servos sit right next to the mic array, and their steps
        # register as speech-loud transients that fool turn detection.
        self.breathing_z_amplitude = 0.005  # 5mm gentle movement
        self.breathing_frequency = 0.1  # Hz
        self.antenna_sway_amplitude = np.deg2rad(8)  # degrees
        self.antenna_frequency = 0.3  # Hz
        
    @property
    def duration(self) -> float:
        """Duration of the move (infinite for breathing)."""
        return float("inf")
        
    def evaluate(self, t: float) -> tuple:
        """Evaluate the breathing pose at time t."""
        if t < self.interpolation_duration:
            # Interpolate to neutral
            alpha = t / self.interpolation_duration
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose, 
                self.neutral_head_pose, 
                alpha
            )
            antennas = (1 - alpha) * self.interpolation_start_antennas + alpha * self.neutral_antennas
            antennas = antennas.astype(np.float64)
        else:
            # Breathing pattern
            breathing_t = t - self.interpolation_duration
            
            z_offset = self.breathing_z_amplitude * np.sin(
                2 * np.pi * self.breathing_frequency * breathing_t
            )
            head_pose = create_head_pose(
                x=0, y=0, z=z_offset, 
                roll=0, pitch=0, yaw=0, 
                degrees=True, mm=False
            )
            
            antenna_sway = self.antenna_sway_amplitude * np.sin(
                2 * np.pi * self.antenna_frequency * breathing_t
            )
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)
            
        return (head_pose, antennas, 0.0)


class HeadLookMove(Move):
    """Move to look in a specific direction."""
    
    # Sign convention: positive pitch = down (toward the body), negative = up
    DIRECTIONS = {
        "left": (0, 0, 0, 0, 0, 30),      # yaw left
        "right": (0, 0, 0, 0, 0, -30),    # yaw right
        "up": (0, 0, 10, 0, -15, 0),      # pitch up, z up
        "down": (0, 0, -5, 0, 15, 0),     # pitch down, z down
        "front": (0, 0, 0, 0, 0, 0),      # neutral
    }
    
    def __init__(
        self,
        direction: str,
        start_pose: NDArray[np.float32],
        start_antennas: Tuple[float, float],
        duration: float = 1.0,
        target_yaw_deg: Optional[float] = None,
    ):
        """Initialize head look move.

        Args:
            direction: One of 'left', 'right', 'up', 'down', 'front'
            start_pose: Current head pose
            start_antennas: Current antenna positions
            duration: Move duration in seconds
            target_yaw_deg: If set, look at this exact yaw (degrees, positive
                = left) instead of the discrete direction target. Used for
                sound-source (DoA) orientation.
        """
        self.direction = direction
        self.start_pose = np.asarray(start_pose, dtype=float)
        self.start_antennas = np.array(start_antennas)
        self.target_yaw_deg = target_yaw_deg

        if target_yaw_deg is not None:
            self.target_pose = create_head_pose(
                x=0, y=0, z=0, roll=0, pitch=0, yaw=target_yaw_deg,
                degrees=True, mm=True
            )
        else:
            # Get target pose from direction
            params = self.DIRECTIONS.get(direction, self.DIRECTIONS["front"])
            self.target_pose = create_head_pose(
                x=params[0], y=params[1], z=params[2],
                roll=params[3], pitch=params[4], yaw=params[5],
                degrees=True, mm=True
            )
        self.target_antennas = np.array([0.0, 0.0])
        # Callers ask for a duration by feel; how violent that is depends on
        # how far the head actually has to travel, which only this object
        # knows. Stretch anything that would breach the motion envelope so
        # no caller can whip the head by passing an eager number.
        self._duration = max(float(duration), self._minimum_duration())

    def _minimum_duration(self) -> float:
        """Shortest duration whose eased profile stays inside the envelope.

        Smoothstep peaks at 1.5*delta/T in speed and 6*delta/T^2 in
        acceleration; invert both for the travel this move covers.
        """
        swing = float(
            (
                R.from_matrix(self.target_pose[:3, :3])
                * R.from_matrix(self.start_pose[:3, :3]).inv()
            ).magnitude()
        )
        shift = float(np.linalg.norm(self.target_pose[:3, 3] - self.start_pose[:3, 3]))

        limits = [
            (swing, HEAD_MAX_SPEED, HEAD_MAX_ACCEL),
            (shift, HEAD_MAX_LIN_SPEED, HEAD_MAX_LIN_ACCEL),
        ]
        return max(
            [0.0]
            + [
                max(1.5 * delta / v_max, float(np.sqrt(6.0 * delta / a_max)))
                for delta, v_max, a_max in limits
                if delta > 0.0
            ]
        )

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple:
        """Evaluate pose at time t."""
        alpha = min(1.0, t / self._duration)
        # Smooth easing
        alpha = alpha * alpha * (3 - 2 * alpha)
        
        head_pose = linear_pose_interpolation(
            self.start_pose,
            self.target_pose,
            alpha
        )
        antennas = (1 - alpha) * self.start_antennas + alpha * self.target_antennas
        
        return (head_pose, antennas.astype(np.float64), 0.0)


class GestureMove(Move):
    """Nod / head-shake / bounce as a windowed oscillation about a held pose.

    These used to be chains of discrete look moves -- a 30 deg reversal
    every 0.22 s, several hundred deg/s of head swing, which is exactly the
    impulse that rocks the base. A windowed sinusoid reads the same
    ("yes" / "no") but has continuous velocity: it starts and ends at rest
    on the pose it began from, so no other stage has to absorb a step.

    Face tracking keeps running underneath -- these are primary moves and
    tracking is an additive secondary offset -- so the robot nods *at* the
    person it is looking at instead of nodding at where they used to be.
    """

    # axis -> (create_head_pose keyword, amplitude, frequency Hz, cycles).
    # A windowed sinusoid peaks at A*(2*pi*f)^2 in acceleration; these are
    # picked to land around 85-90% of HEAD_MAX_ACCEL, so nodding and
    # disagreeing stay the quickest things the robot does while still
    # having headroom rather than riding the limiter.
    PATTERNS: Dict[str, Tuple[str, float, float, float]] = {
        "nod": ("pitch", float(np.deg2rad(9.5)), 1.45, 2.0),
        "shake": ("yaw", float(np.deg2rad(11.0)), 1.35, 2.0),
        "bounce": ("z", 0.007, 1.55, 2.0),
        "sway": ("roll", float(np.deg2rad(8.0)), 1.1, 2.0),
    }

    def __init__(
        self,
        gesture: str,
        start_pose: NDArray[np.float32],
        start_antennas: Tuple[float, float],
        scale: float = GESTURE_SCALE,
    ):
        """Initialize a gesture.

        Args:
            gesture: One of 'nod', 'shake', 'bounce', 'sway'
            start_pose: Body-relative pose to oscillate about (and return to)
            start_antennas: Antenna positions to hold
            scale: Amplitude multiplier (CLAWBODY_GESTURE_SCALE by default)
        """
        axis, amplitude, frequency, cycles = self.PATTERNS.get(
            gesture, self.PATTERNS["nod"]
        )
        self.gesture = gesture
        self.axis = axis
        self.amplitude = amplitude * float(np.clip(scale, 0.2, 1.5))
        self.frequency = frequency
        self._duration = cycles / frequency
        self.start_pose = np.asarray(start_pose, dtype=float).copy()
        self.start_antennas = np.array(start_antennas, dtype=np.float64)
        # Named like HeadLookMove so queued sequences can chain off a gesture
        self.target_pose = self.start_pose
        self.target_antennas = self.start_antennas

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple:
        """Evaluate the gesture at time t."""
        t = float(np.clip(t, 0.0, self._duration))
        # Half-sine window: value AND velocity are zero at both ends, so the
        # gesture blends into and out of the held pose without a kick
        window = np.sin(np.pi * t / self._duration)
        value = self.amplitude * window * np.sin(2 * np.pi * self.frequency * t)

        components = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        components[self.axis] = float(value)
        offset = create_head_pose(degrees=False, mm=False, **components)

        pose = compose_world_offset(self.start_pose, offset, reorthonormalize=True)
        return (pose, self.start_antennas, 0.0)


def combine_full_body(primary: FullBodyPose, secondary: FullBodyPose) -> FullBodyPose:
    """Combine primary pose with secondary offsets."""
    primary_head, primary_ant, primary_yaw = primary
    secondary_head, secondary_ant, secondary_yaw = secondary
    
    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=True)
    combined_ant = (
        primary_ant[0] + secondary_ant[0],
        primary_ant[1] + secondary_ant[1],
    )
    combined_yaw = primary_yaw + secondary_yaw
    
    return (combined_head, combined_ant, combined_yaw)


def clone_pose(pose: FullBodyPose) -> FullBodyPose:
    """Deep copy a full body pose."""
    head, ant, yaw = pose
    return (head.copy(), (float(ant[0]), float(ant[1])), float(yaw))


def move_start_state(movement_manager: Any) -> Tuple[NDArray, Tuple[float, float]]:
    """Pose and antennas a newly queued move should start from.

    Always ask the manager, never `robot.get_current_head_pose()`. The
    measured pose is world-frame: it already contains the base yaw and
    every secondary offset (face tracking, speech wobble, thinking sway).
    Feeding it back in as a *primary* pose made the loop add both a second
    time, so the commanded pose jumped by the whole tracking offset plus
    the whole base angle in a single 10 ms tick -- the snap that rocks the
    chassis. The manager's primary pose is body-relative and composes
    cleanly.
    """
    getter = getattr(movement_manager, "get_primary_pose", None)
    if getter is not None:
        try:
            head, antennas, _ = getter()
            return head, antennas
        except Exception as e:
            logger.debug("Falling back to neutral move start pose: %s", e)
    return create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0.0, 0.0)


@dataclass
class MovementState:
    """State for the movement system."""
    current_move: Optional[Move] = None
    move_start_time: Optional[float] = None
    last_activity_time: float = 0.0
    speech_offsets: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    face_tracking_offsets: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    thinking_offsets: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    last_primary_pose: Optional[FullBodyPose] = None
    
    def update_activity(self) -> None:
        self.last_activity_time = time.monotonic()


class MovementManager:
    """Coordinate robot movements at 100Hz.
    
    This class manages:
    - Sequential primary moves (dances, emotions, head movements)
    - Additive secondary offsets (speech wobble, face tracking)
    - Automatic idle breathing animation
    - Thread-safe communication with other components
    
    Example:
        manager = MovementManager(robot)
        manager.start()
        
        # Queue a head movement
        manager.queue_move(HeadLookMove("left", ...))
        
        # Set speech offsets (called by HeadWobbler)
        manager.set_speech_offsets((0, 0, 0.01, 0.1, 0, 0))
        
        manager.stop()
    """
    
    def __init__(
        self,
        current_robot: ReachyMini,
        camera_worker: Any = None,
    ):
        """Initialize movement manager.
        
        Args:
            current_robot: Connected ReachyMini instance
            camera_worker: Optional camera worker for face tracking
        """
        self.current_robot = current_robot
        self.camera_worker = camera_worker
        
        self._now = time.monotonic
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        
        # Initialize neutral pose
        neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral, (0.0, 0.0), 0.0)
        
        # Move queue
        self.move_queue: deque[Move] = deque()
        
        # Configuration
        self.idle_inactivity_delay = 0.3  # seconds before breathing starts
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency
        
        # Thread state
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_listening = False
        self._breathing_active = False
        
        # Last commanded pose for smooth transitions
        self._last_commanded_pose = clone_pose(self.state.last_primary_pose)
        self._listening_antennas = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4
        
        # Cross-thread communication
        self._command_queue: Queue[Tuple[str, Any]] = Queue()
        
        # Speech offsets (thread-safe)
        self._speech_lock = threading.Lock()
        self._pending_speech_offsets: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._speech_dirty = False
        
        # Processing/thinking animation state
        self._processing = False
        self._processing_start_time = 0.0
        self._thinking_amplitude = 0.0  # 0..1 envelope for smooth fade in/out
        self._thinking_antenna_offsets: Tuple[float, float] = (0.0, 0.0)

        # Persistent base body yaw (radians): slewed toward its target each
        # tick and added to every command, so the base can rotate while head
        # moves/offsets compose on top. Driven either by an absolute target
        # (turn_body / body_sway) or by a velocity command (body-follow);
        # either way the slew is speed- and acceleration-limited.
        self._body_yaw_current = 0.0
        self._body_yaw_target = 0.0
        self._body_yaw_vel = 0.0
        self._body_yaw_vel_cmd: Optional[float] = None
        self.body_yaw_rate = BODY_YAW_MAX_SPEED  # max slew speed, rad/s
        self.body_yaw_accel = BODY_YAW_MAX_ACCEL  # rad/s^2
        # Hard range of the base motor; None only for tests
        self.body_yaw_limit: Optional[float] = BODY_YAW_RANGE
        self._last_cmd_error_log = float("-inf")

        # Final output smoothing: nothing reaches the robot without passing
        # through these, whatever upstream does
        self._head_rot_limiter = MotionLimiter(
            3, HEAD_MAX_SPEED, HEAD_MAX_ACCEL, self.target_period, wrap=True
        )
        self._head_pos_limiter = MotionLimiter(
            3, HEAD_MAX_LIN_SPEED, HEAD_MAX_LIN_ACCEL, self.target_period
        )
        self._last_resync_poll = float("-inf")
        self._diverged_since: Optional[float] = None
        # While suspended the loop composes nothing and sends nothing, so
        # another controller (goto_sleep/wake_up) owns the robot outright
        self._suspended = False

        # Body-follow face tracking state
        self._body_follow_active = False
        self._last_face_side = 1.0  # +1 = face was to the left, -1 = right
        self._search_started: Optional[float] = None
        self._search_turned = 0.0
        self._external_yaw_cmd_time = float("-inf")

        # Shared state lock
        self._shared_lock = threading.Lock()
        self._shared_last_activity = self.state.last_activity_time
        self._shared_is_listening = False
        self._shared_body_yaw = (0.0, 0.0)  # (current, target)
        self._shared_primary_pose = clone_pose(self.state.last_primary_pose)
        self._shared_suspended = False
        self._base_active_until = float("-inf")
        self._shared_base_active_until = float("-inf")
        
    def queue_move(self, move: Move) -> None:
        """Queue a primary move. Thread-safe."""
        self._command_queue.put(("queue_move", move))
        
    def clear_move_queue(self) -> None:
        """Clear all queued moves. Thread-safe."""
        self._command_queue.put(("clear_queue", None))
        
    def set_speech_offsets(self, offsets: SpeechOffsets) -> None:
        """Update speech-driven offsets. Thread-safe."""
        with self._speech_lock:
            self._pending_speech_offsets = offsets
            self._speech_dirty = True
            
    def set_listening(self, listening: bool) -> None:
        """Set listening state (freezes antennas). Thread-safe."""
        self._command_queue.put(("set_listening", listening))
        
    def set_processing(self, processing: bool) -> None:
        """Set processing state (triggers thinking animation). Thread-safe.

        When True, the robot shows a continuous 'thinking' animation as
        secondary offsets -- gentle head sway and asymmetric antenna scanning.
        Face tracking continues underneath since this is additive.
        """
        self._command_queue.put(("set_processing", processing))

    def set_body_yaw(self, yaw_rad: float, relative: bool = False) -> None:
        """Set the persistent base body yaw target in radians. Thread-safe.

        The control loop slews toward the target at body_yaw_rate, so large
        turns (including a full 360°) happen smoothly over multiple ticks.
        """
        self._command_queue.put(("set_body_yaw", (float(yaw_rad), bool(relative))))

    def halt_body_yaw(self) -> None:
        """Stop any body rotation in progress at its current angle. Thread-safe."""
        self._command_queue.put(("halt_body_yaw", None))

    def set_suspended(self, suspended: bool) -> None:
        """Stop or resume issuing set_target. Thread-safe.

        Hand the robot over to something else -- goto_sleep, wake_up, any
        blocking goto_target -- by suspending first, otherwise this loop
        keeps writing at 100Hz and drags the robot straight back out of
        whatever pose that call put it in. On resume the smoother is
        re-seeded from the robot's real pose, so control returns without a
        step no matter where it was left.
        """
        self._command_queue.put(("set_suspended", bool(suspended)))

    def is_suspended(self) -> bool:
        """True while output to the robot is suspended. Thread-safe."""
        with self._shared_lock:
            return self._shared_suspended

    def get_body_yaw(self) -> Tuple[float, float]:
        """Get (current, target) base body yaw in radians. Thread-safe."""
        with self._shared_lock:
            return self._shared_body_yaw

    def get_primary_pose(self) -> FullBodyPose:
        """Body-relative primary pose the loop is holding. Thread-safe.

        This -- not the robot's measured head pose -- is what a newly
        queued move must start from; see `move_start_state`.
        """
        with self._shared_lock:
            return clone_pose(self._shared_primary_pose)

    def is_base_active(self) -> bool:
        """True while the base motor is slewing (or just stopped). Thread-safe.

        Used to gate quiet mic frames: the base motor's noise reaches the
        chassis mics and can fool the server VAD into phantom user turns.
        """
        with self._shared_lock:
            return self._now() < self._shared_base_active_until
        
    def is_idle(self) -> bool:
        """Check if robot has been idle. Thread-safe."""
        with self._shared_lock:
            if self._shared_is_listening:
                return False
            return self._now() - self._shared_last_activity >= self.idle_inactivity_delay
            
    def _poll_signals(self, current_time: float) -> None:
        """Process queued commands and pending offsets."""
        # Apply speech offsets
        with self._speech_lock:
            if self._speech_dirty:
                self.state.speech_offsets = self._pending_speech_offsets
                self._speech_dirty = False
                self.state.update_activity()
                
        # Process commands
        while True:
            try:
                cmd, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(cmd, payload, current_time)
            
    def _update_face_tracking(self, current_time: float) -> None:
        """Get face tracking offsets from camera worker thread."""
        if self.camera_worker is not None:
            offsets = self.camera_worker.get_face_tracking_offsets()
            self.state.face_tracking_offsets = offsets
        else:
            # No camera worker, use neutral offsets
            self.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _update_body_follow(self, current_time: float) -> None:
        """Rotate the base so face tracking works beyond the head's yaw range.

        While a face is tracked, head-yaw excursions past the start threshold
        recruit the base with proportional velocity; the loop closes through
        the camera (as the base turns toward the face, the head offset
        shrinks), so the robot ends up squarely facing the person. When the
        face is lost and head-only scanning stays empty, the base slowly
        turns toward the side where the face was last seen, up to about one
        full turn. Explicit yaw commands (turn_body/body_sway) and
        choreographed moves take precedence.
        """
        # Any tick that does not explicitly ask the base to turn releases it
        self._body_yaw_vel_cmd = None

        cw = self.camera_worker
        if cw is None or not (BODY_FOLLOW_ENABLED or BODY_SEARCH_ENABLED):
            return
        if not hasattr(cw, "is_face_tracked"):
            return

        # Yield to explicit rotations and non-idle moves (dances/emotions)
        if (
            current_time - self._external_yaw_cmd_time < EXTERNAL_YAW_HOLDOFF
            or abs(self._body_yaw_target - self._body_yaw_current) > np.deg2rad(3.0)
            or (self.state.current_move is not None and not self._breathing_active)
        ):
            self._body_follow_active = False
            self._search_started = None
            return

        if BODY_FOLLOW_ENABLED and cw.is_face_tracked():
            self._search_started = None
            yaw_off = float(self.state.face_tracking_offsets[5])
            if abs(yaw_off) > np.deg2rad(3.0):
                self._last_face_side = 1.0 if yaw_off > 0 else -1.0
            if not self._body_follow_active:
                if abs(yaw_off) < np.deg2rad(BODY_FOLLOW_START_DEG):
                    return
                self._body_follow_active = True
                logger.debug(
                    "Body follow engaged (head yaw %.0f°)", float(np.rad2deg(yaw_off))
                )
            elif abs(yaw_off) < np.deg2rad(BODY_FOLLOW_STOP_DEG):
                self._body_follow_active = False
                return
            # Velocity command, not a position target: _advance_body_yaw
            # ramps into and out of it under the acceleration limit, so the
            # base eases into the turn instead of jerking the chassis
            self._body_yaw_vel_cmd = float(np.clip(
                BODY_FOLLOW_GAIN * yaw_off,
                -BODY_FOLLOW_MAX_SPEED,
                BODY_FOLLOW_MAX_SPEED,
            ))
            return

        self._body_follow_active = False

        # Face lost: let the base join the search after the head-only scan
        # has come up empty for a while
        if not (BODY_SEARCH_ENABLED and cw.is_scanning() and cw.has_seen_face()):
            self._search_started = None
            return
        if self._search_started is None:
            self._search_started = current_time
            self._search_turned = 0.0
            return
        if current_time - self._search_started < BODY_SEARCH_DELAY:
            return
        if self._search_turned >= BODY_SEARCH_MAX_TURN:
            return
        if self._search_turned == 0.0:
            logger.info(
                "Base joining face search, turning %s",
                "left" if self._last_face_side > 0 else "right",
            )
        step = BODY_SEARCH_SPEED * self.target_period
        # The base can't cross its hard stop; when the preferred side is
        # blocked, unwind the long way around so the search still covers the
        # sector behind the seam
        if self.body_yaw_limit is not None:
            next_yaw = self._body_yaw_current + self._last_face_side * step
            if abs(next_yaw) > self.body_yaw_limit - np.deg2rad(2.0) and (
                np.sign(next_yaw) == np.sign(self._last_face_side)
            ):
                self._last_face_side = -self._last_face_side
                logger.info(
                    "Base at its rotation stop; search unwinding %s",
                    "left" if self._last_face_side > 0 else "right",
                )
        self._search_turned += step
        self._body_yaw_vel_cmd = self._last_face_side * BODY_SEARCH_SPEED

    def _update_thinking_offsets(self, current_time: float) -> None:
        """Compute thinking animation as secondary offsets.
        
        Produces a gentle head sway (yaw drift, slight upward pitch, z bob)
        and asymmetric antenna scanning pattern. The amplitude envelope
        smoothly ramps up over 0.5s and decays over 0.5s for organic feel.
        """
        # Update amplitude envelope
        if self._processing:
            # Ramp up over 0.5s
            elapsed = current_time - self._processing_start_time
            self._thinking_amplitude = min(1.0, elapsed / 0.5)
        elif self._thinking_amplitude > 0:
            # Smooth decay at 2.0/s (full decay in 0.5s)
            self._thinking_amplitude = max(
                0.0, self._thinking_amplitude - 2.0 * self.target_period
            )
        
        # If fully decayed, zero everything and bail
        if self._thinking_amplitude < 0.001:
            self._thinking_amplitude = 0.0
            self.state.thinking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._thinking_antenna_offsets = (0.0, 0.0)
            return
        
        amp = self._thinking_amplitude
        t = current_time - self._processing_start_time
        
        # Head offsets (radians / metres -- degrees=False, mm=False)
        # Slow yaw drift: ±12° at 0.15 Hz
        yaw = amp * np.deg2rad(12) * np.sin(2 * np.pi * 0.15 * t)
        # Slight upward pitch (negative = up): 6° base + 3° oscillation at 0.2 Hz
        pitch = -amp * (np.deg2rad(6) + np.deg2rad(3) * np.sin(2 * np.pi * 0.2 * t))
        # Gentle z bob: 3 mm at 0.12 Hz
        z = amp * 0.003 * np.sin(2 * np.pi * 0.12 * t)
        
        self.state.thinking_offsets = (0.0, 0.0, z, 0.0, pitch, yaw)
        
        # Antenna offsets: asymmetric scan (phase offset creates "searching"
        # feel); kept modest to limit servo noise near the mics
        left_ant = amp * np.deg2rad(12) * np.sin(2 * np.pi * 0.4 * t)
        right_ant = amp * np.deg2rad(12) * np.sin(2 * np.pi * 0.4 * t + 1.2)
        self._thinking_antenna_offsets = (left_ant, right_ant)
        
    def _handle_command(self, cmd: str, payload: Any, current_time: float) -> None:
        """Handle a single command."""
        if cmd == "queue_move":
            if isinstance(payload, Move):
                self.move_queue.append(payload)
                self.state.update_activity()
                logger.debug("Queued move, queue size: %d", len(self.move_queue))
        elif cmd == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.info("Cleared move queue")
        elif cmd == "set_listening":
            desired = bool(payload)
            if self._is_listening != desired:
                self._is_listening = desired
                if desired:
                    self._listening_antennas = self._last_commanded_pose[1]
                    self._antenna_unfreeze_blend = 0.0
                else:
                    self._antenna_unfreeze_blend = 0.0
                self.state.update_activity()
        elif cmd == "set_processing":
            desired = bool(payload)
            if desired and not self._processing:
                self._processing = True
                self._processing_start_time = self._now()
                # Interrupt breathing so thinking animation is clean
                if self._breathing_active and isinstance(self.state.current_move, BreathingMove):
                    self.state.current_move = None
                    self.state.move_start_time = None
                    self._breathing_active = False
                self.state.update_activity()
                logger.debug("Processing started - thinking animation active")
            elif not desired and self._processing:
                self._processing = False
                # Amplitude will decay smoothly in _update_thinking_offsets
                self.state.update_activity()
                logger.debug("Processing ended - thinking animation decaying")
        elif cmd == "set_body_yaw":
            yaw, relative = payload
            self._body_yaw_target = (self._body_yaw_target + yaw) if relative else yaw
            self._body_yaw_vel_cmd = None  # explicit target wins over body-follow
            self._external_yaw_cmd_time = current_time
            self.state.update_activity()
            logger.info("Body yaw target: %.0f°", float(np.rad2deg(self._body_yaw_target)))
        elif cmd == "halt_body_yaw":
            self._body_yaw_target = self._body_yaw_current
            self._body_yaw_vel_cmd = None
            self._external_yaw_cmd_time = current_time
        elif cmd == "set_suspended":
            desired = bool(payload)
            if self._suspended != desired:
                self._suspended = desired
                if desired:
                    # Drop queued motion: it belongs to the old context and
                    # would fire the moment control came back
                    self.move_queue.clear()
                    self.state.current_move = None
                    self.state.move_start_time = None
                    self._breathing_active = False
                    self._body_yaw_vel_cmd = None
                    logger.info("Movement output suspended")
                else:
                    self._reseed_from_measured()
                    self.state.update_activity()
                    logger.info("Movement output resumed")
                
    def _manage_move_queue(self, current_time: float) -> None:
        """Advance the move queue."""
        # Check if current move is done
        if self.state.current_move is not None and self.state.move_start_time is not None:
            elapsed = current_time - self.state.move_start_time
            if elapsed >= self.state.current_move.duration:
                self.state.current_move = None
                self.state.move_start_time = None
                
        # Start next move if available
        if self.state.current_move is None and self.move_queue:
            self.state.current_move = self.move_queue.popleft()
            self.state.move_start_time = current_time
            self._breathing_active = isinstance(self.state.current_move, BreathingMove)
            logger.debug("Starting move with duration: %s", self.state.current_move.duration)
            
    def _manage_breathing(self, current_time: float) -> None:
        """Start breathing when idle."""
        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._breathing_active
            and not self._processing
        ):
            idle_for = current_time - self.state.last_activity_time
            if idle_for >= self.idle_inactivity_delay:
                try:
                    # Start from the pose the loop is already holding, NOT
                    # from the robot's measured pose. The measured pose is
                    # world-frame and already includes the base yaw and the
                    # face-tracking offset, both of which get composed on
                    # again below -- and breathing restarts after every
                    # move, so that double count fired constantly.
                    start_head, start_antennas = move_start_state(self)

                    breathing = BreathingMove(
                        interpolation_start_pose=start_head,
                        interpolation_start_antennas=start_antennas,
                        interpolation_duration=1.0,
                    )
                    self.move_queue.append(breathing)
                    self._breathing_active = True
                    self.state.update_activity()
                    logger.debug("Started breathing after %.1fs idle", idle_for)
                except Exception as e:
                    logger.error("Failed to start breathing: %s", e)
                    
        # Stop breathing if new moves queued
        if isinstance(self.state.current_move, BreathingMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            
    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get current primary pose from move or last pose."""
        if self.state.current_move is not None and self.state.move_start_time is not None:
            t = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(t)
            
            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([0.0, 0.0])
            if body_yaw is None:
                body_yaw = 0.0
                
            pose = (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))
            self.state.last_primary_pose = clone_pose(pose)
            return pose
            
        if self.state.last_primary_pose is not None:
            return clone_pose(self.state.last_primary_pose)
            
        neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        return (neutral, (0.0, 0.0), 0.0)
        
    def _get_secondary_pose(self) -> FullBodyPose:
        """Get secondary offsets (speech + face tracking + thinking)."""
        offsets = [
            self.state.speech_offsets[i]
            + self.state.face_tracking_offsets[i]
            + self.state.thinking_offsets[i]
            for i in range(6)
        ]
        
        secondary_head = create_head_pose(
            x=offsets[0], y=offsets[1], z=offsets[2],
            roll=offsets[3], pitch=offsets[4], yaw=offsets[5],
            degrees=False, mm=False
        )
        return (secondary_head, self._thinking_antenna_offsets, 0.0)
        
    def _compose_pose(self, current_time: float) -> FullBodyPose:
        """Compose final pose from primary and secondary."""
        primary = self._get_primary_pose(current_time)
        secondary = self._get_secondary_pose()
        return combine_full_body(primary, secondary)
        
    def _blend_antennas(self, target: Tuple[float, float]) -> Tuple[float, float]:
        """Blend antennas with listening freeze state."""
        if self._is_listening:
            return self._listening_antennas
            
        # Blend back from freeze
        blend = min(1.0, self._antenna_unfreeze_blend + self.target_period / self._antenna_blend_duration)
        self._antenna_unfreeze_blend = blend
        
        return (
            self._listening_antennas[0] * (1 - blend) + target[0] * blend,
            self._listening_antennas[1] * (1 - blend) + target[1] * blend,
        )
        
    def _advance_body_yaw(self) -> float:
        """Slew the persistent base yaw; return the current value.

        Speed *and* acceleration limited. The base carries the entire robot,
        so stepping straight to the speed cap throws the head sideways and
        rocks the chassis; ramping in and out keeps the turn planted.
        """
        dt = self.target_period
        accel = self.body_yaw_accel

        if self.body_yaw_limit is not None:
            self._body_yaw_target = float(np.clip(
                self._body_yaw_target, -self.body_yaw_limit, self.body_yaw_limit
            ))

        if self._body_yaw_vel_cmd is not None:
            desired = float(np.clip(
                self._body_yaw_vel_cmd, -self.body_yaw_rate, self.body_yaw_rate
            ))
            # Start braking before the hard stop rather than slamming into it
            if self.body_yaw_limit is not None and desired * self._body_yaw_current > 0:
                room = max(0.0, self.body_yaw_limit - abs(self._body_yaw_current))
                stop_speed = float(np.sqrt(2.0 * accel * room))
                desired = float(np.clip(desired, -stop_speed, stop_speed))
        else:
            desired = float(_approach_speed(
                self._body_yaw_target - self._body_yaw_current,
                self.body_yaw_rate,
                accel,
                dt,
            ))

        max_dv = accel * dt
        self._body_yaw_vel += float(np.clip(desired - self._body_yaw_vel, -max_dv, max_dv))
        self._body_yaw_current += self._body_yaw_vel * dt

        if self.body_yaw_limit is not None:
            clamped = float(np.clip(
                self._body_yaw_current, -self.body_yaw_limit, self.body_yaw_limit
            ))
            if clamped != self._body_yaw_current:
                self._body_yaw_current = clamped
                self._body_yaw_vel = 0.0

        if self._body_yaw_vel_cmd is not None:
            # Velocity mode owns the base: keep the position target with it
            # so the "yield to explicit rotations" check can't self-trigger
            self._body_yaw_target = self._body_yaw_current

        if abs(self._body_yaw_vel) > 1e-4:
            # Motor noise lingers briefly after motion stops
            self._base_active_until = self._now() + 0.4
        return self._body_yaw_current

    def _clamp_head_pose(self, head: NDArray) -> NDArray:
        """Keep the composed head pose inside the safe mechanical envelope.

        Yaw beyond the neck range makes the daemon reject the whole target
        (silent full freeze); pitch/roll/z extremes drive the head shell
        into the body with a loud bump. Offsets stack additively, so the
        clamp guards the SUM of all sources.
        """
        euler = R.from_matrix(head[:3, :3]).as_euler("xyz")
        clamped = [
            float(np.clip(euler[0], -HEAD_ROLL_LIMIT, HEAD_ROLL_LIMIT)),
            float(np.clip(euler[1], HEAD_PITCH_UP_LIMIT, HEAD_PITCH_DOWN_LIMIT)),
            float(np.clip(euler[2], -NECK_YAW_LIMIT, NECK_YAW_LIMIT)),
        ]
        if not np.allclose(clamped, euler, atol=1e-9):
            head[:3, :3] = R.from_euler("xyz", clamped).as_matrix()
        head[2, 3] = float(np.clip(head[2, 3], HEAD_Z_MIN, HEAD_Z_MAX))
        return head

    def _smooth_head_pose(self, head: NDArray, current_time: float) -> NDArray:
        """Serve the composed pose under the speed/acceleration envelope.

        Upstream stages are individually smooth but their *transitions* are
        not: a move starting, a face being reacquired across the frame, the
        scan sweep handing over to tracking. Each of those is a step, and a
        step at 100Hz is a full-torque impulse into the chassis. Everything
        leaves through here so none of them can reach the motors as one.
        """
        euler = R.from_matrix(head[:3, :3]).as_euler("xyz")
        translation = np.asarray(head[:3, 3], dtype=float)

        self._resync_after_external_move(current_time)

        rotation = self._head_rot_limiter.step(euler)
        self._head_rot_limiter.clamp_pos(
            [-HEAD_ROLL_LIMIT, HEAD_PITCH_UP_LIMIT, -NECK_YAW_LIMIT],
            [HEAD_ROLL_LIMIT, HEAD_PITCH_DOWN_LIMIT, NECK_YAW_LIMIT],
        )
        rotation = self._head_rot_limiter.pos
        position = self._head_pos_limiter.step(translation)

        smoothed = np.eye(4)
        smoothed[:3, :3] = R.from_euler("xyz", rotation).as_matrix()
        smoothed[:3, 3] = position
        return smoothed

    def _resync_after_external_move(self, current_time: float) -> None:
        """Re-seed the smoother when something else has moved the robot.

        The daemon ignores our set_target while it plays a recorded move of
        its own (that is how the emotion/dance tools work), so control comes
        back with the head somewhere we never commanded. Left alone, the
        next tick would command the whole difference at once. Checked only
        while our own output is at rest, so ordinary servo lag during a
        commanded move can never be mistaken for a takeover.
        """
        if RESYNC_TOLERANCE <= 0 or current_time - self._last_resync_poll < RESYNC_POLL:
            return
        self._last_resync_poll = current_time

        if not (
            self._head_rot_limiter.at_rest(np.deg2rad(20.0))
            and abs(self._body_yaw_vel) < np.deg2rad(10.0)
        ):
            self._diverged_since = None
            return

        measured = self._read_relative_pose()
        if measured is None:
            return
        base_yaw, measured_euler, translation = measured
        drift = float(np.max(np.abs(
            _wrap_angles(measured_euler - self._head_rot_limiter.pos)
        )))

        if drift < RESYNC_TOLERANCE:
            self._diverged_since = None
            return
        if self._diverged_since is None:
            self._diverged_since = current_time
            return
        if current_time - self._diverged_since < RESYNC_HOLD:
            return

        self._diverged_since = None
        logger.info(
            "Head %.0f° from commanded pose (external move?); easing back in",
            float(np.rad2deg(drift)),
        )
        self._seed_state(base_yaw, measured_euler, translation)

    def _read_relative_pose(self) -> Optional[Tuple[float, NDArray, NDArray]]:
        """Read the robot's real pose as (base yaw, body-relative euler, xyz).

        The measured head pose is world-frame (FK over all seven joints),
        so the base yaw is undone to make it comparable with the
        body-relative pose this loop composes.
        """
        try:
            joints, _ = self.current_robot.get_current_joint_positions()
            measured = np.asarray(
                self.current_robot.get_current_head_pose(), dtype=float
            )
        except Exception as e:
            logger.debug("Pose read failed: %s", e)
            return None
        base_yaw = float(joints[0])
        relative = self._rotate_head_by_base_yaw(measured, -base_yaw)
        euler = R.from_matrix(relative[:3, :3]).as_euler("xyz")
        return base_yaw, euler, np.asarray(relative[:3, 3], dtype=float)

    def _seed_state(
        self, base_yaw: float, euler: NDArray, translation: NDArray
    ) -> None:
        """Point the smoother and base tracking at a known real pose."""
        self._head_rot_limiter.reset(euler)
        self._head_pos_limiter.reset(translation)
        self._body_yaw_current = base_yaw
        self._body_yaw_target = base_yaw
        self._body_yaw_vel = 0.0
        self._body_yaw_vel_cmd = None

    def _reseed_from_measured(self) -> bool:
        """Re-seed from wherever the robot actually is right now.

        Used when control returns from something that drove the robot
        directly (goto_sleep / wake_up / a daemon-played move), so the
        first tick back eases out of the real pose instead of stepping
        from the one this loop was holding before it let go.
        """
        measured = self._read_relative_pose()
        if measured is None:
            return False
        self._seed_state(*measured)
        return True

    def _rotate_head_by_base_yaw(self, head: NDArray, base_yaw: float) -> NDArray:
        """Express the body-relative head pose in the daemon's world frame."""
        if base_yaw == 0.0:
            return head
        rot = R.from_euler("z", base_yaw).as_matrix()
        rotated = head.copy()
        rotated[:3, :3] = rot @ head[:3, :3]
        rotated[:3, 3] = rot @ head[:3, 3]
        return rotated

    def _issue_command(self, head: NDArray, antennas: Tuple[float, float], body_yaw: float) -> None:
        """Send command to robot."""
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            self._last_commanded_pose = (head.copy(), antennas, body_yaw)
        except Exception as e:
            now = self._now()
            if now - self._last_cmd_error_log > 5.0:
                self._last_cmd_error_log = now
                logger.warning("set_target failed: %s", e)
            else:
                logger.debug("set_target failed: %s", e)
            
    def _publish_shared_state(self) -> None:
        """Update shared state for external queries."""
        with self._shared_lock:
            self._shared_last_activity = self.state.last_activity_time
            self._shared_is_listening = self._is_listening
            self._shared_body_yaw = (self._body_yaw_current, self._body_yaw_target)
            self._shared_base_active_until = self._base_active_until
            self._shared_suspended = self._suspended
            if self.state.last_primary_pose is not None:
                self._shared_primary_pose = clone_pose(self.state.last_primary_pose)
            
    def start(self) -> None:
        """Start the control loop thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MovementManager already running")
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("MovementManager started")
        
    def stop(self) -> None:
        """Stop the control loop and reset to neutral."""
        if self._thread is None or not self._thread.is_alive():
            return
            
        logger.info("Stopping MovementManager...")
        self.clear_move_queue()
        
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        
        # Reset to neutral. Unwind the base to the nearest full turn rather
        # than absolute zero: head-pose interpolation wraps at +/-180 deg, so
        # a long unwind would transiently exceed the neck range.
        try:
            neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            reset_yaw = 2.0 * np.pi * round(self._body_yaw_current / (2.0 * np.pi))
            self.current_robot.goto_target(
                head=neutral,
                antennas=[0.0, 0.0],
                duration=2.0,
                body_yaw=reset_yaw,
            )
            logger.info("Reset to neutral position")
        except Exception as e:
            logger.error("Failed to reset: %s", e)
            
    def _run_loop(self) -> None:
        """Main control loop at 100Hz."""
        logger.debug("Starting 100Hz control loop")
        
        while not self._stop_event.is_set():
            loop_start = self._now()
            
            # Process signals
            self._poll_signals(loop_start)

            if self._suspended:
                # Someone else owns the robot; compose nothing, send nothing
                self._publish_shared_state()
                elapsed = self._now() - loop_start
                time.sleep(max(0.0, self.target_period - elapsed))
                continue

            # Manage moves
            self._manage_move_queue(loop_start)
            self._manage_breathing(loop_start)
            
            # Update face tracking offsets from camera worker
            self._update_face_tracking(loop_start)

            # Recruit the base when the head alone can't keep the face in view
            self._update_body_follow(loop_start)

            # Update thinking animation offsets
            self._update_thinking_offsets(loop_start)
            
            # Compose the body-relative pose; moves carry transient yaw, the
            # persistent base yaw (turn_body/body_sway/body-follow) is added
            # on top. Clamp the neck twist so IK stays solvable, hold the
            # result inside the motion envelope, then express the head in
            # the world frame the daemon expects.
            head, antennas, body_yaw = self._compose_pose(loop_start)
            head = self._clamp_head_pose(head)
            head = self._smooth_head_pose(head, loop_start)
            base_yaw = self._advance_body_yaw()
            head = self._rotate_head_by_base_yaw(head, base_yaw)
            body_yaw += base_yaw

            # Blend antennas for listening
            antennas = self._blend_antennas(antennas)
            
            # Send to robot
            self._issue_command(head, antennas, body_yaw)
            
            # Update shared state
            self._publish_shared_state()
            
            # Maintain timing
            elapsed = self._now() - loop_start
            sleep_time = max(0.0, self.target_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        logger.debug("Control loop stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status for debugging."""
        return {
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "breathing_active": self._breathing_active,
            "processing": self._processing,
            "thinking_amplitude": round(self._thinking_amplitude, 3),
            "last_commanded_pose": {
                "head": self._last_commanded_pose[0].tolist(),
                "antennas": self._last_commanded_pose[1],
                "body_yaw": self._last_commanded_pose[2],
            },
        }
