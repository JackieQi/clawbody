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


# Body-follow face tracking (wireless base): when the head must yaw far to
# keep a tracked face in view, the base rotates underneath it so tracking
# continues past the head's range (the base yaw is continuous 360°). When
# the face is lost and the head-only scan comes up empty, the base joins
# the search by slowly turning toward where the face was last seen.
BODY_FOLLOW_ENABLED = _env_flag("CLAWBODY_BODY_FOLLOW")
# Head yaw (degrees) that engages the base; hysteresis releases it near center
BODY_FOLLOW_START_DEG = float(os.getenv("CLAWBODY_BODY_FOLLOW_START", "12") or 12)
BODY_FOLLOW_STOP_DEG = 3.0
# Proportional gain (rad/s of base speed per rad of head yaw) and speed cap
BODY_FOLLOW_GAIN = float(os.getenv("CLAWBODY_BODY_FOLLOW_GAIN", "2.5") or 2.5)
BODY_FOLLOW_MAX_SPEED = float(np.deg2rad(
    float(os.getenv("CLAWBODY_BODY_FOLLOW_MAX_SPEED", "60") or 60)))
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

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]
SpeechOffsets = Tuple[float, float, float, float, float, float]


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
        
        # Breathing parameters
        self.breathing_z_amplitude = 0.005  # 5mm gentle movement
        self.breathing_frequency = 0.1  # Hz
        self.antenna_sway_amplitude = np.deg2rad(15)  # degrees
        self.antenna_frequency = 0.5  # Hz
        
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
        self.start_pose = start_pose
        self.start_antennas = np.array(start_antennas)
        self._duration = duration
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
        # tick and added to every command, so the base can rotate (up to a
        # full 360°) while head moves/offsets compose on top
        self._body_yaw_current = 0.0
        self._body_yaw_target = 0.0
        self.body_yaw_rate = float(np.deg2rad(120.0))  # max slew speed, rad/s
        # Hard range of the base motor; None only for tests
        self.body_yaw_limit: Optional[float] = BODY_YAW_RANGE
        self._last_cmd_error_log = float("-inf")

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

    def get_body_yaw(self) -> Tuple[float, float]:
        """Get (current, target) base body yaw in radians. Thread-safe."""
        with self._shared_lock:
            return self._shared_body_yaw

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
            speed = float(np.clip(
                BODY_FOLLOW_GAIN * yaw_off,
                -BODY_FOLLOW_MAX_SPEED,
                BODY_FOLLOW_MAX_SPEED,
            ))
            # Keep the target glued just ahead of current so the slew in
            # _advance_body_yaw moves at exactly this speed with no windup
            self._body_yaw_target = self._body_yaw_current + speed * self.target_period
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
        self._body_yaw_target = self._body_yaw_current + self._last_face_side * step

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
        
        # Antenna offsets: asymmetric scan (phase offset creates "searching" feel)
        # ±20° at 0.4 Hz, right antenna lags left by ~70° of phase
        left_ant = amp * np.deg2rad(20) * np.sin(2 * np.pi * 0.4 * t)
        right_ant = amp * np.deg2rad(20) * np.sin(2 * np.pi * 0.4 * t + 1.2)
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
            self._external_yaw_cmd_time = current_time
            self.state.update_activity()
            logger.info("Body yaw target: %.0f°", float(np.rad2deg(self._body_yaw_target)))
        elif cmd == "halt_body_yaw":
            self._body_yaw_target = self._body_yaw_current
            self._external_yaw_cmd_time = current_time
                
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
                    _, current_ant = self.current_robot.get_current_joint_positions()
                    current_head = self.current_robot.get_current_head_pose()
                    
                    breathing = BreathingMove(
                        interpolation_start_pose=current_head,
                        interpolation_start_antennas=current_ant,
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
        """Slew the persistent base yaw toward its target; return current value."""
        if self.body_yaw_limit is not None:
            self._body_yaw_target = float(np.clip(
                self._body_yaw_target, -self.body_yaw_limit, self.body_yaw_limit
            ))
        delta = self._body_yaw_target - self._body_yaw_current
        max_step = self.body_yaw_rate * self.target_period
        if abs(delta) <= max_step:
            self._body_yaw_current = self._body_yaw_target
        else:
            self._body_yaw_current += max_step if delta > 0 else -max_step
        if delta != 0.0:
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
            # on top. Clamp the neck twist so IK stays solvable, then express
            # the head in the world frame the daemon expects.
            head, antennas, body_yaw = self._compose_pose(loop_start)
            head = self._clamp_head_pose(head)
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
