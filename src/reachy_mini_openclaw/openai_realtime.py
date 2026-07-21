"""ClawBody - OpenAI Realtime API handler with OpenClaw identity.

This module implements ClawBody's voice conversation system using OpenAI Realtime API
with the robot embodying the actual OpenClaw agent's personality and context.

Architecture:
    Startup: Fetch OpenClaw agent context (personality, memories, user info)
    Runtime: User speaks -> OpenAI Realtime (as OpenClaw agent) -> Robot speaks
             -> Tools for movements + OpenClaw queries for extended capabilities
             -> Conversations synced back to OpenClaw for memory continuity

The robot IS the OpenClaw agent - same personality, same memories, same context.
"""

import os
import re
import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Literal, Optional, Tuple
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from reachy_mini_openclaw.config import config
from reachy_mini_openclaw.prompts import get_session_voice
from reachy_mini_openclaw.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call

logger = logging.getLogger(__name__)

# OpenAI Realtime API audio format
OPENAI_SAMPLE_RATE: Final[Literal[24000]] = 24000

# Gesture mode for speech-synchronized head gestures:
# - "natural" (default): turn-level gestures at response start + live
#   keyword-triggered gestures while speaking
# - "turn": turn-level gestures only
# - "off": no automatic gestures
GESTURE_MODE = os.getenv("CLAWBODY_GESTURE_MODE", "natural").strip().lower()

# Echo defense: while robot audio is playing, mic frames quieter than this
# RMS (float scale, 0..1) are dropped so the robot doesn't hear its own
# speaker and spawn phantom turns. Louder speech still passes, so the user
# can barge in. 0 disables the gate entirely. The same gate applies while
# the base motor is slewing (its noise reaches the mics on the chassis).
BARGE_IN_RMS = float(os.getenv("CLAWBODY_BARGE_IN_RMS", "0.06") or 0.0)

# Server VAD tuning. The mics share a chassis with the speaker and motors,
# so the default threshold errs high; lower it if the robot has trouble
# hearing you, raise it if phantom turns persist (0..1, OpenAI default 0.5).
VAD_THRESHOLD = float(os.getenv("CLAWBODY_VAD_THRESHOLD", "0.8") or 0.8)
VAD_SILENCE_MS = int(float(os.getenv("CLAWBODY_VAD_SILENCE_MS", "700") or 700))

# Always-on mic noise floor: frames below this RMS never reach the server,
# so hum/servo whine can't become phantom turns. 0 disables.
MIC_FLOOR_RMS = float(os.getenv("CLAWBODY_MIC_FLOOR_RMS", "0.02") or 0.0)

# Log mic RMS statistics every N seconds (0 disables); used to diagnose
# what the VAD is actually hearing on this chassis.
MIC_STATS_INTERVAL = float(os.getenv("CLAWBODY_MIC_STATS_S", "5") or 0.0)

# Sound-source orientation: when the user starts speaking, read Direction of
# Arrival from the mic array and turn the head toward the voice.
# The ReSpeaker reports 0=left, pi/2=front, pi=right across a 180° arc.
DOA_MODE = os.getenv("CLAWBODY_DOA_MODE", "on").strip().lower()
# Set to 1 if the robot turns away from you instead of toward you
DOA_FLIP = os.getenv("CLAWBODY_DOA_FLIP", "0").strip() in ("1", "true", "yes")
DOA_MAX_YAW_DEG = 35.0
DOA_DEADBAND_DEG = 8.0

# Base instructions for the robot body capabilities
ROBOT_BODY_INSTRUCTIONS = """
## Your Robot Body (Reachy Mini)
You are currently embodied in a Reachy Mini robot. You have physical capabilities:

**Movement Tools (use these naturally during conversation):**
- `look` - Move head to look left, right, up, down, or front (center)
- `emotion` - Express an emotion through movement. Accepts any string; unknown names fall back to built-in macros.
- `dance` - Perform a dance animation. Accepts any string; uses installed dance libraries when available, otherwise falls back to macros.
- `turn_body` - Rotate your body/base by degrees ("turn around" = 180, full spin = 360; positive = left)
- `body_sway` - Sway your body left-right for expressive emphasis
- `capabilities` - List available dances/emotions detected at runtime
- `camera` - Capture what you see through your camera
- `face_tracking` - Enable/disable automatic face tracking

**Important:** If the user asks you to "list dances" or "what emotions/dances are available", call the local `capabilities` tool directly. Do NOT forward that request via `ask_openclaw`.

**How to Use Your Body:**
- Look around while thinking or to emphasize points
- Express emotions that match what you're saying
- Dance when celebrating good news
- Use the camera when asked "what do you see?"
- Reference your body naturally ("let me look", "I can see...")

**Conversation Style for Voice:**
- Keep responses concise - you're speaking out loud, not typing
- Use natural speech patterns ("hmm", "well", "let me see")
- Be warm, personable, and conversational

## ⚠️ CRITICAL: When to Use ask_openclaw (MANDATORY)

You are the robot body, but your BRAIN lives in OpenClaw.
For anything beyond movement and camera, you MUST use `ask_openclaw`.

**ALWAYS use ask_openclaw for:**
- 📧 Sending emails or messages
- 🌤️ Weather lookups
- 📅 Calendar and schedule queries
- 🔍 Web searches and news lookups
- 🧠 Accessing memories (past conversations, contacts, notes)
- 🏠 Smart home control
- 💰 Crypto wallet, blockchain, or financial queries
- 📝 Creating, modifying, or deleting anything (files, reminders, etc.)
- ❓ Any factual question you're not 100% certain about
- 🔧 Any task requiring external tools or APIs

**HARD RULE — External Effects:**
If the user asks you to DO something that produces an external effect
(send, create, modify, delete, purchase, post, schedule, etc.),
you MUST use `ask_openclaw` to actually execute it.
DO NOT pretend you did it. DO NOT say "I've sent it" without calling the tool.
DO NOT say "I can't do that" — instead, use ask_openclaw and let your cloud brain handle it.

**HARD RULE — Information Queries:**
If the user asks about weather, stock prices, news, emails, contacts, memories,
or anything requiring real-time or stored data, you MUST use `ask_openclaw`.
DO NOT make up answers. DO NOT say "I don't have access" — ask_openclaw DOES.

**When NOT to use ask_openclaw (handle locally):**
- Robot movement (look, emotion, dance, turn_body, body_sway)
- Camera capture
- Face tracking toggle
- Listing capabilities (use local `capabilities` tool)
- Simple chitchat that doesn't need external data
- Responding to greetings

**Fallback Rule:**
If you're unsure whether you can handle something locally, default to
ask_openclaw. Say "let me check" and call the tool. Never say "I can't
do that" or "I don't have access" — your OpenClaw brain has access to
almost everything.
"""

# Fallback if OpenClaw context fetch fails
FALLBACK_IDENTITY = """You are Clawson, the OpenClaw AI assistant. You're a friendly, helpful space lobster 🦞 who loves helping people. You're warm, occasionally witty, and genuinely care about being useful.

You speak as yourself - not as "an AI assistant" but as Clawson, with your own personality and warmth."""

_CJK_RE = re.compile("[\u3000-\u9fff\uf900-\ufaff]")


def _find_cue(text: str, cues: list[str]) -> Optional[str]:
    """Return the first cue found in text, or None.

    CJK cues match as substrings; Latin cues match on word boundaries
    (case-insensitive) so short words like "hi" don't fire inside "this".
    Boundaries are explicit ASCII-alnum lookarounds rather than \\b, which
    counts CJK ideographs as word characters and would stop Latin cues
    matching when adjacent to CJK text (e.g. "你好hi").
    """
    for cue in cues:
        if _CJK_RE.search(cue):
            # re.search instead of `in` so any Latin chars inside a mixed
            # CJK/Latin cue still match case-insensitively
            if re.search(re.escape(cue), text, re.IGNORECASE):
                return cue
        elif re.search(
            rf"(?<![a-zA-Z0-9]){re.escape(cue)}(?![a-zA-Z0-9])", text, re.IGNORECASE
        ):
            return cue
    return None


class OpenAIRealtimeHandler(AsyncStreamHandler):
    """Handler for OpenAI Realtime API embodying the OpenClaw agent.
    
    This handler:
    - Fetches OpenClaw's personality and context at startup
    - Maintains voice conversation AS the OpenClaw agent
    - Executes robot movement tools locally for low latency
    - Calls OpenClaw for extended capabilities (web, calendar, memory)
    - Syncs conversations back to OpenClaw for memory continuity
    """
    
    def __init__(
        self,
        deps: ToolDependencies,
        openclaw_bridge: Optional[Any] = None,
        gradio_mode: bool = False,
    ):
        """Initialize the handler.
        
        Args:
            deps: Tool dependencies for robot control
            openclaw_bridge: Bridge to OpenClaw gateway
            gradio_mode: Whether running with Gradio UI
        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPENAI_SAMPLE_RATE,
            input_sample_rate=OPENAI_SAMPLE_RATE,
        )
        
        self.deps = deps
        self.openclaw_bridge = openclaw_bridge
        self.gradio_mode = gradio_mode
        
        # OpenAI connection
        self.client: Optional[AsyncOpenAI] = None
        self.connection: Any = None
        
        # Output queue
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        
        # State tracking
        self.last_activity_time = 0.0
        self.start_time = 0.0
        self._speaking = False  # True when robot is speaking
        
        # OpenClaw agent context (fetched at startup)
        self._agent_context: Optional[str] = None
        
        # Conversation tracking for sync
        self._last_user_message: Optional[str] = None
        self._last_assistant_response: Optional[str] = None

        # Per-response gesture state (reset on response.created)
        self._gesture_buffer = ""
        self._gesture_processed_len = 0
        self._gesture_fired: dict[str, bool] = {}
        self._gesture_last_t = 0.0

        # Interruption/echo handling state
        self._turn_counter = 0  # increments on each user speech start
        self._current_item_id: Optional[str] = None  # assistant item being spoken
        self._audio_play_start: Optional[float] = None  # wall time first audio of response
        self._audio_enqueued_s = 0.0  # seconds of audio enqueued for current response

        # Mic diagnostics: what the VAD actually hears on this chassis
        self._last_mic_rms = 0.0
        self._mic_stats = [0.0, 0.0, 0, 0]  # [sum, max, frames, dropped]
        self._mic_stats_t = 0.0

        # Strong refs to fire-and-forget tasks (event loop keeps only weak refs)
        self._bg_tasks: set = set()
        
        # Lifecycle flags
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()
        
    def copy(self) -> "OpenAIRealtimeHandler":
        """Create a copy of the handler (required by fastrtc)."""
        return OpenAIRealtimeHandler(self.deps, self.openclaw_bridge, self.gradio_mode)
    
    def _build_tools(self) -> list[dict]:
        """Build the tool list for the session."""
        tools = []
        
        # Robot movement tools (executed locally)
        for spec in get_tool_specs():
            tools.append(spec)
        
        # OpenClaw query tool (for extended capabilities)
        if self.openclaw_bridge is not None:
            tools.append({
                "type": "function",
                "name": "ask_openclaw",
                "description": """Query OpenClaw for information or actions requiring external tools.
Use this for: weather, calendar, web searches, news, smart home control, 
accessing conversation memory, or any task needing external data/tools.
OpenClaw has access to many capabilities you don't have directly.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question or request to send to OpenClaw"
                        },
                        "include_image": {
                            "type": "boolean",
                            "description": "Whether to include current camera image (for 'what do you see' queries)",
                            "default": False
                        }
                    },
                    "required": ["query"]
                }
            })
        
        return tools
        
    async def start_up(self) -> None:
        """Start the handler and connect to OpenAI.
        
        Runs an infinite reconnection loop so the robot stays alive
        even if the WebSocket drops (network blip, idle timeout, etc.).
        """
        api_key = config.OPENAI_API_KEY
        if not api_key:
            logger.error("OPENAI_API_KEY not configured")
            raise ValueError("OPENAI_API_KEY required")
            
        self.client = AsyncOpenAI(api_key=api_key)
        self.start_time = asyncio.get_event_loop().time()
        self.last_activity_time = self.start_time
        
        attempt = 0
        max_backoff = 30  # Cap backoff at 30 seconds
        
        while not self._shutdown_requested:
            attempt += 1
            try:
                await self._run_session()
                # Session ended cleanly (shouldn't normally happen)
                if self._shutdown_requested:
                    return
                # Reset attempt counter on a clean exit
                attempt = 0
            except ConnectionClosedError as e:
                logger.warning("WebSocket closed unexpectedly (attempt %d): %s", attempt, e)
            except Exception as e:
                logger.error("Session error (attempt %d): %s", attempt, e)
            finally:
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass
            
            if self._shutdown_requested:
                return
                
            # Exponential backoff with jitter, capped at max_backoff
            delay = min(max_backoff, (2 ** min(attempt - 1, 5))) + random.uniform(0, 1)
            logger.info("Reconnecting in %.1f seconds...", delay)
            await asyncio.sleep(delay)
                    
    async def _run_session(self) -> None:
        """Run a single OpenAI Realtime session."""
        model = config.OPENAI_MODEL
        logger.info("Connecting to OpenAI Realtime API with model: %s", model)
        
        # Fetch OpenClaw agent context (personality, memories, user info)
        system_instructions = await self._build_system_instructions()
        
        # GA Realtime API (the beta API shape was retired by OpenAI in May 2026)
        async with self.client.realtime.connect(model=model) as conn:
            # Configure session with OpenClaw's identity + robot body capabilities
            tools = self._build_tools()
            
            await conn.session.update(
                session={
                    "type": "realtime",
                    "output_modalities": ["audio"],
                    "instructions": system_instructions,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                            "transcription": {
                                "model": config.OPENAI_TRANSCRIPTION_MODEL,
                            },
                            # Robot mic sits away from the speaker's mouth
                            "noise_reduction": {"type": "far_field"},
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": VAD_THRESHOLD,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": VAD_SILENCE_MS,
                                "create_response": True,
                                "interrupt_response": True,
                            },
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                            "voice": get_session_voice(),
                        },
                    },
                    "tools": tools,
                    "tool_choice": "auto",
                },
            )
            logger.info("OpenAI Realtime session configured with %d tools", len(tools))
            
            self.connection = conn
            self._connected_event.set()
            
            # Process events
            async for event in conn:
                await self._handle_event(event)
    
    async def _build_system_instructions(self) -> str:
        """Build system instructions by fetching OpenClaw's context.
        
        Returns:
            Complete system instructions combining OpenClaw identity + robot capabilities
        """
        # Try to fetch context from OpenClaw
        agent_context = None
        if self.openclaw_bridge and self.openclaw_bridge.is_connected:
            logger.info("Fetching agent context from OpenClaw...")
            agent_context = await self.openclaw_bridge.get_agent_context()
            
        if agent_context:
            self._agent_context = agent_context
            logger.info("Using OpenClaw agent context (%d chars)", len(agent_context))
            # Combine OpenClaw's identity/context with robot body instructions
            return f"""{agent_context}

{ROBOT_BODY_INSTRUCTIONS}"""
        else:
            logger.warning("Could not fetch OpenClaw context, using fallback identity")
            return f"""{FALLBACK_IDENTITY}

{ROBOT_BODY_INSTRUCTIONS}"""
                
    async def _handle_event(self, event: Any) -> None:
        """Handle an event from the OpenAI Realtime API."""
        event_type = event.type
        
        # Speech detection
        if event_type == "input_audio_buffer.speech_started":
            # User started speaking - the new turn takes priority
            self._turn_counter += 1
            was_speaking = self._speaking
            self._speaking = False
            self.deps.movement_manager.set_processing(False)

            # Tell the server where playback actually stopped, so its
            # conversation history matches what the user heard instead of
            # the full generated response
            if was_speaking and self._current_item_id and self._audio_play_start is not None:
                now = asyncio.get_event_loop().time()
                played_ms = int(
                    max(0.0, min(now - self._audio_play_start, self._audio_enqueued_s)) * 1000
                )
                try:
                    await self.connection.conversation.item.truncate(
                        item_id=self._current_item_id,
                        content_index=0,
                        audio_end_ms=played_ms,
                    )
                except Exception as e:
                    logger.debug("Truncate failed: %s", e)
                # Collapse the playback estimate: the local queue is flushed
                # below, so only the short device tail remains
                self._audio_enqueued_s = max(0.0, now - self._audio_play_start)

            # Flush un-played audio and stale gestures from the old turn
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self.deps.movement_manager.clear_move_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            logger.info("User started speaking (mic RMS %.4f)", self._last_mic_rms)

            # Turn the head toward the voice (Direction of Arrival)
            if DOA_MODE == "on":
                task = asyncio.create_task(self._turn_toward_voice())
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)
            
        if event_type == "input_audio_buffer.speech_stopped":
            self.deps.movement_manager.set_listening(False)
            logger.info("User stopped speaking")
            
        # Transcription (for logging, UI, and sync)
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
            if transcript and transcript.strip():
                logger.info("User: %s", transcript)
                self._last_user_message = transcript  # Track for sync
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": transcript})
                )
            
        # Response started - robot is about to speak
        if event_type == "response.created":
            self._speaking = True
            # Reset per-response gesture state
            self._gesture_buffer = ""
            self._gesture_processed_len = 0
            self._gesture_fired = {}
            self._gesture_last_t = 0.0
            # Reset playback tracking for the new response
            self._current_item_id = None
            self._audio_play_start = None
            self._audio_enqueued_s = 0.0
            logger.debug("Response started")
            if GESTURE_MODE in ("natural", "turn"):
                try:
                    await self._trigger_turn_gesture(self._last_user_message)
                except Exception as e:
                    logger.debug("Turn gesture failed: %s", e)
            
        # Audio output from TTS (GA event name; was response.audio.delta in beta)
        if event_type == "response.output_audio.delta":
            # Audio arriving means we have a response - stop thinking animation
            self.deps.movement_manager.set_processing(False)
            
            # Feed to head wobbler for expressive movement
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(event.delta)
            
            self.last_activity_time = asyncio.get_event_loop().time()
            
            # Queue audio for playback
            audio_data = np.frombuffer(
                base64.b64decode(event.delta),
                dtype=np.int16
            ).reshape(1, -1)
            await self.output_queue.put((OPENAI_SAMPLE_RATE, audio_data))

            # Track playback progress for truncation and the echo gate
            item_id = getattr(event, "item_id", None)
            if isinstance(item_id, str):
                self._current_item_id = item_id
            if self._audio_play_start is None:
                self._audio_play_start = asyncio.get_event_loop().time()
            self._audio_enqueued_s += audio_data.shape[-1] / OPENAI_SAMPLE_RATE
            
        # Response text (for logging, UI, and live gesture triggers)
        if event_type == "response.output_audio_transcript.delta":
            # Streaming transcript of what's being said (while audio plays)
            if GESTURE_MODE == "natural":
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    try:
                        await self._on_assistant_transcript_delta(delta)
                    except Exception as e:
                        logger.debug("Live gesture failed: %s", e)
            
        if event_type == "response.output_audio_transcript.done":
            response_text = event.transcript
            logger.info("Assistant: %s", response_text[:100] if len(response_text) > 100 else response_text)
            self._last_assistant_response = response_text  # Track for sync
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": response_text})
            )
            
        # Response completed - sync conversation to OpenClaw
        if event_type == "response.done":
            self._speaking = False
            self.deps.movement_manager.set_processing(False)
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            logger.debug("Response completed")
            
            # Sync conversation to OpenClaw for memory continuity
            await self._sync_to_openclaw()
            
        # Tool calls
        if event_type == "response.function_call_arguments.done":
            await self._handle_tool_call(event)
            
        # Errors
        if event_type == "error":
            err = getattr(event, "error", None)
            msg = getattr(err, "message", str(err))
            code = getattr(err, "code", "")
            logger.error("OpenAI error [%s]: %s", code, msg)
            
    async def _handle_tool_call(self, event: Any) -> None:
        """Handle a tool call from OpenAI."""
        tool_name = getattr(event, "name", None)
        args_json = getattr(event, "arguments", None)
        call_id = getattr(event, "call_id", None)
        
        if not isinstance(tool_name, str) or not isinstance(args_json, str):
            return
            
        logger.info("Tool call: %s(%s)", tool_name, args_json[:50] if len(args_json) > 50 else args_json)

        # Start thinking animation while we process the tool call.
        # It will stop when the next audio delta arrives or response completes.
        self.deps.movement_manager.set_processing(True)
        turn_at_start = self._turn_counter

        try:
            if tool_name == "ask_openclaw":
                result = await self._handle_openclaw_query(args_json)
            else:
                # Robot movement tools - dispatch locally
                result = await dispatch_tool_call(tool_name, args_json, self.deps)
                
            # Log tool results at INFO when relevant (helps debugging on robot)
            if isinstance(result, dict) and result.get("error"):
                logger.warning("Tool '%s' error: %s", tool_name, result.get("error"))
            elif tool_name != "ask_openclaw":
                logger.info("Tool '%s' result: %s", tool_name, str(result)[:200])
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            result = {"error": str(e)}
            
        # Send result back to continue the conversation
        if isinstance(call_id, str) and self.connection:
            await self.connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                }
            )
            # Trigger response generation after tool result — unless the user
            # started a new turn while the tool ran, in which case forcing a
            # response now would answer the stale question over the new one.
            # The result stays in history for the next turn to use.
            if self._turn_counter == turn_at_start:
                await self.connection.response.create()
            else:
                logger.info(
                    "Skipping response for stale tool '%s' result (user started a new turn)",
                    tool_name,
                )
            
    async def _sync_to_openclaw(self) -> None:
        """Sync the last conversation turn to OpenClaw for memory continuity."""
        if not self.openclaw_bridge or not self.openclaw_bridge.is_connected:
            return
            
        if self._last_user_message and self._last_assistant_response:
            try:
                await self.openclaw_bridge.sync_conversation(
                    self._last_user_message,
                    self._last_assistant_response
                )
                # Clear after sync
                self._last_user_message = None
                self._last_assistant_response = None
            except Exception as e:
                logger.debug("Failed to sync conversation: %s", e)
    
    async def _handle_openclaw_query(self, args_json: str) -> dict:
        """Handle a query to OpenClaw."""
        if self.openclaw_bridge is None:
            return {
                "error": "OpenClaw bridge is not initialized. "
                "Tell the user you cannot reach your backend right now and to try again later."
            }
        if not self.openclaw_bridge.is_connected:
            # Try to reconnect once
            logger.info("OpenClaw bridge disconnected, attempting reconnect...")
            try:
                connected = await self.openclaw_bridge.connect()
                if not connected:
                    return {
                        "error": "OpenClaw gateway is temporarily unreachable. "
                        "Tell the user your backend connection is down and to try again in a moment."
                    }
            except Exception as e:
                logger.error("OpenClaw reconnect failed: %s", e)
                return {
                    "error": "OpenClaw gateway reconnection failed. "
                    "Tell the user your backend is temporarily unavailable."
                }
            
        try:
            args = json.loads(args_json)
            query = args.get("query", "")
            include_image = args.get("include_image", False)
            
            # Capture image if requested
            image_b64 = None
            if include_image and self.deps.camera_worker:
                frame = self.deps.camera_worker.get_latest_frame()
                if frame is not None:
                    import cv2
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    image_b64 = base64.b64encode(buffer).decode('utf-8')
                    logger.debug("Captured camera image for OpenClaw query")
            
            # Query OpenClaw — this may take a while if the backend LLM is slow
            logger.info("Sending ask_openclaw query: %s", query[:80])
            response = await self.openclaw_bridge.chat(
                query, 
                image_b64=image_b64,
                system_context="User is asking through their Reachy Mini robot. Keep response concise for voice.",
            )
            
            if response.error:
                logger.warning("OpenClaw query error: %s", response.error)
                if "timeout" in response.error.lower():
                    return {
                        "error": "The request to OpenClaw timed out — the backend is taking too long. "
                        "Tell the user you're having trouble reaching your backend and to try again."
                    }
                return {
                    "error": f"OpenClaw returned an error: {response.error}. "
                    "Tell the user there was a problem processing their request."
                }
            
            if not response.content:
                return {
                    "error": "OpenClaw returned an empty response. "
                    "Tell the user you got no data back and to try again."
                }
            
            return {"response": response.content}
            
        except Exception as e:
            logger.error("OpenClaw query failed: %s", e)
            return {
                "error": f"OpenClaw query failed: {e}. "
                "Tell the user there was a technical issue reaching your backend."
            }

    async def _trigger_turn_gesture(self, user_text: Optional[str]) -> None:
        """Trigger a small, natural gesture at the start of a response.

        Turn-level (not word-aligned): should feel conversational.
        """
        if not user_text:
            return
        t = str(user_text)

        # Greetings -> friendly wave-ish
        if _find_cue(t, ["哈囉", "你好", "嗨", "안녕", "hello", "hi", "hey", "good morning", "good evening"]):
            await self._queue_headlook_sequence(["right", "left", "front"], [0.22, 0.22, 0.45])
            return

        # Weather -> look up
        if _find_cue(t, ["天氣", "下雨", "溫度", "幾度", "氣象", "weather", "rain", "temperature", "forecast", "sunny", "snow"]):
            await self._queue_headlook_sequence(["up", "front"], [0.35, 0.65])
            return

        # News/accidents -> look down (serious)
        if _find_cue(t, ["新聞", "車禍", "死亡", "受傷", "意外", "災", "地震", "火災", "news", "accident", "earthquake", "injured"]):
            await self._queue_headlook_sequence(["down", "front"], [0.45, 0.75])
            return

        # Thanks -> nod
        if _find_cue(t, ["謝", "thanks", "thank you", "thx"]):
            await self._queue_headlook_sequence(["down", "up", "front"], [0.22, 0.22, 0.45])
            return

        # Questions -> curious glance
        if ("?" in t) or ("？" in t) or t.strip().endswith(("嗎", "呢")) or _find_cue(t, ["是不是", "會不會", "可不可以"]):
            await self._queue_headlook_sequence(["right", "front"], [0.30, 0.55])
            return

    async def _on_assistant_transcript_delta(self, delta: str) -> None:
        """Fire small head gestures from language cues while speaking.

        Simplified from upstream PR #2: gestures fire immediately when a cue
        appears in the streaming transcript (rate-limited, most categories
        once per response) instead of being scheduled against audio playback
        position. Voice streaming is never interrupted; moves are queued in
        parallel via the movement manager.
        """
        self._gesture_buffer += delta

        # Cooldown to avoid machine-gun gestures
        now = asyncio.get_event_loop().time()
        if now - self._gesture_last_t < 0.9:
            return

        # Search all text not yet consumed by a fired gesture: consumption
        # stops the same cue re-triggering after the cooldown, and scanning
        # the whole unconsumed region (it's small) means cues that streamed
        # in during a cooldown aren't lost to a fixed-size window
        tail = self._gesture_buffer[self._gesture_processed_len:]

        # 1) Shy / hide face
        if not self._gesture_fired.get("shy"):
            if _find_cue(tail, ["害羞", "不好意思", "別看", "不要看", "shy", "embarrassed"]):
                self._gesture_fired["shy"] = True
                self._gesture_last_t = now
                self._gesture_processed_len = len(self._gesture_buffer)
                await self._queue_headlook_sequence(["down", "front"], [0.8, 1.0])
                return

        # 2) Negative -> shake head
        if not self._gesture_fired.get("neg"):
            if _find_cue(tail, ["不是", "不對", "不行", "沒有", "不要", "不可以", "no", "not", "nope", "never", "cannot"]):
                self._gesture_fired["neg"] = True
                self._gesture_last_t = now
                self._gesture_processed_len = len(self._gesture_buffer)
                await self._queue_headlook_sequence(
                    ["left", "right", "left", "right", "left", "front"],
                    [0.22, 0.22, 0.22, 0.22, 0.22, 0.35],
                )
                return

        # 3) Positive -> nod
        if not self._gesture_fired.get("pos"):
            if _find_cue(tail, ["沒錯", "對", "可以", "好", "同意", "yes", "yeah", "yep", "sure", "exactly", "of course"]):
                self._gesture_fired["pos"] = True
                self._gesture_last_t = now
                self._gesture_processed_len = len(self._gesture_buffer)
                await self._queue_headlook_sequence(
                    ["down", "up", "down", "up", "front"],
                    [0.22, 0.22, 0.22, 0.22, 0.40],
                )
                return

        # 4) Explicit stage directions (assistant narrates the gesture);
        #    may fire more than once per response, limited by the cooldown
        hit = _find_cue(tail, ["搖頭", "shake my head", "點頭", "nod", "彈跳", "跳起來", "bounce", "搖擺", "搖晃", "左右搖", "擺動", "sway"])
        if hit:
            self._gesture_last_t = now
            self._gesture_processed_len = len(self._gesture_buffer)
            if hit in ("搖頭", "shake my head"):
                await self._queue_headlook_sequence(
                    ["left", "right", "left", "right", "left", "front"],
                    [0.22, 0.22, 0.22, 0.22, 0.22, 0.35],
                )
            elif hit in ("點頭", "nod"):
                await self._queue_headlook_sequence(
                    ["down", "up", "down", "up", "front"],
                    [0.22, 0.22, 0.22, 0.22, 0.40],
                )
            elif hit in ("彈跳", "跳起來", "bounce"):
                await self._queue_headlook_sequence(
                    ["down", "up", "down", "front"],
                    [0.20, 0.20, 0.20, 0.35],
                )
            else:
                # Body sway approximated with a head sway; the body_sway tool
                # remains available for explicit requests
                await self._queue_headlook_sequence(
                    ["right", "left", "right", "front"],
                    [0.22, 0.22, 0.22, 0.45],
                )
            return

        # 5) Question -> gentle side glance
        if not self._gesture_fired.get("q"):
            if "?" in tail or "？" in tail or tail.rstrip().endswith(("嗎", "呢")):
                self._gesture_fired["q"] = True
                self._gesture_last_t = now
                self._gesture_processed_len = len(self._gesture_buffer)
                await self._queue_headlook_sequence(
                    ["right", "left", "right", "front"],
                    [0.22, 0.22, 0.22, 0.45],
                )
                return

    async def _turn_toward_voice(self) -> None:
        """Read Direction of Arrival and turn the head toward the speaker.

        Runs as a fire-and-forget task on user speech start. The ReSpeaker
        answers over a USB control read, so it runs in a worker thread to
        keep the event loop free.
        """
        robot = getattr(self.deps, "robot", None)
        media = getattr(robot, "media", None)
        get_doa = getattr(media, "get_DoA", None)
        if not callable(get_doa):
            return

        try:
            doa = await asyncio.to_thread(get_doa)
            if doa is None:
                return
            angle, _speech = doa

            # 0=left, pi/2=front, pi=right -> positive yaw = left (robot frame)
            yaw_deg = float(np.degrees(np.pi / 2 - float(angle)))
            if DOA_FLIP:
                yaw_deg = -yaw_deg
            yaw_deg = max(-DOA_MAX_YAW_DEG, min(DOA_MAX_YAW_DEG, yaw_deg))

            if abs(yaw_deg) < DOA_DEADBAND_DEG:
                return  # already roughly facing the speaker

            from reachy_mini_openclaw.moves import HeadLookMove

            _, current_ant = robot.get_current_joint_positions()
            current_head = robot.get_current_head_pose()
            move = HeadLookMove(
                direction="front",
                start_pose=current_head,
                start_antennas=tuple(current_ant),
                duration=0.5,
                target_yaw_deg=yaw_deg,
            )
            self.deps.movement_manager.queue_move(move)
            logger.info("Turning toward voice: DoA %.2f rad -> yaw %+.0f°", angle, yaw_deg)
        except Exception as e:
            logger.debug("DoA turn failed: %s", e)

    async def _queue_headlook_sequence(self, directions: list[str], durations: list[float]) -> None:
        """Queue a short sequence of HeadLookMove moves.

        Each move's start pose is chained to the previous move's target so
        back-to-back queued moves stay continuous instead of snapping back
        to the pose the robot had at queue time.
        """
        from reachy_mini_openclaw.moves import HeadLookMove

        if not getattr(self.deps, "robot", None):
            return

        prev_move = None
        for i, direction in enumerate(directions):
            duration = durations[i] if i < len(durations) else durations[-1]
            try:
                if prev_move is None:
                    _, current_ant = self.deps.robot.get_current_joint_positions()
                    start_pose = self.deps.robot.get_current_head_pose()
                    start_antennas = tuple(current_ant)
                else:
                    start_pose = prev_move.target_pose
                    start_antennas = tuple(prev_move.target_antennas)
                move = HeadLookMove(
                    direction=direction,
                    start_pose=start_pose,
                    start_antennas=start_antennas,
                    duration=float(duration),
                )
                self.deps.movement_manager.queue_move(move)
                prev_move = move
            except Exception:
                # If pose read fails, skip gracefully
                return

    def _robot_audio_playing(self) -> bool:
        """Best-effort: is robot speech currently audible from the speaker?"""
        if self._speaking:
            return True
        if self._audio_play_start is None:
            return False
        now = asyncio.get_event_loop().time()
        # 1.0s grace covers the device-side pipeline tail plus room reverb
        return now < self._audio_play_start + self._audio_enqueued_s + 1.0

    def _mic_stats_update(self, rms: float, dropped: bool) -> None:
        """Accumulate mic RMS stats and periodically log them."""
        if MIC_STATS_INTERVAL <= 0.0:
            return
        s = self._mic_stats
        s[0] += rms
        s[1] = max(s[1], rms)
        s[2] += 1
        s[3] += int(dropped)
        now = asyncio.get_event_loop().time()
        if self._mic_stats_t == 0.0:
            self._mic_stats_t = now
            return
        if now - self._mic_stats_t >= MIC_STATS_INTERVAL and s[2] > 0:
            logger.info(
                "Mic stats: avg RMS %.4f, max %.4f, %d frames, %d dropped (playing=%s base=%s)",
                s[0] / s[2], s[1], s[2], s[3],
                self._robot_audio_playing(), self._base_in_motion(),
            )
            self._mic_stats = [0.0, 0.0, 0, 0]
            self._mic_stats_t = now

    def _base_in_motion(self) -> bool:
        """Is the base motor actively slewing (loud enough to fool the VAD)?"""
        try:
            mm = self.deps.movement_manager
            return bool(mm is not None and mm.is_base_active())
        except Exception:
            return False

    async def receive(self, frame: Tuple[int, NDArray]) -> None:
        """Receive audio from the robot microphone."""
        if not self.connection:
            return

        input_sr, audio = frame

        # Handle stereo
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]

        audio = audio.flatten()

        # Convert to float for resampling
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        self._last_mic_rms = rms
        dropped = False

        # Always-on floor: hum and servo whine never reach the server VAD
        if MIC_FLOOR_RMS > 0.0 and rms < MIC_FLOOR_RMS:
            dropped = True
        # Echo/noise gate: while the robot's own speech is playing OR the
        # base motor is slewing, drop quiet mic frames (speaker bleed and
        # motor noise) so the server VAD doesn't spawn phantom user turns;
        # loud speech still passes so the user can interrupt or call out to
        # a searching robot. BARGE_IN_RMS=0 disables the gate.
        elif BARGE_IN_RMS > 0.0 and (self._robot_audio_playing() or self._base_in_motion()):
            if rms < BARGE_IN_RMS:
                dropped = True

        self._mic_stats_update(rms, dropped)
        if dropped:
            return
                
        # Resample to OpenAI sample rate
        if input_sr != OPENAI_SAMPLE_RATE:
            num_samples = int(len(audio) * OPENAI_SAMPLE_RATE / input_sr)
            audio = resample(audio, num_samples).astype(np.float32)
            
        # Convert to int16 for OpenAI
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Send to OpenAI
        try:
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except Exception as e:
            logger.debug("Failed to send audio: %s", e)
            
    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Get the next output (audio or transcript)."""
        return await wait_for_item(self.output_queue)
        
    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
            
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.debug("Connection close: %s", e)
            self.connection = None
            
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
