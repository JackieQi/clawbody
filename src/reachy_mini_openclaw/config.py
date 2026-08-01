"""Configuration management for Reachy Mini OpenClaw.

Handles environment variables and configuration settings for the application.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # OpenAI Configuration
    OPENAI_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv(
        "OPENAI_MODEL", "gpt-realtime-2.1-mini"))
    OPENAI_VOICE: str = field(
        default_factory=lambda: os.getenv("OPENAI_VOICE", "cedar"))
    # Speech-to-text model for the Realtime session's input transcription
    OPENAI_TRANSCRIPTION_MODEL: str = field(default_factory=lambda: os.getenv(
        "OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe-2025-12-15"))
    # Vision model for camera image analysis (cloud image-to-text fallback)
    OPENAI_VISION_MODEL: str = field(
        default_factory=lambda: os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"))

    # OpenClaw Gateway Configuration
    OPENCLAW_GATEWAY_URL: str = field(default_factory=lambda: os.getenv(
        "OPENCLAW_GATEWAY_URL", "ws://localhost:18789"))
    OPENCLAW_TOKEN: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENCLAW_TOKEN"))
    OPENCLAW_AGENT_ID: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_AGENT_ID", "main"))
    # Session key for OpenClaw - uses "main" to share context with WhatsApp and other channels
    # Format: agent:<agent_id>:<session_key>, but we only need the session key part here
    OPENCLAW_SESSION_KEY: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_SESSION_KEY", "main"))

    # Robot Configuration
    ROBOT_NAME: Optional[str] = field(
        default_factory=lambda: os.getenv("ROBOT_NAME"))

    # Feature Flags
    ENABLE_OPENCLAW_TOOLS: bool = field(default_factory=lambda: os.getenv(
        "ENABLE_OPENCLAW_TOOLS", "true").lower() == "true")
    ENABLE_CAMERA: bool = field(default_factory=lambda: os.getenv(
        "ENABLE_CAMERA", "true").lower() == "true")
    ENABLE_FACE_TRACKING: bool = field(default_factory=lambda: os.getenv(
        "ENABLE_FACE_TRACKING", "true").lower() == "true")

    # Face Tracking Configuration
    # Options: "yolo", "mediapipe", or None for auto-detect
    HEAD_TRACKER_TYPE: Optional[str] = field(
        default_factory=lambda: os.getenv("HEAD_TRACKER_TYPE", "yolo"))

    # Local Vision Processing
    ENABLE_LOCAL_VISION: bool = field(default_factory=lambda: os.getenv(
        "ENABLE_LOCAL_VISION", "false").lower() == "true")
    LOCAL_VISION_MODEL: str = field(default_factory=lambda: os.getenv(
        "LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"))
    VISION_DEVICE: str = field(default_factory=lambda: os.getenv(
        "VISION_DEVICE", "auto"))  # "auto", "cuda", "mps", "cpu"
    HF_HOME: str = field(default_factory=lambda: os.getenv(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    # Custom Profile (for personality customization)
    CUSTOM_PROFILE: Optional[str] = field(
        default_factory=lambda: os.getenv("REACHY_MINI_CUSTOM_PROFILE"))

    # ------------------------------------------------------------------
    # Voice input gating
    #
    # The mics hear the whole room, and the transcriber returns text for
    # almost any of it -- including other people, and including wrong-
    # language garbage for speech it can't resolve. These settings decide
    # what actually reaches the model. See voice_gate.py.
    #
    # Set CLAWBODY_VOICE_GATE=false to restore the old behaviour of
    # answering everything.
    # ------------------------------------------------------------------
    VOICE_GATE_ENABLED: bool = field(default_factory=lambda: os.getenv(
        "CLAWBODY_VOICE_GATE", "true").lower() == "true")
    # Require the robot be addressed by name before it answers.
    WAKE_WORD_REQUIRED: bool = field(default_factory=lambda: os.getenv(
        "CLAWBODY_WAKE_WORD_REQUIRED", "true").lower() == "true")
    # Comma-separated. Includes the mistranscriptions the STT actually
    # produces for "Kira", not just the correct spelling.
    WAKE_WORDS: str = field(default_factory=lambda: os.getenv(
        "CLAWBODY_WAKE_WORDS",
        "kira,akira,kiera,kyra,keira,ra,奇拉,基拉"))
    # Seconds after an addressed turn during which follow-ups don't need
    # the name again. 0 = say the name every time.
    WAKE_GRACE_S: float = field(default_factory=lambda: float(
        os.getenv("CLAWBODY_WAKE_GRACE_S", "30") or 30.0))
    # Language families forwarded to the model; anything else is dropped
    # silently. "ja" is excluded on purpose (children in the room).
    VOICE_LANGUAGES: str = field(default_factory=lambda: os.getenv(
        "CLAWBODY_VOICE_LANGUAGES", "en,zh"))
    # Latin transcripts with fewer letters than this are treated as noise.
    VOICE_MIN_CHARS: int = field(default_factory=lambda: int(float(
        os.getenv("CLAWBODY_VOICE_MIN_CHARS", "2") or 2)))
    # Drop bare "mhm"/"ok"/"嗯" instead of answering them.
    SUPPRESS_ACKS: bool = field(default_factory=lambda: os.getenv(
        "CLAWBODY_SUPPRESS_ACKS", "true").lower() == "true")
    # Say "sorry, didn't catch that" on unintelligible input instead of
    # staying silent. Off by default: silence is less annoying than a
    # robot apologising to the room every time a chair scrapes.
    APOLOGIZE_ON_REJECT: bool = field(default_factory=lambda: os.getenv(
        "CLAWBODY_APOLOGIZE_ON_REJECT", "false").lower() == "true")
    # Inject the current local time into the session on every turn, so the
    # model can't guess the time of day. IANA zone name.
    TIME_ZONE: str = field(default_factory=lambda: os.getenv(
        "CLAWBODY_TIMEZONE", "America/Los_Angeles"))
    TIME_GROUNDING_ENABLED: bool = field(default_factory=lambda: os.getenv(
        "CLAWBODY_TIME_GROUNDING", "true").lower() == "true")
    # Voice-callable sleep/wake. Uses the SDK's goto_sleep()/wake_up().
    ENABLE_SLEEP_TOOLS: bool = field(default_factory=lambda: os.getenv(
        "CLAWBODY_ENABLE_SLEEP_TOOLS", "true").lower() == "true")
    # Suppress duplicate turn syncs to the OpenClaw gateway.
    SYNC_DEDUPE_ENABLED: bool = field(default_factory=lambda: os.getenv(
        "CLAWBODY_SYNC_DEDUPE", "true").lower() == "true")
    SYNC_DEDUPE_TTL_S: float = field(default_factory=lambda: float(
        os.getenv("CLAWBODY_SYNC_DEDUPE_TTL_S", "900") or 900.0))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        return errors


# Global configuration instance
config = Config()


def set_custom_profile(profile: Optional[str]) -> None:
    """Update the custom profile at runtime."""
    global config
    config.CUSTOM_PROFILE = profile
    os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile or ""


def set_face_tracking_enabled(enabled: bool) -> None:
    """Enable or disable face tracking at runtime."""
    global config
    config.ENABLE_FACE_TRACKING = enabled


def set_local_vision_enabled(enabled: bool) -> None:
    """Enable or disable local vision processing at runtime."""
    global config
    config.ENABLE_LOCAL_VISION = enabled
