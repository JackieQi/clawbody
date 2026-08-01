"""Voice input gating: decide whether an utterance is meant for the robot.

The chassis mics hear everything in the room -- the user, children playing,
a television, the robot's own servos -- and the Realtime API's transcriber
will return *something* for almost any of it. Left ungated that produces
three distinct failures, all observed on this robot:

1. Language misdetection. English/Mandarin speech comes back as Korean
   ("이거?"), Croatian ("Pogledaj."), Turkish ("bir") or Cyrillic ("В").
2. Confabulation. Handed one of those fragments, the model does not say
   "I didn't understand"; it invents a context that would make the
   fragment sensible. "Pogledaj." at 13:28 produced "Goodnight! Hope you
   sleep well."
3. Greeting spam. Every stray "Mhm" / "啊" becomes a full turn, so the
   robot re-introduces itself several times a minute.

This module is the filter in front of the model. It is deliberately pure:
no I/O, no API objects, no robot handles -- just text in, decision out --
so the rules can be unit tested without a robot or a network.

The rules, in order:

    asleep      -> only a wake word gets through, nothing else
    empty       -> silence
    foreign     -> silence (script outside the allowlist)
    ack         -> silence ("mhm", "嗯", "ok")
    null        -> silence or a brief apology (too short to mean anything)
    not addressed -> silence (no wake word, no open grace window)
    otherwise   -> forward to the model

Every rule is individually disableable through `GateSettings` so a bad
default can be turned off from the environment rather than by editing code.
"""

from __future__ import annotations

import re
import time
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Script detection
# --------------------------------------------------------------------------
#
# Detecting *language* from a five-character transcript is not reliable, and
# every library that claims to do it needs far more text than a voice turn
# provides. Detecting *script* is exact: a string either contains Hangul
# codepoints or it does not. That is enough to catch the failure actually
# observed here, where the transcriber returns Korean/Cyrillic/kana for
# English or Mandarin audio -- the wrong-script output is the tell.

_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "hangul": (
        (0x1100, 0x11FF),  # Jamo
        (0x3130, 0x318F),  # Compatibility Jamo
        (0xA960, 0xA97F),  # Jamo Extended-A
        (0xAC00, 0xD7AF),  # Syllables
    ),
    "cyrillic": (
        (0x0400, 0x04FF),
        (0x0500, 0x052F),
    ),
    "kana": (
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x31F0, 0x31FF),  # Katakana phonetic extensions
    ),
    "han": (
        (0x3400, 0x4DBF),  # Ext A
        (0x4E00, 0x9FFF),  # Unified ideographs
        (0xF900, 0xFAFF),  # Compatibility ideographs
    ),
    "thai": ((0x0E00, 0x0E7F),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "hebrew": ((0x0590, 0x05FF),),
    "devanagari": ((0x0900, 0x097F),),
    "greek": ((0x0370, 0x03FF),),
    "latin": (
        (0x0041, 0x005A),
        (0x0061, 0x007A),
        (0x00C0, 0x024F),  # Latin-1 Supplement + Extended-A/B
    ),
}

# Latin letters that never appear in English or in pinyin as normally
# transcribed. Their presence in an otherwise-Latin string means the
# transcriber has produced some other European language.
_NON_ENGLISH_LATIN = set("àáâãäåæçèéêëìíîïñòóôõöøùúûüýÿāăąćĉċčďđēĕėęěĝğġģĥħĩīĭįıĵķĺļľłńņňŋōŏőœŕŗřśŝşšţťŧũūŭůűųŵŷźżžșțğıİ")

# Script names that mean "this is Mandarin or English", i.e. what Jackie
# actually speaks. Anything else is somebody else, or a mistranscription.
LANG_TO_SCRIPTS: dict[str, tuple[str, ...]] = {
    "en": ("latin",),
    "zh": ("han", "latin"),  # pinyin and mixed English/Mandarin are normal
}


def scripts_in(text: str) -> set[str]:
    """Return the set of Unicode script names present in `text`.

    Only letters are classified; digits, punctuation and whitespace are
    ignored, so "OK?" and "OK" produce the same answer.

    Args:
        text: Arbitrary transcript text.

    Returns:
        Set of script names drawn from `_SCRIPT_RANGES` keys. Characters in
        no known range are ignored rather than reported as "unknown", so a
        stray emoji cannot make an English sentence look foreign.
    """
    found: set[str] = set()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for name, ranges in _SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                found.add(name)
                break
    return found


def detect_language_family(text: str) -> str:
    """Classify a transcript into a coarse language family.

    This is a script classifier with two refinements that matter for the
    room this robot lives in:

    - Kana anywhere means Japanese, even mixed with Han characters. The
      children in the room speak Japanese and their speech must never be
      taken as input, so kana is decisive rather than advisory.
    - Latin text carrying non-English diacritics ("Pogledaj" is clean ASCII
      and cannot be caught this way, but "čeka" can) is reported as
      "other".

    Args:
        text: Transcript text.

    Returns:
        One of "en", "zh", "ja", "ko", "other", or "unknown" when the text
        contains no letters at all.
    """
    found = scripts_in(text)
    if not found:
        return "unknown"
    if "kana" in found:
        return "ja"
    if "hangul" in found:
        return "ko"
    if "han" in found:
        return "zh"
    if found <= {"latin"}:
        if any(ch in _NON_ENGLISH_LATIN for ch in text.lower()):
            return "other"
        return "en"
    return "other"


# --------------------------------------------------------------------------
# Acknowledgment / semantic-null detection
# --------------------------------------------------------------------------

# Tokens that carry no request. Answering these with a greeting is the
# specific behaviour Jackie asked to stop: "If I don't call you, you don't
# have to greet again and again."
DEFAULT_ACK_TOKENS: tuple[str, ...] = (
    # English
    "mm", "mmm", "mhm", "mhmm", "hm", "hmm", "uh", "uhh", "um", "umm",
    "ah", "ahh", "eh", "ehh", "oh", "ohh", "huh", "yeah", "yep", "yup",
    "ok", "okay", "k", "right", "sure", "uh huh", "mm hmm",
    # Mandarin
    "嗯", "嗯嗯", "啊", "呃", "哦", "喔", "噢", "唔", "欸", "誒", "诶",
    "好", "好的", "好吧", "是", "是啊", "是的", "对", "对啊", "對", "對啊",
    "嗯哼", "哈", "耶",
)

# Punctuation stripped before matching. Includes the CJK forms, since the
# transcriber emits "嗯。" and "啊？" rather than the ASCII equivalents.
_PUNCT_RE = re.compile(r"[\s.,!?;:'\"()\[\]{}…\-–—。，！？；：、「」『』（）〈〉《》“”‘’]+")


def normalize_transcript(text: str) -> str:
    """Lowercase and strip punctuation/whitespace for matching.

    Args:
        text: Raw transcript.

    Returns:
        Lowercased text with punctuation collapsed to single spaces and
        the ends trimmed. Interior spacing is preserved as single spaces so
        multi-word acks ("uh huh") still match.
    """
    return _PUNCT_RE.sub(" ", text.lower()).strip()


def is_acknowledgment(text: str, tokens: Iterable[str] = DEFAULT_ACK_TOKENS) -> bool:
    """Is this transcript nothing but an acknowledgment noise?

    Matches only when the *entire* utterance is acknowledgment tokens, so
    "ok, what's the weather" is not suppressed while "ok" and "嗯嗯" are.

    Args:
        text: Raw transcript.
        tokens: Acknowledgment vocabulary; defaults to `DEFAULT_ACK_TOKENS`.

    Returns:
        True when every token in the utterance is an acknowledgment.
    """
    norm = normalize_transcript(text)
    if not norm:
        return False
    token_set = {t.lower() for t in tokens}
    if norm in token_set:
        return True

    # CJK acks arrive unspaced ("嗯嗯啊"); split them per character as well
    # as on whitespace, since there is no space to split on.
    parts = norm.split()
    if all(p in token_set for p in parts):
        return True
    if not any(ch.isascii() and ch.isalnum() for ch in norm):
        chars = [ch for ch in norm if not ch.isspace()]
        if chars and all(ch in token_set for ch in chars):
            return True
    return False


def _is_logographic(ch: str) -> bool:
    """Is this character a CJK ideograph or kana (one char = one word)?"""
    cp = ord(ch)
    for name in ("han", "kana"):
        if any(lo <= cp <= hi for lo, hi in _SCRIPT_RANGES[name]):
            return True
    return False


def is_semantically_null(text: str, min_chars: int = 2) -> bool:
    """Is the transcript too small to carry a request?

    A one- or two-character fragment ("В", "bir", "Ra") is never a usable
    instruction, and handing it to the model is what triggers confabulation.
    Counted in letters, so "?" and "..." are null regardless of length.

    Args:
        text: Raw transcript.
        min_chars: Minimum letter count to be considered meaningful.
            Logographic scripts are exempt: a single Han ideograph or kana
            character can be a complete word, where a single Latin or
            Cyrillic letter cannot.

    Returns:
        True when the transcript should be treated as noise.
    """
    norm = normalize_transcript(text)
    letters = [ch for ch in norm if ch.isalpha()]
    if not letters:
        return True
    # Exempt only logographic scripts. An earlier version exempted every
    # non-ASCII letter, which let the observed single Cyrillic "В" through
    # as if it were a Chinese word.
    if any(_is_logographic(ch) for ch in letters):
        return False
    return len(letters) < max(1, min_chars)


# --------------------------------------------------------------------------
# Wake-word matching
# --------------------------------------------------------------------------

# "Kira" as the transcriber has actually rendered it on this robot. Short
# variants are included deliberately: the cost of a false accept is one
# unnecessary reply, the cost of a false reject is the robot ignoring its
# own name.
DEFAULT_WAKE_WORDS: tuple[str, ...] = (
    "kira", "akira", "kiera", "kyra", "keira", "kira's", "ra",
    "奇拉", "基拉", "keira",
)

_ASCII_WORD_BOUNDARY = r"(?<![a-zA-Z0-9]){}(?![a-zA-Z0-9])"


def find_wake_word(text: str, wake_words: Iterable[str]) -> Optional[str]:
    """Return the wake word found in `text`, or None.

    Latin wake words match on ASCII-alnum boundaries rather than ``\\b``:
    ``\\b`` treats CJK ideographs as word characters, so "奇拉kira" would
    fail to match. Non-Latin wake words match as plain substrings, since
    Mandarin is not space-delimited.

    Args:
        text: Raw transcript.
        wake_words: Candidate wake words, matched case-insensitively.

    Returns:
        The first wake word that matched, in the order given, or None.
    """
    if not text:
        return None
    for word in wake_words:
        w = word.strip()
        if not w:
            continue
        if w.isascii():
            pattern = _ASCII_WORD_BOUNDARY.format(re.escape(w))
        else:
            pattern = re.escape(w)
        if re.search(pattern, text, re.IGNORECASE):
            return w
    return None


# --------------------------------------------------------------------------
# Turn deduplication
# --------------------------------------------------------------------------


def turn_fingerprint(*parts: Optional[str]) -> str:
    """Stable content hash for a conversation turn.

    Used to stop the same captured turn being replayed to OpenClaw. Keyed
    on content rather than on time, because the observed duplicates were
    byte-identical resends thirteen minutes apart -- a timestamp in the key
    would have made every one of them look new.

    Args:
        *parts: Turn components (user text, assistant text, item id...).
            None parts are treated as empty strings.

    Returns:
        Hex SHA-256 digest, truncated to 32 chars (collision risk is
        irrelevant at the scale of one conversation).
    """
    joined = "\x1f".join((p or "") for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


class TurnDeduper:
    """Bounded set of recently seen turn fingerprints.

    An ``OrderedDict`` used as an LRU: unbounded memory in a process that
    runs for days is not acceptable, and the duplicates being defended
    against are always near each other in the stream.
    """

    def __init__(self, capacity: int = 128, ttl_s: float = 900.0) -> None:
        """
        Args:
            capacity: Maximum fingerprints retained.
            ttl_s: Age after which a fingerprint is forgotten, so a genuine
                repetition ("thanks" an hour later) is not swallowed. 0
                disables expiry.
        """
        self.capacity = max(1, int(capacity))
        self.ttl_s = float(ttl_s)
        self._seen: "OrderedDict[str, float]" = OrderedDict()

    def seen(self, fingerprint: str, now: Optional[float] = None) -> bool:
        """Record a fingerprint and report whether it was already present.

        Args:
            fingerprint: Key from `turn_fingerprint`.
            now: Injectable clock for tests; defaults to `time.monotonic`.

        Returns:
            True if this fingerprint was recorded recently (a duplicate).
        """
        t = time.monotonic() if now is None else now
        prev = self._seen.get(fingerprint)
        duplicate = prev is not None and (
            self.ttl_s <= 0.0 or (t - prev) < self.ttl_s
        )
        self._seen[fingerprint] = t
        self._seen.move_to_end(fingerprint)
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        return duplicate

    def reset(self) -> None:
        """Forget every fingerprint (used on session boot)."""
        self._seen.clear()


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@dataclass
class GateSettings:
    """Tunables for `VoiceGate`. Every rule can be switched off here.

    Attributes:
        enabled: Master switch. False makes the gate a pass-through.
        require_wake_word: Require the robot be addressed by name.
        wake_words: Accepted names and mistranscriptions.
        grace_s: How long after the exchange goes quiet -- the robot has
            finished its reply -- follow-ups are accepted without repeating
            the name. 0 means every turn needs the name.
        allowed_languages: Language families forwarded to the model.
        min_chars: Minimum letters for a Latin transcript to be meaningful.
        suppress_acks: Drop pure acknowledgment noises.
        ack_tokens: Acknowledgment vocabulary.
        apologize_on_reject: Speak a short "didn't catch that" instead of
            staying silent, for unintelligible input only. Never used for
            foreign-language or unaddressed speech -- replying to those
            would defeat the point of ignoring them.
    """

    enabled: bool = True
    require_wake_word: bool = True
    wake_words: tuple[str, ...] = DEFAULT_WAKE_WORDS
    grace_s: float = 150.0
    allowed_languages: tuple[str, ...] = ("en", "zh")
    min_chars: int = 2
    suppress_acks: bool = True
    ack_tokens: tuple[str, ...] = DEFAULT_ACK_TOKENS
    apologize_on_reject: bool = False


@dataclass(frozen=True)
class GateDecision:
    """Outcome of evaluating one transcript.

    Attributes:
        allow: Forward this turn to the model.
        reason: Machine-readable rule that decided it, for logs.
        apology: Speak a brief non-comprehension line. Only ever set when
            `allow` is False.
        wake_word: The wake word matched, if any.
        woke_from_standby: True when this turn ended a standby period, so
            the caller knows to run the physical wake-up.
        language: Detected language family.
    """

    allow: bool
    reason: str
    apology: bool = False
    wake_word: Optional[str] = None
    woke_from_standby: bool = False
    language: str = "unknown"


class VoiceGate:
    """Decides whether a transcribed utterance should reach the model.

    Holds the small amount of state the rules need: when the conversational
    grace window closes, and whether the robot is in standby. Not
    thread-safe; it is driven from the audio event loop only.
    """

    def __init__(self, settings: Optional[GateSettings] = None) -> None:
        """
        Args:
            settings: Tunables; defaults to `GateSettings()`.
        """
        self.settings = settings or GateSettings()
        self._grace_until = 0.0
        self._standby = False

    # -- standby ---------------------------------------------------------

    @property
    def standby(self) -> bool:
        """True while the robot is asleep and ignoring everything but its name."""
        return self._standby

    def enter_standby(self) -> None:
        """Go to sleep: close the grace window and require the wake word."""
        self._standby = True
        self._grace_until = 0.0
        logger.info("Voice gate: entering standby (wake word required)")

    def leave_standby(self, now: Optional[float] = None) -> None:
        """Wake up and open the grace window as if just addressed."""
        self._standby = False
        self._open_grace(now)
        logger.info("Voice gate: leaving standby")

    # -- grace window ----------------------------------------------------

    def _open_grace(self, now: Optional[float] = None) -> None:
        self.refresh_grace(now)

    def refresh_grace(self, now: Optional[float] = None, delay: float = 0.0) -> None:
        """Restart the follow-up window: the conversation is still live.

        `delay` postpones the start, for activity that is still in
        progress. The robot's own reply is part of the exchange, so the
        window has to begin when that reply *finishes* -- opening it when
        the reply was merely generated spends most of it on the robot's
        own talking, and the user is cut off partway through thinking.
        """
        t = time.monotonic() if now is None else now
        self._grace_until = t + max(0.0, delay) + max(0.0, self.settings.grace_s)

    def grace_open(self, now: Optional[float] = None) -> bool:
        """Is the conversational follow-up window still open?"""
        t = time.monotonic() if now is None else now
        return t < self._grace_until

    def close_grace(self) -> None:
        """Re-arm the gate immediately (used when a turn is rejected as noise)."""
        self._grace_until = 0.0

    # -- the rules -------------------------------------------------------

    def evaluate(self, text: Optional[str], now: Optional[float] = None) -> GateDecision:
        """Apply every gate rule to one transcript.

        Args:
            text: The transcript, or None/"" when transcription failed.
            now: Injectable monotonic clock for tests.

        Returns:
            A `GateDecision`. Callers must not generate a response when
            `allow` is False; they may speak a fixed apology line when
            `apology` is True.
        """
        s = self.settings
        t = time.monotonic() if now is None else now

        if not s.enabled:
            return GateDecision(True, "gate_disabled", language=detect_language_family(text or ""))

        raw = (text or "").strip()
        if not raw:
            # Transcription failed or returned nothing. There is no content
            # to reason about, so there is nothing safe to say.
            return GateDecision(False, "empty")

        language = detect_language_family(raw)
        wake = find_wake_word(raw, s.wake_words)

        # Standby overrides everything else: only the name gets through.
        if self._standby:
            if wake and language in s.allowed_languages:
                self._standby = False
                self._open_grace(t)
                return GateDecision(
                    True, "wake_from_standby", wake_word=wake,
                    woke_from_standby=True, language=language,
                )
            return GateDecision(False, "standby", language=language)

        # Language allowlist. Checked before the wake word so that a Korean
        # or Japanese utterance is dropped even if it happens to contain a
        # sound the matcher would read as "Ra".
        if language not in s.allowed_languages and language != "unknown":
            return GateDecision(False, f"language_{language}", language=language)

        if s.suppress_acks and is_acknowledgment(raw, s.ack_tokens):
            # Deliberately does not touch the grace window: an "mhm" during
            # a conversation should neither extend it nor end it.
            return GateDecision(False, "acknowledgment", language=language)

        if is_semantically_null(raw, s.min_chars) and not wake:
            # The confabulation trigger. Nothing here to answer.
            return GateDecision(
                False, "too_short", apology=s.apologize_on_reject, language=language
            )

        if s.require_wake_word and not wake and not self.grace_open(t):
            return GateDecision(False, "not_addressed", language=language)

        # Accepted. Any accepted turn refreshes the follow-up window, so a
        # conversation does not require the name on every sentence.
        self._open_grace(t)
        return GateDecision(
            True,
            "wake_word" if wake else ("grace_window" if s.require_wake_word else "open"),
            wake_word=wake,
            language=language,
        )


def settings_from_config(cfg: object) -> GateSettings:
    """Build `GateSettings` from the application config object.

    Kept here rather than in config.py so the parsing of the list-valued
    settings lives next to the code that consumes them.

    Args:
        cfg: The `Config` instance (duck-typed; missing attributes fall
            back to the dataclass defaults).

    Returns:
        Populated `GateSettings`.
    """
    def _get(name: str, default):
        value = getattr(cfg, name, None)
        return default if value is None else value

    def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        raw = getattr(cfg, name, None)
        if not raw:
            return default
        parts = tuple(p.strip() for p in str(raw).split(",") if p.strip())
        return parts or default

    return GateSettings(
        enabled=bool(_get("VOICE_GATE_ENABLED", True)),
        require_wake_word=bool(_get("WAKE_WORD_REQUIRED", True)),
        wake_words=_csv("WAKE_WORDS", DEFAULT_WAKE_WORDS),
        grace_s=float(_get("WAKE_GRACE_S", 150.0)),
        allowed_languages=_csv("VOICE_LANGUAGES", ("en", "zh")),
        min_chars=int(_get("VOICE_MIN_CHARS", 2)),
        suppress_acks=bool(_get("SUPPRESS_ACKS", True)),
        apologize_on_reject=bool(_get("APOLOGIZE_ON_REJECT", False)),
    )
