"""Tests for the voice input gate.

Every case here is drawn from something the robot actually did on
2026-08-01 between 13:00 and 13:52, or from a constraint Jackie stated
directly. The transcripts are verbatim.
"""

from __future__ import annotations

import pytest

from reachy_mini_openclaw.voice_gate import (
    DEFAULT_WAKE_WORDS,
    GateSettings,
    TurnDeduper,
    VoiceGate,
    detect_language_family,
    find_wake_word,
    is_acknowledgment,
    is_semantically_null,
    normalize_transcript,
    scripts_in,
    settings_from_config,
    turn_fingerprint,
)


# ---------------------------------------------------------------------------
# Script / language detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello there", "en"),
        ("What's the weather today?", "en"),
        ("你好，今天天氣怎麼樣", "zh"),
        ("Kira 你好", "zh"),          # mixed EN/ZH is still Mandarin-family
        ("这是 pinyin ma", "zh"),
        ("이거?", "ko"),               # observed misdetection
        ("뭐?", "ko"),
        ("하이", "ko"),
        ("В", "other"),                # observed Cyrillic misdetection
        ("Jest.", "en"),               # clean ASCII; caught by length, not script
        ("これは何ですか", "ja"),       # children in the room
        ("ありがとう", "ja"),
        ("日本語です", "ja"),           # kana beats han
        ("こんにちは", "ja"),
        ("čeka", "other"),
        ("", "unknown"),
        ("123 ...", "unknown"),
    ],
)
def test_detect_language_family(text: str, expected: str) -> None:
    assert detect_language_family(text) == expected


def test_kana_plus_han_is_japanese_not_chinese() -> None:
    """Kids' Japanese often mixes kanji and kana; it must not read as zh."""
    assert detect_language_family("私は元気です") == "ja"


def test_scripts_in_ignores_punctuation_and_digits() -> None:
    assert scripts_in("OK? 123!") == {"latin"}
    assert scripts_in("!!!") == set()


# ---------------------------------------------------------------------------
# Acknowledgments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["Mhm", "mhm.", "Ah", "Eh", "啊", "嗯", "嗯嗯", "ok", "OK.", "好的",
     "是啊", "yeah", "uh huh", "對啊", "Hmm..."],
)
def test_acknowledgments_are_detected(text: str) -> None:
    assert is_acknowledgment(text) is True


@pytest.mark.parametrize(
    "text",
    ["ok what's the weather", "yeah but why", "嗯，今天天氣如何",
     "hello there", "好的我知道了"],
)
def test_real_content_is_not_an_acknowledgment(text: str) -> None:
    assert is_acknowledgment(text) is False


def test_normalize_strips_cjk_punctuation() -> None:
    assert normalize_transcript("嗯。") == "嗯"
    assert normalize_transcript("  Ok!  ") == "ok"


# ---------------------------------------------------------------------------
# Semantic nulls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["В", "b", "?", "...", "", "  ", "!!"])
def test_short_fragments_are_null(text: str) -> None:
    assert is_semantically_null(text) is True


def test_single_cyrillic_letter_is_null_not_a_word() -> None:
    """Regression: only CJK/kana get the one-character-is-a-word exemption.

    An earlier version exempted every non-ASCII letter, so the observed
    single Cyrillic "В" was treated as meaningful content.
    """
    assert is_semantically_null("В") is True
    assert is_semantically_null("好") is False


@pytest.mark.parametrize("text", ["好", "嗎", "hey", "what", "bir"])
def test_words_are_not_null(text: str) -> None:
    """A lone CJK ideograph is a word; two Latin letters is the floor."""
    assert is_semantically_null(text) is False


# ---------------------------------------------------------------------------
# Wake words
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Kira", "kira"),
        ("kira, what time is it", "kira"),
        ("Hey Akira!", "akira"),
        ("KYRA?", "kyra"),
        ("Kiera are you there", "kiera"),
        ("奇拉你好", "奇拉"),
        ("奇拉kira", "kira"),       # Latin cue adjacent to CJK still matches
    ],
)
def test_wake_word_found(text: str, expected: str) -> None:
    assert find_wake_word(text, DEFAULT_WAKE_WORDS) == expected


@pytest.mark.parametrize(
    "text",
    ["what's the weather", "kirana", "akirah", "the camera", "hello"],
)
def test_wake_word_not_found(text: str) -> None:
    """Word boundaries: 'kira' inside 'kirana' must not fire."""
    assert find_wake_word(text, DEFAULT_WAKE_WORDS) is None


def test_wake_word_ignores_empty_entries() -> None:
    assert find_wake_word("kira", ["", "  ", "kira"]) == "kira"


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def _gate(**overrides) -> VoiceGate:
    return VoiceGate(GateSettings(**overrides))


def test_wake_word_opens_the_gate() -> None:
    g = _gate()
    d = g.evaluate("Kira, what's the weather?", now=100.0)
    assert d.allow
    assert d.wake_word == "kira"
    assert d.reason == "wake_word"


def test_unaddressed_speech_is_dropped() -> None:
    g = _gate()
    d = g.evaluate("what's the weather?", now=100.0)
    assert not d.allow
    assert d.reason == "not_addressed"


def test_grace_window_admits_followups() -> None:
    g = _gate(grace_s=30.0)
    assert g.evaluate("Kira, what's the weather?", now=100.0).allow
    d = g.evaluate("and tomorrow?", now=110.0)
    assert d.allow
    assert d.reason == "grace_window"


def test_grace_window_expires_and_rearms() -> None:
    g = _gate(grace_s=30.0)
    assert g.evaluate("Kira, hello", now=100.0).allow
    assert not g.evaluate("and tomorrow?", now=140.0).allow


def test_grace_window_is_refreshed_by_each_accepted_turn() -> None:
    """A running conversation shouldn't time out mid-way."""
    g = _gate(grace_s=30.0)
    assert g.evaluate("Kira, hello", now=100.0).allow
    assert g.evaluate("how are you", now=125.0).allow
    assert g.evaluate("and the news?", now=150.0).allow


def test_zero_grace_requires_the_name_every_turn() -> None:
    g = _gate(grace_s=0.0)
    assert g.evaluate("Kira, hello", now=100.0).allow
    assert not g.evaluate("how are you", now=100.5).allow


def test_japanese_is_dropped_even_inside_the_grace_window() -> None:
    """Kids in the room must not be able to hijack an open conversation."""
    g = _gate(grace_s=30.0)
    assert g.evaluate("Kira, hello", now=100.0).allow
    d = g.evaluate("これは何ですか", now=105.0)
    assert not d.allow
    assert d.reason == "language_ja"


@pytest.mark.parametrize("text", ["이거?", "뭐?", "하이", "В"])
def test_misdetected_languages_are_dropped(text: str) -> None:
    g = _gate(grace_s=30.0)
    g.evaluate("Kira, hello", now=100.0)
    assert not g.evaluate(text, now=105.0).allow


def test_pogledaj_is_dropped_when_unaddressed() -> None:
    """The confabulation bug: this produced 'Goodnight!' at 13:28.

    Outside the grace window the wake-word rule catches it, which is the
    situation it actually occurred in.
    """
    g = _gate(grace_s=30.0)
    d = g.evaluate("Pogledaj.", now=100.0)
    assert not d.allow
    assert d.reason == "not_addressed"


def test_known_limitation_ascii_only_foreign_words_pass_in_grace_window() -> None:
    """Documents what this gate cannot do.

    "Pogledaj" is Croatian but contains no non-English character: it is
    nine plain ASCII letters, so script detection classifies it as
    English, and it is long enough not to be a semantic null. Inside an
    open grace window it therefore reaches the model.

    Catching this would need real language identification, which is not
    reliable on a single word and would add a model dependency to a hot
    path. The second line of defence is instead prompt-side: the address
    discipline block forbids guessing at unintelligible input, and time
    grounding removes the specific "Goodnight!" failure by making the
    model's time-of-day claim checkable.

    This test exists so the limitation is visible rather than forgotten;
    if language ID is added later it should start failing.
    """
    g = _gate(grace_s=30.0)
    g.evaluate("Kira, hello", now=100.0)
    assert g.evaluate("Pogledaj.", now=105.0).allow


def test_acknowledgments_are_dropped_inside_the_grace_window() -> None:
    g = _gate(grace_s=30.0)
    g.evaluate("Kira, what's the weather?", now=100.0)
    d = g.evaluate("Mhm", now=105.0)
    assert not d.allow
    assert d.reason == "acknowledgment"


def test_acknowledgment_does_not_close_the_grace_window() -> None:
    """Nodding along shouldn't end the conversation."""
    g = _gate(grace_s=30.0)
    g.evaluate("Kira, hello", now=100.0)
    g.evaluate("Mhm", now=105.0)
    assert g.evaluate("so what about tomorrow", now=110.0).allow


def test_empty_transcript_is_dropped() -> None:
    g = _gate()
    assert not g.evaluate("", now=100.0).allow
    assert not g.evaluate(None, now=100.0).allow


def test_short_fragment_can_request_an_apology() -> None:
    g = _gate(grace_s=30.0, apologize_on_reject=True)
    g.evaluate("Kira, hello", now=100.0)
    d = g.evaluate("b", now=105.0)
    assert not d.allow
    assert d.apology is True


def test_apology_is_off_for_foreign_language() -> None:
    """Answering the kids to say 'I didn't understand' defeats the point."""
    g = _gate(grace_s=30.0, apologize_on_reject=True)
    g.evaluate("Kira, hello", now=100.0)
    d = g.evaluate("これは何ですか", now=105.0)
    assert not d.allow
    assert d.apology is False


def test_disabled_gate_is_a_passthrough() -> None:
    g = _gate(enabled=False)
    assert g.evaluate("Pogledaj.", now=100.0).allow
    assert g.evaluate("これは何ですか", now=100.0).allow


def test_wake_word_not_required_still_filters_noise() -> None:
    """Turning off addressing must not turn off the language allowlist."""
    g = _gate(require_wake_word=False)
    assert g.evaluate("what's the weather", now=100.0).allow
    assert not g.evaluate("これは何ですか", now=100.0).allow
    assert not g.evaluate("Mhm", now=100.0).allow


# ---------------------------------------------------------------------------
# Standby
# ---------------------------------------------------------------------------


def test_standby_ignores_everything_but_the_wake_word() -> None:
    g = _gate()
    g.enter_standby()
    assert g.standby
    assert not g.evaluate("what's the weather", now=100.0).allow
    assert not g.evaluate("hello", now=100.0).allow


def test_wake_word_leaves_standby() -> None:
    g = _gate()
    g.enter_standby()
    d = g.evaluate("Kira, wake up", now=100.0)
    assert d.allow
    assert d.woke_from_standby is True
    assert not g.standby


def test_standby_ignores_the_wake_word_in_a_foreign_language() -> None:
    g = _gate()
    g.enter_standby()
    assert not g.evaluate("キラ", now=100.0).allow
    assert g.standby


def test_leave_standby_opens_the_grace_window() -> None:
    g = _gate(grace_s=30.0)
    g.enter_standby()
    g.leave_standby(now=100.0)
    assert g.evaluate("what's the weather", now=105.0).allow


def test_standby_survives_a_grace_window() -> None:
    """Going to sleep mid-conversation must close the window immediately."""
    g = _gate(grace_s=30.0)
    g.evaluate("Kira, hello", now=100.0)
    g.enter_standby()
    assert not g.evaluate("and tomorrow?", now=105.0).allow


# ---------------------------------------------------------------------------
# Turn dedupe
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    a = turn_fingerprint("hello", "hi there")
    assert a == turn_fingerprint("hello", "hi there")
    assert a != turn_fingerprint("hello", "hi there!")
    assert a != turn_fingerprint("hello", None)


def test_fingerprint_is_not_confused_by_field_boundaries() -> None:
    """('ab', 'c') and ('a', 'bc') must not collide."""
    assert turn_fingerprint("ab", "c") != turn_fingerprint("a", "bc")


def test_deduper_detects_a_repeat() -> None:
    d = TurnDeduper()
    fp = turn_fingerprint("hello", "hi")
    assert d.seen(fp, now=0.0) is False
    assert d.seen(fp, now=1.0) is True


def test_deduper_catches_the_thirteen_minute_replay() -> None:
    """The observed bug: identical turn re-sent at 13:07 and 13:20."""
    d = TurnDeduper(ttl_s=900.0)
    fp = turn_fingerprint("what time is it", "It's just after one.")
    assert d.seen(fp, now=0.0) is False
    assert d.seen(fp, now=13 * 60.0) is True


def test_deduper_forgets_after_ttl() -> None:
    d = TurnDeduper(ttl_s=900.0)
    fp = turn_fingerprint("thanks", "you're welcome")
    assert d.seen(fp, now=0.0) is False
    assert d.seen(fp, now=1000.0) is False


def test_deduper_is_bounded() -> None:
    d = TurnDeduper(capacity=4)
    for i in range(10):
        d.seen(turn_fingerprint(str(i)), now=float(i))
    assert len(d._seen) == 4
    # Oldest evicted, newest retained
    assert d.seen(turn_fingerprint("0"), now=10.0) is False
    assert d.seen(turn_fingerprint("9"), now=10.0) is True


def test_deduper_reset() -> None:
    d = TurnDeduper()
    fp = turn_fingerprint("a", "b")
    d.seen(fp, now=0.0)
    d.reset()
    assert d.seen(fp, now=1.0) is False


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class _FakeConfig:
    VOICE_GATE_ENABLED = True
    WAKE_WORD_REQUIRED = False
    WAKE_WORDS = "kira, robot ,"
    WAKE_GRACE_S = 12.5
    VOICE_LANGUAGES = "en"
    VOICE_MIN_CHARS = 3
    SUPPRESS_ACKS = False
    APOLOGIZE_ON_REJECT = True


def test_settings_from_config_parses_csv_lists() -> None:
    s = settings_from_config(_FakeConfig())
    assert s.wake_words == ("kira", "robot")
    assert s.allowed_languages == ("en",)
    assert s.grace_s == 12.5
    assert s.min_chars == 3
    assert s.require_wake_word is False
    assert s.suppress_acks is False
    assert s.apologize_on_reject is True


def test_settings_from_config_falls_back_on_blanks() -> None:
    class Blank:
        WAKE_WORDS = ""
        VOICE_LANGUAGES = "   "

    s = settings_from_config(Blank())
    assert s.wake_words == DEFAULT_WAKE_WORDS
    assert s.allowed_languages == ("en", "zh")


def test_settings_from_config_tolerates_missing_attributes() -> None:
    s = settings_from_config(object())
    assert s.enabled is True
    assert s.wake_words == DEFAULT_WAKE_WORDS
