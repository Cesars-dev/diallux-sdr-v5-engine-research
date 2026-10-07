"""Speech normalization for TTS — the Retell-native feature, self-hosted.

Retell normalizes numbers/addresses before speaking (its platform layer does
it for you). Raw LangGraph has no such layer, so an LLM that writes "my
number is 3124001234" would be read at full speed by Cartesia/ElevenLabs —
the "665657 ramble" problem. This module is the deterministic fix, applied
to every sentence right before it hits the TTS socket (never to history,
never to dvs, never to the prompt).

Provider research that shaped the rules (docs.cartesia.ai, 2026-08-14):
  - Sonic reads CONVENTIONAL written forms naturally: "(415) 555-1212",
    "$19.99", "04/20/2025", "user@example.com". Pre-normalization is only
    recommended for edge cases.
  - For digit-by-digit read-out Sonic 3.5 wants space/comma delimiters:
    "A B C, 1 2 3" (NOT the Sonic-3 "A. B. C." style), or <spell> tags.
  - ElevenLabs has no <spell>; comma delimiters work on both providers.
  - Both providers read markdown/emoji literally — strip them.

Our policy (tts_phone_style):
  "digits"  (default) phone-like and long digit runs -> "3, 1, 2, 4, 0, 0, 1, 2, 3, 4"
            slow, unambiguous, provider-agnostic — what an SDR confirming a
            number wants.
  "natural" leave phone-shaped strings as written ("(312) 400-1234" is read
            correctly by both engines); still split bare 7+ digit runs.
  "spell"   Cartesia-only: wrap runs in <spell> tags (works on one sentence
            chunk at a time; our gate never splits a sentence).

Everything here is pure regex — zero latency (µs), zero API calls.
"""
from __future__ import annotations

import re

# markdown symbols that TTS engines would read aloud or stutter on
_MD_CHARS = re.compile(r"[*_`~>#|]+")
# emoji & symbol ranges (keep Latin-1 accents for names like José)
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U0000FE0F\U0000200D\U00002190-\U000021FF\U00002B00-\U00002BFF]+")
# phone-shaped candidate: digits + separators, filtered by digit count >= 10
_PHONE_CAND_RE = re.compile(r"\+?\(?[\d][\d\s().-]{8,}\d\b(?![\w])")
# bare digit runs (665657, booking numbers, readouts)
_DIGIT_RUN_RE = re.compile(r"\d{5,}")
# mixed alnum tokens (qeTqHuZ1EDzH8bxEdhPQ6H) read best char-by-char
_MIXED_TOKEN_RE = re.compile(r"\b(?=\w*\d)(?=\w*[A-Za-z])\w{8,}\b")
_TERMINAL = ".!?…"


def _digits_out(digit_str: str) -> str:
    return ", ".join(digit_str)


def _spell(digits: str) -> str:
    return f"<spell>{digits}</spell>"


def normalize_for_tts(
    text: str,
    phone_style: str = "digits",
    strip_markdown: bool = True,
) -> str:
    """Normalize one TTS-bound sentence chunk. Deterministic, order-stable.

    Steps:
      1. strip markdown symbols + emoji, collapse whitespace
      2. phone-shaped strings + bare 5+ digit runs -> comma-separated digits
         (or <spell> when phone_style="spell", or untouched when "natural")
      3. long mixed alnum tokens (booking UIDs) -> char-separated
      4. guarantee terminal punctuation (both engines pace on it)
    """
    if not text or not text.strip():
        return text
    out = text
    if strip_markdown:
        out = _MD_CHARS.sub("", out)
        out = _EMOJI_RE.sub("", out)
        out = re.sub(r"\s+", " ", out).strip()

    if phone_style == "spell":
        out = _PHONE_CAND_RE.sub(lambda m: _spell(re.sub(r"\D", "", m.group(0))), out)
        out = _DIGIT_RUN_RE.sub(lambda m: _spell(m.group(0)), out)
    elif phone_style == "natural":
        # phone-shaped strings stay as written; long BARE runs still split
        out = _DIGIT_RUN_RE.sub(lambda m: m.group(0) if len(m.group(0)) < 7
                                else _digits_out(m.group(0)), out)
    else:  # "digits" — the SDR default
        def _phone_sub(m: re.Match) -> str:
            if ", " in m.group(0):
                return m.group(0)                 # already normalized
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) < 10:
                return m.group(0)                 # not a phone; leave alone
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]               # drop US country code
            return _digits_out(digits)
        out = _PHONE_CAND_RE.sub(_phone_sub, out)
        out = _DIGIT_RUN_RE.sub(lambda m: _digits_out(m.group(0)), out)

    out = _MIXED_TOKEN_RE.sub(lambda m: ", ".join(m.group(0)), out)
    if out and out[-1] not in _TERMINAL:
        out += "."
    return out
