"""LangGraph builder — V3 production graph.

V1 translated the Retell runtime 1:1. V2 kept the same 9 state nodes and
loop semantics, plus:

  1. Prompts are read from `diallux/prompts/*.md` (editable source of truth,
     same content as the deployed llm.json — verified identical at build time).
  2. A `deterministic` node runs after EVERY state round — your "fix queued in
     the deterministic layer" open items, finally real:
       - leak math: when the three inputs exist and weekly_leak is empty,
         compute it server-side (the Closer can no longer stochastically skip
         the monetization pitch — open item #1 in your AGENTS.md)
       - today prefetch: entering ConfirmSlots with an empty today_date fetches
         /today directly (time never depends on the model remembering the tool)
  3. dvs flow through schema.DynamicVariables (typed; "" defaults; server-owned
     keys protected — see graph/tools.py).
  4. Transitions are hard-gated by the executor (edge `required` params must be
     truthy); the graph structure itself is unchanged because the executor is
     the single door for state swaps.
  5. Optional Postgres checkpointer (LANGGRAPH_CHECKPOINT=postgres) for
     multi-worker durability; MemorySaver fallback per call.

V3 adds (see ITERATIONS.md):
  6. RAG knowledge bases: pgvector top-3 retrieval per turn (the DEPLOYED
     Retell kb_config was {filter_score: 0.6, top_k: 3} — server-side RAG,
     now self-hosted). Inline full-KB fallback when Postgres is unavailable.
  7. VOICE OUTPUT RULES appended to the system prompt (Cartesia's voice-agent
     starter rules) so the model writes text the TTS reads well; the
     deterministic normalizer (media/normalize.py) backs it up per sentence.

One graph invocation == one user utterance (unchanged).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from ..config import Settings
from .. import rag as ragmod
from ..schema import DynamicVariables
from ..state import GraphState, initial_state
from .llm import LITE_NOOP_TOOL, StreamingLLM, build_tool_schemas
from .subst import Substitutor
from .tools import ToolExecutor, _GOODBYE_RE, calculate_monthly_leak, leak_inputs_present

log = logging.getLogger("diallux.graph")   # iter43 T1: round-usage telemetry

# V3: TTS-friendly output rules — adapted from Cartesia's voice-agent starter
# prompt (docs.cartesia.ai, prompting tips). ON in production: the LLM itself
# writes speakable text; media/normalize.py catches what it misses.
VOICE_OUTPUT_RULES = """

## VOICE OUTPUT RULES (your text is spoken by a real-time TTS engine)
- Plain conversational prose. No markdown, bullet points, emoji, or special characters — they get read aloud.
- Always finish sentences with . ? or !
- Phone numbers, verification codes, and any run of 5+ digits: one character at a time, comma-separated, like 3, 1, 2, 4, 0, 0, 1, 2, 3, 4.
- Dollar amounts, dates, times: write them normally ($6,500, September 4th, 2:30 PM) — the engine reads them naturally.
- Tool arguments are never spoken; only your reply text is.
"""

# iter32: mechanical = bookkeeping the model may fire silently ONCE. After a
# silent mechanical round the tool channel closes (forced-speech round).
# end_call included: a blocked silent end_call must be answered with speech
# (the goodbye), never another silent end_call. Same prefix-anchor style as
# FIRE_AND_FORGET (extract_ prefix-match); NO $ anchor — it would break
# extract_* matching (the most common mechanical silent round).
_MECHANICAL_RE = re.compile(
    r"^(extract_|record_reach_details|set_callback_number|intake_completed|"
    r"discovery_completed|closer_completed|offer_completed|"
    r"contact_details_completed|record_booking_outcome|end_call)")

# iter41 T1: sentence boundaries for the TTS dedupe buffer. A sentence flushes
# at . ? ! followed by whitespace (or end of buffer) or at a newline (the
# model's doubles often join with "\n", e.g. "Got it — after-hours…\nGot it…").
_SENTENCE_SPLIT_RE = re.compile(r"[.?!]+(?=\s|$)|\n+")

# iter56 (C2.1): literal {{var}} tokens in the static head — rewritten to the
# stable anchor "CURRENT CALL STATE" when head_strip_vars is on. Matches the
# same token shape the substituter owns (lowercase snake_case names).
_VAR_TOKEN_RE = re.compile(r"\{\{\s*[a-z_][a-z0-9_]*\s*\}\}")

# ---- iter49 T4: live multi-lane retrieval ----------------------------------
# Closing.md has ZERO refer tags (lane A would be empty) and raw "ok thanks
# bye" refuses at arctic 0.24 — the goodbye KB only surfaces via a hand-written
# anchor query (pre-flight: call-closing 4/4). E3: after a successful booking
# the close is a RECAP, not a rescue — swap the anchor.
_CLOSING_ANCHOR = "warm goodbye wrap-up next steps"
_CLOSING_ANCHOR_BOOKED = "recap booking SMS confirmation next steps"
# Pinned pack #6: leak dv values render $-formatted WITH units in lane-A
# queries — raw numerals ("60,000") score below 0.24 and refused the Closer
# Loss-Playback lane in the pre-flight. calculate_monthly_leak stores
# comma-formatted strings ("60,000"); the $ + unit make it "$60,000 a week".
_LEAK_UNITS = {"weekly_leak": "a week", "monthly_leak": "a month"}
# Lane budget (owner 2026-09-13): <=3 engine lanes + <=3 caller lanes per turn.
_MAX_LANES_A = 3
_MAX_LANES_B = 3

# ---- iter52: pinned industry vertical ---------------------------------------
# Resolver stoplist (plan T3): values that NEVER pin — including the literal
# 'header' tag so the header row is unreachable by resolution (defense in
# depth: stoplist + embed-fallback skip + no alias targets it).
_PIN_STOPLIST = {"", "unknown", "n/a", "none", "not sure", "unsure", "tbd",
                 "various", "multiple", "header"}
# Alias map: word-boundary substring, every alias -> exactly ONE tag,
# sorted longest-first. >=2 DISTINCT tag hits = ambiguous -> embed fallback
# (never guess). NO bare "contractor" (a generic "contractor" must not beat
# "roofing contractor"). plumbing/plumber -> the 20th vertical (owner v3).
_PIN_ALIASES: tuple[tuple[str, str], ...] = tuple(sorted((
    ("dental", "Dental Practices"),
    ("dentist", "Dental Practices"),
    ("law firm", "Law Firms (Personal Injury, Family, Immigration)"),
    ("personal injury", "Law Firms (Personal Injury, Family, Immigration)"),
    ("attorney", "Law Firms (Personal Injury, Family, Immigration)"),
    ("auto repair", "Auto Repair & Body Shops"),
    ("auto body", "Auto Repair & Body Shops"),
    ("mechanic", "Auto Repair & Body Shops"),
    ("landscap", "Landscaping, Maintenance & Tree Services"),
    ("lawn", "Landscaping, Maintenance & Tree Services"),
    ("tree service", "Landscaping, Maintenance & Tree Services"),
    ("remodel", "Home Remodeling & Small General Contractors"),
    ("renovation", "Home Remodeling & Small General Contractors"),
    ("general contractor", "Home Remodeling & Small General Contractors"),
    ("roofing", "Residential Roofing & Solar Installers"),
    ("solar", "Residential Roofing & Solar Installers"),   # iter62: dv
    # "solar company" fell to embed-fallback < 0.24 and cached a per-dv
    # miss for the whole call (pinned="" on all 67 spans, cc22e0a0185e)
    ("hvac", "HVAC & Home Services"),
    ("heating", "HVAC & Home Services"),
    ("air conditioning", "HVAC & Home Services"),
    ("med spa", "Medical Spas & Aesthetic Clinics"),
    ("medical spa", "Medical Spas & Aesthetic Clinics"),
    ("aesthetic", "Medical Spas & Aesthetic Clinics"),
    ("vet", "Veterinary Clinics (General Practice, Small Animal)"),
    ("veterinary", "Veterinary Clinics (General Practice, Small Animal)"),
    ("moving", "Local & Regional Moving Companies"),
    ("maid", "Residential Home Cleaning & Maid Services"),
    ("house cleaning", "Residential Home Cleaning & Maid Services"),
    ("cleaning service", "Residential Home Cleaning & Maid Services"),
    ("property management", "Residential Property Management"),
    ("pest", "Pest Control (Residential & Commercial)"),
    ("garage door", "Garage Door & Locksmith Services"),
    ("locksmith", "Garage Door & Locksmith Services"),
    ("appliance repair", "Appliance Repair & Handyman Services"),
    ("handyman", "Appliance Repair & Handyman Services"),
    ("security system", "Residential/Commercial Security & Smart Home"),
    ("smart home", "Residential/Commercial Security & Smart Home"),
    ("alarm", "Residential/Commercial Security & Smart Home"),
    ("msp", "IT Managed Service Providers (MSPs)"),
    ("it services", "IT Managed Service Providers (MSPs)"),
    ("it support", "IT Managed Service Providers (MSPs)"),
    ("managed service", "IT Managed Service Providers (MSPs)"),
    ("urgent care", "Private Clinics (Urgent Care, Physical Therapy, Chiropractic)"),
    ("physical therapy", "Private Clinics (Urgent Care, Physical Therapy, Chiropractic)"),
    ("chiropract", "Private Clinics (Urgent Care, Physical Therapy, Chiropractic)"),
    ("dermatology", "Specialty Clinics (Dermatology & Ophthalmology)"),
    ("ophthalmology", "Specialty Clinics (Dermatology & Ophthalmology)"),
    ("plumbing", "Plumbing"),
    ("plumber", "Plumbing"),
), key=lambda t: -len(t[0])))
# iter52 T5: deep-dive scope triggers — normalized {{industry}} containing
# the keyword adds the KB to that round's lane-B scope (INDEPENDENT of the
# pin; similarity decides; the deep-dive KB is NEVER pinned). Generalizable
# industry-keyword -> extra-KB map. Gated by the same rag_pin_industry
# kill switch (False = byte-exact iter49 live behavior).
_INDUSTRY_KB_TRIGGERS: dict[str, tuple[str, ...]] = {"plumb": ("plumbing",)}


def _fmt_leak_value(name: str, value) -> str:
    """"$60,000 a week" from the tool's raw "60,000" — passthrough when the
    value is already $-formatted or empty."""
    s = str(value or "").strip()
    if not s or "$" in s:
        return s
    unit = _LEAK_UNITS.get(name)
    return f"${s} {unit}" if unit else s


def _fast_flush_cut(text: str, settings) -> int:
    """iter46 T5: fast first-flush boundary for the sentence gate — the first
    "," that follows ≥12 chars, or the whole buffer once it reaches
    tts_gate_fast_first_flush_chars (28 default). 0 = no flush."""
    comma = text.find(",")
    if comma >= 12:
        return comma + 1
    limit = getattr(settings, "tts_gate_fast_first_flush_chars", 28)
    if limit > 0 and len(text) >= limit:
        return len(text)
    return 0


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, pure python (1536 dims is trivial work)."""
    num = da = db = 0.0
    for x, y in zip(a, b):
        num += x * y
        da += x * x
        db += y * y
    if not da or not db:
        return 0.0
    return num / ((da ** 0.5) * (db ** 0.5))


def _norm_sentence(s: str) -> str:
    """Dedupe key: case-insensitive, whitespace-collapsed sentence text."""
    return " ".join(s.split()).casefold()


# iter65 T2: spoken-sentence sanitizer — runs BEFORE the dedupe window on
# every flushed sentence (mid-stream cut + stream-end pending). Model
# output/history/final content are NEVER touched; only the spoken stream.
_FUNC_CALLS_RE = re.compile(r"\bfunction_calls\b", re.IGNORECASE)
_CODEISH_CHARS = set("{}[]=>")


def _sanitize_spoken(text: str, settings) -> str | None:
    """None = drop the sentence from speech (code/JSON artifact).
    1) strip a verbatim `function_calls` protocol token;
    2) drop if any structural code char { } [ ] => OR the JSON-ish pair `":`
       (single ':' = natural times "10:30"; single '"' = quotes — kept)."""
    if not getattr(settings, "tts_sanitize_tokens", True):
        return text
    t = _FUNC_CALLS_RE.sub("", text).strip()
    if not t:
        return None
    if any(c in t for c in _CODEISH_CHARS) or '":' in t:
        return None
    return t


async def _embed_stem(runtime, sentence: str) -> list[float] | None:
    """iter59 T4: ONE query embedding of a spoken question stem via the
    store's reused embed client (rag.py embed_query). None on store-absent /
    failure — the caller keeps the exact-only fallback and never blocks
    speech. Best-effort, never raises."""
    try:
        store = await runtime._resolve_kb_store()
        if store is None:
            return None
        return await store.embed_query(sentence)
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a)
    db = sum(y * y for y in b)
    if not da or not db:
        return 0.0
    return num / ((da ** 0.5) * (db ** 0.5))


async def _semantic_dup(runtime, sentence: str,
                        stem_window: list[dict]) -> bool:
    """iter59 T4: True when the question's stem is a near-verbatim re-ask
    of a question already spoken this call (cosine >= threshold). ONE
    embed; embed failure -> False (exact-only fallback, never block)."""
    try:
        vec = await _embed_stem(runtime, sentence)
        if vec is None:
            return False
        thr = float(runtime.settings.tts_dedupe_cos_threshold)
        for entry in stem_window:
            prev = entry.get("vec")
            if not prev:
                continue
            if _cosine(vec, prev) >= thr:
                return True
        return False
    except Exception:
        return False


def _spoken_entry_has(stem_window: list[dict], key: str) -> bool:
    """iter59 T4: the stem key already in the window (avoid double-embed of
    the same question text across rounds of a turn)."""
    return any(e.get("text") == key for e in stem_window)

# iter33 Phase B: engine-owned interstitial fillers. Spoken via tts_token —
# never an LLM call. Anti-repeat within call (mirrors general_prompt's rule).
_ENGINE_FILLERS = [
    "One sec — finishing that up.",
    "Alright, getting that locked in for you.",
    "Just confirming those details now.",
    "Great — booking that for you as we speak.",
    "Almost there, locking in your time.",
    "Perfect, just a moment while I finish this.",
]
_PHASEB_ABORT = {"gate_failed", "incomplete", "unfixable_format", "book_failed",
                 "error", "invalid_uid", "uid_not_found", "request_failed"}
_PHASEB_SLOW = {"verify_lead_data", "create_livecall_booking", "record_booking_uid"}


def _chain_args(step: str, dvs: dict, booking_uid: str = "") -> dict:
    """Deterministic chain args — byte-equal to the model's templates today."""
    if step == "validate_lead":
        return {"first_name": dvs.get("first_name", ""), "last_name": dvs.get("last_name", ""),
                "company_name": dvs.get("company_name", ""),
                "callback_number": dvs.get("callback_number", ""),
                "prospect_timezone": dvs.get("prospect_timezone", ""),
                "selected_time": dvs.get("selected_time", ""),
                "missed_calls_per_week": dvs.get("missed_calls_weekly", ""),
                "avg_job_value": dvs.get("avg_job_value", ""),
                "close_rate_pct": dvs.get("close_rate_pct", "")}
    if step == "verify_lead_data":
        return {"first_name": dvs.get("first_name", ""), "last_name": dvs.get("last_name", ""),
                "company_name": dvs.get("company_name", ""),
                "callback_number": dvs.get("callback_number", ""),
                "prospect_timezone": dvs.get("prospect_timezone", ""),
                "selected_time": dvs.get("selected_time", ""),
                "slot_options": dvs.get("requested_slot", "")}
    if step == "create_livecall_booking":
        name = f"{dvs.get('first_name', '')} {dvs.get('last_name', '')}".strip()
        return {"account_id": "diallux_live", "timezone": dvs.get("prospect_timezone", ""),
                "time": dvs.get("selected_time", ""), "name": name or dvs.get("company_name", ""),
                "email": "jaydiallux@gmail.com", "attendeePhoneNumber": dvs.get("callback_number", ""),
                "title": f"Dialux Live call for {dvs.get('company_name', '')}",
                "notes": f"Industry: {dvs.get('industry', '')} | Pain points: {dvs.get('pain_points', '')}",
                "slot_reservation_uids": dvs.get("slot_reservation_uids", ""),
                "booking_intent": dvs.get("booking_intent", ""), "preferred_time": ""}
    if step == "record_booking_uid":
        return {"booking_uid": booking_uid}
    return {}


class CallRuntime:
    """Everything scoped to ONE phone call: graph, llm, executor, tracer."""

    def __init__(
        self,
        settings: Settings,
        llm_json: dict,
        tracer=None,
        llm: StreamingLLM | None = None,
        http_client=None,
        checkpointer=None,
        kb_store=...,   # ellipsis = resolve lazily; None = force off; object = injected (tests/eval)
    ):
        self.settings = settings
        self.llm_json = llm_json
        self.states: dict[str, dict] = {s["name"]: s for s in llm_json.get("states", [])}
        # iter87 VERSION B fire law: FSM successor map derived from the SAME
        # edges that generate the transition_to_* tool schemas (llm.py
        # build_tool_schemas) — one source of truth, no hand-written table.
        self._state_successors: dict[str, list[str]] = {
            name: [e["destination_state_name"] for e in cfg.get("edges", [])]
            for name, cfg in self.states.items()}
        self.subst = Substitutor(settings.knowledge_base_dir)
        self.tracer = tracer
        self.llm = llm or StreamingLLM(settings)
        # iter40 L2: shared warm store for the ASAP availability payload.
        # {"payload": <webhook body>, "ts": time.monotonic(), "tz": <IANA>}
        # Written by _warm_slots (bg task), read by ToolExecutor prefetch.
        self._slots_warm: dict = {}   # {date: {"payload","ts","tz","task"}} (iter40 L2)
        self._slots_warm_task: asyncio.Task | None = None
        self.executor = ToolExecutor(settings, llm_json, tracer=tracer, http_client=http_client,
                                     slots_warm=self._slots_warm)
        self.prompts = self._load_prompts()
        self._kb_store = kb_store
        self._kb_store_resolved = kb_store is not ...
        self._kb_stats = {"rag_turns": 0, "rag_ms_total": 0.0, "rag_chars": 0}
        self._rag_cache: dict = {}          # iter29: per-(state, query) retrieval cache
        # iter31 C3: byte-stable prefix — static head + tool schemas cached per
        # (state, expand_kb); RAG chunks frozen per state-visit.
        self._head_cache: dict = {}
        self._tools_cache: dict = {}
        self._frozen_state: str | None = None
        self._frozen_chunks: list = []
        # iter44 T6c: drift-requery stash — vector + text of the query that
        # produced _frozen_chunks (vec may be None until lazily embedded).
        self._frozen_query_vec: list | None = None
        self._frozen_query_text: str = ""
        # iter44 T6b: transition RAG prefetch — staged chunks consumed by the
        # state_node rag lane (pop), in-flight task registry + per-state latch.
        self._rag_staging: dict[str, dict] = {}
        self._rag_staging_tasks: dict[str, asyncio.Task] = {}
        self._frozen_rag_tail: str = ""          # iter44 append-only: byte-stable knowledge tail
        self._rag_delta_chunks: list[dict] = []  # chunks appended AFTER history on drift
        # iter44 T4: loud no-KB warning is logged once per call, not per round.
        self._inline_guard_logged = False
        # iter44 T6c: drift-embed failure backoff (60 s, monotonic)
        self._drift_failed_at = 0.0
        # iter46 T2: async drift check — one serialized task per state,
        # cancel-superseded when a newer caller message lands.
        self._drift_tasks: dict[str, asyncio.Task] = {}
        # iter43 T3: async prompt-cache prewarm — latched per state, tasks
        # cancelled in aclose()
        self._prewarmed: set[str] = set()
        self._warm_tasks: set[asyncio.Task] = set()
        # iter46 T4c: newest warm task per state — the bounded entry await
        # targets this one (the fire-early warm may be superseded by the
        # post-execute warm with corrected dvs bytes).
        self._latest_warm: dict[str, asyncio.Task] = {}
        # iter55 T0: per-call warm failure diag ({key: {error, retried_lite}})
        # — read by the warm done-callback to mark the span degraded/failed.
        self._warm_diag: dict[str, dict] = {}
        # iter49 T3: live-retrieve bg tasks (key = state_name) + the previous
        # turn's landed fresh set per state (the degrade source). The eager
        # EOT fires the task; the round bounded-awaits it (rag_live_await_ms)
        # and degrades to _live_prev — never a stall. Cancelled in aclose().
        self._live_tasks: dict[str, asyncio.Task] = {}
        self._live_prev: dict[str, dict] = {}
        self._live_msgs: dict[str, str] = {}
        # iter59 T3: turn-keyed supersede bookkeeping. Keys are the composite
        # "<state>|<lane>" ("full" | "laneA" | "laneB"):
        #   _live_seq        monotonically increasing fire seq per state
        #   _live_task_seq   the seq of the task currently stored per key
        #   _live_consumed   highest seq CONSUMED (fresh) per state
        #   _live_consumed_res the last consumed FRESH raw/merged pieces per
        #                    state — the timeout fallback degrades to THIS,
        #                    never to _live_prev's unconsumed overwrite (C4)
        self._live_seq: dict[str, int] = {}
        self._live_task_seq: dict[str, int] = {}
        self._live_consumed: dict[str, int] = {}
        self._live_consumed_res: dict[str, dict] = {}
        # iter60 AUD-3: per-turn merged-set stash — rounds 2+ of the SAME
        # turn reuse round-1's merged set (the seq floor rightly rejects
        # stale laneA but also rejected THIS turn's laneA on tool rounds).
        self._live_merged_turn: dict[str, dict] = {}
        # iter59 T3: consume-mode label for the node's rag span
        # ("live-async" | "live-sync" | "hybrid").
        self._live_mode_label: str = "live-async"
        # iter52 T5: the pinned industry vertical — {"dv", "tag", "chunks"}
        # fetched ONCE per call (per dv value) inside the bg _live_retrieve
        # task; re-resolved only when the dv value changes; reset per call
        # in initial_state(). The pin renders OUTSIDE merge_lane_chunks
        # (own block, own budget) in the post-history delta.
        self._pinned_industry: dict = {}
        # iter49 T6 (T4b): the state whose head the LAST round rendered —
        # the state-entry-lite trigger compares against it. None until a
        # non-turn-1-lite round runs (the turn-1 lite round renders no state
        # head, so the turn-2 entry into the initial state still counts as
        # an entry). Reset per call in initial_state().
        self._prev_round_state: str | None = None
        # iter96: per-state STAGE frames — {"block": str, "dvs": flat dict}
        # keyed by state name. The block (state frame + RAG stage frame) is
        # rendered ONCE per state entry and rides the cached prefix
        # ([tools][head][stage][history]) byte-stable across the rounds of
        # the state; the trailing delta carries only the changed dvs values.
        # Reset per call in initial_state().
        self._stage_blocks: dict[str, dict] = {}
        self.graph = self._build(checkpointer)

    async def aclose(self):
        for task in list(self._warm_tasks):
            task.cancel()
        for task in self._drift_tasks.values():
            task.cancel()
        for task in self._live_tasks.values():
            task.cancel()
        if self._warm_tasks:
            await asyncio.gather(*self._warm_tasks, return_exceptions=True)
        if self._drift_tasks:
            await asyncio.gather(*self._drift_tasks.values(), return_exceptions=True)
        if self._live_tasks:
            await asyncio.gather(*self._live_tasks.values(),
                                 return_exceptions=True)
        self._warm_tasks.clear()
        self._drift_tasks.clear()
        self._live_tasks.clear()
        await self.executor.aclose()

    # ------------------------------------------------------------------ #
    # iter43 T3: async prompt-cache prewarm. The EXACT next-round payload
    # shape (tools FIRST + stripped static head + hysteretic history window)
    # is sent with a 16-token cap; OpenAI caches the prefix bytes, the caller
    # never waits, output is discarded. Cold-vs-cached TTFT gap (~800 ms at
    # iter21) closes without touching the hot path.
    def warm_prompt_cache(self, state_name: str, history: list[dict] | None = None,
                          dvs: dict | None = None, lite: bool = False,
                          force: bool = False) -> None:
        """Fire-and-forget prefix warm for `state_name`'s FULL payload shape.
        Latched once per state per call; no-op outside a running loop.
        iter46 T3: `lite=True` warms the first_turn_lite shape instead (no
        tools, no tail). iter46 T4a: `force=True` re-fires a warm for an
        already-latched state — the post-execute transition warm supersedes
        the tool-detection warm with the post-patch dvs bytes.
        `_latest_warm[<key>]` always points at the newest task (T4c bounded
        await awaits THAT one at state entry; the non-lite key is the bare
        state name, the lite warm registers under `lite:<state>`)."""
        if not self.settings.prompt_prewarm:
            return
        key = f"lite:{state_name}" if lite else state_name
        if key in self._prewarmed and not force:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._prewarmed.add(key)
        fired = time.time()                # iter55 T0: warm fire timestamp
        task = asyncio.get_running_loop().create_task(
            self._warm(state_name, history or [], dvs, lite=lite))
        self._warm_tasks.add(task)
        # iter55 T0: done-callback emits the `warm:{key}` span (fire→complete/
        # fail per shape) — fail-safe, never raises into the event loop.
        task.add_done_callback(self._make_warm_done_cb(key, state_name, lite,
                                                       fired))
        # iter48 Commit B (M12): lite warms are awaitable too — registered
        # under `lite:<state>` so the turn-1 (lite) round can bounded-await
        # them at entry (audit A5: the lite task was previously invisible to
        # _latest_warm → live turn-1 cache_read=0). Collides with nothing.
        if lite:
            self._latest_warm[key] = task
        else:
            self._latest_warm[state_name] = task

    def _make_warm_done_cb(self, key: str, state_name: str, lite: bool,
                           fired: float):
        """iter55 T0: done-callback factory for warm tasks — emits the
        `warm:{key}` span (shape, state, fired_at, ms, outcome, degraded,
        error) + an info log. Fail-safe: its own try/except, same contract
        as the tracer."""
        def _cb(task: asyncio.Task) -> None:
            try:
                ms = round((time.time() - fired) * 1000)
                if task.cancelled():
                    outcome = "cancelled"
                elif task.exception() is not None:
                    outcome = "failed"
                else:
                    outcome = "ok"
                diag = self._warm_diag.get(key)
                degraded = bool(diag and diag.get("retried_lite"))
                error = diag.get("error") if diag else None
                if outcome == "failed" and error is None:
                    error = str(task.exception())
                log.info("warm done key=%s shape=%s outcome=%s degraded=%s ms=%s",
                         key, "lite" if lite else "full", outcome, degraded, ms)
                if getattr(self.settings, "warm_observability", True) \
                        and self.tracer:
                    self.tracer.span(f"warm:{key}", metadata={
                        "shape": "lite" if lite else "full",
                        "state": state_name,
                        "fired_at": fired,
                        "ms": ms,
                        "outcome": outcome,
                        "degraded": degraded,
                        "error": error,
                    })
            except Exception:
                pass
        return _cb

    async def _warm(self, state_name: str, history: list[dict],
                    dvs: dict | None = None, lite: bool = False) -> None:
        """Warm request — BYTE-IDENTICAL to the state_node path (iter44 T1:
        composed through the SAME _build_messages as the hot path — same
        static head, same full tool array, same hysteretic history window,
        same STATE-BLOCK tail and RAG section)."""
        try:
            # iter46 T4b: normalize the dvs through the SAME schema path as
            # ingest's frozen STATE-BLOCK (to_flat field order) — a raw
            # dvs+patch merge appends NEW keys at the END, diverging from the
            # entry round's tail bytes and capping the transition cache at
            # tools+head (the audit's tax #4).
            dvs = DynamicVariables.from_flat(dvs or {}).to_flat()
            if lite:
                # iter46 T3: turn-1 hot payload = [lite head][greeting history
                # message][user msg] — the warm covers the byte-prefix BEFORE
                # the user message exists (it fires during the greeting).
                # iter48b: binds LITE_NOOP_TOOL — warm shape == hot shape, and
                # gpt-5.4 only caches tool-bearing requests (FIND-5).
                messages = [{"role": "system", "content": self._lite_head()}] + \
                    self._history_window(history, self.settings.history_window,
                                         self.settings.history_trim_step)
                await self.llm.warm(messages, [LITE_NOOP_TOOL])
                return
            store = await self._resolve_kb_store()
            system, expand_kb = self._head_for(state_name, store)
            tools = self._tools_cache.get((state_name, expand_kb))
            if tools is None:
                tools = build_tool_schemas(
                    self.states[state_name],
                    self.llm_json.get("general_tools", []),
                    lambda text: self.subst.subst(text, {}, kb=expand_kb),
                    {},
                )
                self._tools_cache[(state_name, expand_kb)] = tools
            tail = ""
            # iter56 (C2.5): under state_in_delta the hot path's pre-history
            # prefix carries NO state-block (it rides the post-history delta,
            # which never rides cache) — the warm prefills EXACTLY
            # [tools][head][history], the new append-only prefix. Warming a
            # state-block here would warm the WRONG shape and the latch
            # would poison the round. The staged-RAG branch (revert mode
            # only) stays unchanged behind the flag.
            if self.settings.prewarm_byte_exact \
                    and not getattr(self.settings, "state_in_delta", False):
                tail = self._state_block(dvs)
                # iter49 T4: under rag_live_retrieve the hot path's pre-history
                # prefix carries NO chunks (fresh chunks render post-history)
                # — the staged-RAG tail would make warm bytes != entry bytes
                # at EVERY transition (cold re-prefill). The warm now prefills
                # EXACTLY [tools][head][state-block][history].
                if not getattr(self.settings, "rag_live_retrieve", False) \
                        and store is not None and self.kb_slugs_for(state_name):
                    # staged chunks (this prefetch or a pending transition one)
                    # make the warm tail byte-equal to the next state entry's.
                    staged = await self._await_staging(state_name)
                    if staged is None:
                        staged = await self._stage_rag(state_name, dvs or {})
                    if staged is None:
                        staged = self._rag_staging.get(state_name)
                    tail += ragmod.render_knowledge_section(
                        staged["chunks"] if staged else [])
            # iter96: under stage-static the hot ENTRY ACK round is
            # [tools][head][stage dvs-frame][history][directive] — the warm
            # prefills EXACTLY that shape ([tools][head][stage-frame]
            # [history], NO RAG — the ack runs no retrieval lanes): the
            # ack reads the warm at hit-class, and the state's first heavy
            # round extends the frame with the RAG once. Without this the
            # warm would cover only [tools][head] and the frame would
            # re-bill at every state entry.
            warm_stage = ""
            if getattr(self.settings, "state_in_delta", False) \
                    and getattr(self.settings, "delta_stage_static", False):
                warm_stage = self._state_block(dvs)
            messages = self._build_messages(state_name, history, tail, system,
                                            stage=warm_stage)
            await self.llm.warm(messages, tools)
        except Exception as exc:
            # iter48 Commit B (M13): warm failures are no longer swallowed at
            # info level — a full-shape warm that fails gets ONE lite-shape
            # retry ([lite head][history], no tools/tail) so the prefix still
            # warms partially; a lite retry that also fails stays silent.
            log.warning("prewarm failed for %s: %s — one lite-shape retry",
                        state_name, exc)
            # iter55 T0 (F-08): record the failure for the warm done-callback
            # — a full-fail→lite-retry is reported degraded in the span.
            self._warm_diag[f"lite:{state_name}" if lite else state_name] = {
                "error": str(exc), "retried_lite": not lite}
            try:
                lite_messages = [{"role": "system",
                                  "content": self._lite_head()}] + \
                    self._history_window(history, self.settings.history_window,
                                         self.settings.history_trim_step)
                await self.llm.warm(lite_messages, [LITE_NOOP_TOOL])
            except Exception:
                pass

    async def _await_warm(self, state_name: str,
                          wait_ms: int | None = None) -> None:
        """iter46 T4c: bounded await of the in-flight warm for this state at
        state entry. asyncio.shield keeps the warm running for the NEXT round
        even on timeout; the cap only bounds the ADDED wait (the primary
        transition fix is fire-at-detection + the dvs fix). Flag 0 = off
        (exact iter44 behavior). iter48 Commit B (M10): `wait_ms` overrides
        the mid-turn flag — None → prewarm_entry_wait_ms (back-compat).
        iter55 T0: the outcome (landed/timeout/already_done/no_task/
        task_error) + waited_ms are logged and — under warm_observability —
        spanned as `await_warm`.
        iter95 (savage piece 1): with prewarm_abort_cap_ms > 0 the bound is
        the ABORT cap - a warm still streaming at the cap is cancelled
        (outcome `aborted`); the prefill cache it wrote survives server-side
        and the real round reads it hit-class. cap=0 restores the legacy
        wait-shield (outcome `timeout`, warm keeps running). A completed
        warm is adopted before any cap applies (already_done). Note: this
        gate sits inside the prewarm_entry_wait_ms off-gate (wait_ms<=0
        returns early - off is off, iter44 behavior)."""
        if wait_ms is None:
            wait_ms = getattr(self.settings, "prewarm_entry_wait_ms", 0)
        if wait_ms <= 0:
            return
        task = self._latest_warm.get(state_name)
        cap_ms = 0
        bound_ms = wait_ms
        if task is None:                            # F-09: warm never fired
            outcome, waited = "no_task", 0
        elif task.cancelled():                      # iter95: previously
            outcome, waited = "aborted", 0          # aborted — nothing to adopt
        elif task.done():
            outcome, waited = "already_done", 0
        else:
            # iter95 (savage piece 1): cancel-not-wait. A warm that is still
            # streaming at the abort cap is CANCELLED — the prefix cache it
            # wrote at prefill survives server-side and the real round reads
            # it at hit-class TTFT. cap=0 = legacy iter46 wait-shield.
            cap_ms = int(getattr(self.settings, "prewarm_abort_cap_ms", 0)
                         or 0)
            # ceiling: the abort cap never LENGTHENS a context's own bound
            # (mid-turn 100 stays 100; EOT 500 and dest 2500 clamp to 400)
            bound_ms = min(wait_ms, cap_ms) if cap_ms > 0 else wait_ms
            t0 = time.perf_counter()
            try:
                await asyncio.wait_for(asyncio.shield(task), bound_ms / 1000.0)
                outcome = "landed"
            except asyncio.TimeoutError:
                if cap_ms > 0:
                    task.cancel()                   # abort mid-completion:
                    outcome = "aborted"            # cache survives, no queue
                else:
                    outcome = "timeout"            # shield keeps the warm
            except Exception:
                outcome = "task_error"   # shield keeps the warm alive
            waited = round((time.perf_counter() - t0) * 1000)
        log.info("await_warm key=%s outcome=%s waited_ms=%s cap_ms=%s",
                 state_name, outcome, waited, bound_ms)
        if getattr(self.settings, "warm_observability", True) and self.tracer:
            self.tracer.span("await_warm", metadata={
                "key": state_name, "outcome": outcome,
                "waited_ms": waited, "cap_ms": bound_ms})

    # ------------------------------------------------------------------ #
    # iter40 L2: slots prefetch — bg-warm the availability payload.
    # FIRES on ConfirmSlots entry + on an ok transition INTO ConfirmSlots.
    # WARM CALL = byte-identical to the model's first slots call (frozen OTEL
    # t40 evidence: {"account_id", "timezone", "slot_target_date",
    # "current_reservation_uids": ""} — freeze mode, holds frozen). The plan's
    # no-date pin was corrected 2026-09-09: the LIVE endpoint REQUIRES
    # slot_target_date (no-date → no_start_time; proven by signed probe), so
    # the warm covers the two days the ConfirmSlots flow ever offers —
    # today + tomorrow. Serve only on exact arg-match (same date, empty
    # current_reservation_uids, no preferred_time) — any other call goes live.
    # Warm holds linger until Cal.com auto-expiry when not served (noted in
    # 02_latency_fixes.md). Fresh (same tz+date, ok, < ttl) → no-op; in-flight
    # warm → no-op; never sets engine_fired.
    def warm_slots_async(self, dvs: dict) -> None:
        if not self.settings.slots_prefetch:
            return
        tz = str(dvs.get("prospect_timezone") or "").strip()
        if not tz:
            return
        today = self._slots_warm_today(dvs, tz)
        if today is None:
            return
        ttl = float(self.settings.slots_prefetch_ttl_s)
        now = time.monotonic()
        wanted = [today.isoformat(), (today + timedelta(days=1)).isoformat()]
        pending = False
        for date in wanted:
            warm = self._slots_warm.get(date) or {}
            if (warm.get("tz") == tz
                    and isinstance(warm.get("payload"), dict)
                    and warm["payload"].get("ok")
                    and (now - warm.get("ts", 0.0)) < ttl):
                continue
            task = self._slots_warm.get(date, {}).get("task")
            if task is not None and not task.done():
                pending = True
                continue
            try:
                self._slots_warm[date] = {
                    "task": asyncio.get_running_loop().create_task(
                        self._warm_slots(tz, date))}
            except RuntimeError:
                return
        self._slots_warm_task = None   # tasks now live per-date in the store

    def _slots_warm_today(self, dvs: dict, tz: str):
        """Today in the prospect's tz: dvs today_date first, zoneinfo fallback."""
        raw = str(dvs.get("today_date") or "").strip()
        if raw:
            try:
                return datetime.strptime(raw[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
        try:
            return datetime.now(ZoneInfo(tz)).date()
        except Exception:
            return None

    async def _warm_slots(self, tz: str, date: str) -> None:
        tool = self.executor._tool_def("query_livecall_slots")
        if not tool or not tool.get("url"):
            return
        payload = {"args": {"account_id": "diallux_live", "timezone": tz,
                            "slot_target_date": date,
                            "current_reservation_uids": ""}}  # freeze-mode (model parity)
        raw = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json",
                   "X-Retell-Signature": self.executor._sign(raw, tool["url"])}
        t0 = time.perf_counter()
        try:
            resp = await self.executor.http.post(tool["url"], content=raw, headers=headers)
            try:
                body = resp.json()
            except ValueError:
                body = {}
        except Exception as exc:
            body = {"ok": False, "error": "prefetch_failed", "message": str(exc)[:200]}
        # mutate IN PLACE — the executor reads this same dict object
        self._slots_warm[date] = {"payload": body, "ts": time.monotonic(), "tz": tz}
        if self.tracer:
            self.tracer.span("slots_prefetch", output={
                "ok": bool(isinstance(body, dict) and body.get("ok")),
                "slots": len((body or {}).get("slots") or []),
                "ms": round((time.perf_counter() - t0) * 1000, 1), "tz": tz,
                "date": date})

    async def warm_rag(self, state_name: str | None = None, dvs: dict | None = None):
        """iter21: warm the RAG path BEFORE turn 1 (embedding client + pgvector
        pool). Fired by CallSession.start() during the greeting — kills the
        ~1.7s cold retrieval that otherwise lands on the caller's first turn.
        iter44: the chunks are STAGED (not discarded) so the byte-exact
        prewarm and the first state entry reuse the same retrieval.
        iter49 T4: under rag_live_retrieve nothing consumes staging — the
        cold-start value left is the EMBEDDER session (arctic ONNX ~833ms)
        + the pgvector pool; warm those, skip the dead retrieval."""
        try:
            state_name = state_name or self.llm_json.get("starting_state", "Intake")
            if getattr(self.settings, "rag_live_retrieve", False):
                store = await self._resolve_kb_store()
                if store is not None:
                    embed_q = getattr(store, "embed_query", None)
                    if embed_q is not None:
                        await embed_q("warm")
                return
            task = self._spawn_stage_rag(state_name, dvs or {})
            if task is not None:
                await task       # registry-registered: the prewarm's
                                 # _await_staging reuses THIS retrieval
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    async def _resolve_kb_store(self):
        """Lazily attach the process-wide KBStore (pgvector) once.
        iter44 T2: the resolve latch is set only AFTER a successful attach —
        one transient failure must never latch None (inline-heavy mode) for
        the whole session; the next caller retries."""
        if self._kb_store_resolved:
            return self._kb_store
        try:
            store = await ragmod.get_kb_store(self.settings)
        except Exception as exc:
            log.warning("kb store resolve failed (will retry): %s", exc)
            return None
        self._kb_store_resolved = True
        self._kb_store = store
        return store

    # ------------------------------------------------------------------ #
    # iter44 T4: expansion decision — a REAL store means retrieval (markers
    # stripped). No store: inline expansion ONLY when the operator asked for
    # it (fallback_inline_kb=True, or the explicit rag_mode="inline" V1
    # mode). The accidental pgvector-outage fallback to the ~16k blob is dead.
    def _expand_kb(self, kb_store) -> bool:
        if kb_store is not None:
            return False
        return self.settings.fallback_inline_kb or \
            getattr(self.settings, "rag_mode", "auto") == "inline"

    def _head_for(self, state_name: str, kb_store) -> tuple[str, bool]:
        """System head + expand flag shared by the hot path and the warm path
        (byte-identical bytes for the same (state, store) decision).
        iter49 T4 rag_keep_markers: True keeps the ##slug-kb## references in
        the cached head (A/B whether the model grounds better with them);
        default False = stripped, iter48 exact."""
        expand_kb = self._expand_kb(kb_store)
        system = self.static_head(state_name, expand_kb)
        # iter49c T4 fix: the old `or not expand_kb` made rag_keep_markers
        # INERT whenever a store is present (store => expand False => always
        # strip — the A/B could never keep markers). Strip when a store is
        # present and the flag is off; with no store + inline the markers
        # were already substituted; with no store + no expansion strip so
        # literal markers never leak to the model.
        strip = (kb_store is not None
                 and not getattr(self.settings, "rag_keep_markers", False)) \
            or (kb_store is None and not expand_kb)
        if strip:
            system = ragmod.strip_kb_markers(system)
        return system, expand_kb

    # ------------------------------------------------------------------ #
    def _entry_lite_off_states(self) -> set[str]:
        """iter49 T6: per-state opt-out parsed from state_entry_lite_off
        (comma list) — states that must tool-call on entry."""
        return {s.strip() for s in
                str(getattr(self.settings, "state_entry_lite_off", "") or "").split(",")
                if s.strip()}

    def _entry_lite_head(self, state_name: str, kb_store) -> str:
        """iter49 T6: the state-entry ack head — the state's kb=False head.
        Strip decision mirrors _head_for so the bytes are IDENTICAL to the
        heavy round's head when a store is present (markers kept only under
        rag_keep_markers + store); with no store the ack stays light (kb
        forced False — never the inline blob, even in the degraded mode)."""
        system = self.static_head(state_name, expand_kb=False)
        keep = bool(getattr(self.settings, "rag_keep_markers", False))
        if not (keep and kb_store is not None):
            system = ragmod.strip_kb_markers(system)
        return system

    # ------------------------------------------------------------------ #
    # iter44 T1: the ONE composer for both the hot path and the warm path —
    # identical bytes for identical (system, tail, history) inputs.
    # iter44 T3 layout (tail_before_history): [head][tail][history] so the
    # cached prefix (tools + head + tail) grows round-to-round; the history
    # window (16/8 hysteretic) bounds the ceiling (~5k tokens). False =
    # legacy [head][history][tail].
    # iter56: with state_in_delta the hot path passes tail="" (no pre-history
    # state-block) and the delta carries state → RAG → directive; the
    # history window defaults 0 = FULL append-only (cache climbs with it).
    def _build_messages(self, state_name: str, history: list[dict],
                        tail: str, system: str, delta: str = "",
                        stage: str = "") -> list[dict]:
        hist = self._history_window(history, self.settings.history_window,
                                    self.settings.history_trim_step)
        messages = [{"role": "system", "content": system}]
        if stage.strip():
            # iter96: the stage-static frame rides BEFORE the history — it
            # is part of the cached prefix ([tools][head][stage][history...],
            # byte-stable across the rounds of the state); the trailing
            # delta below carries only the true per-round change.
            messages.append({"role": "system", "content": stage})
        has_tail = bool(tail.strip())
        if has_tail and self.settings.tail_before_history:
            messages.append({"role": "system", "content": tail})
        messages.extend(hist)
        if has_tail and not self.settings.tail_before_history:
            messages.append({"role": "system", "content": tail})
        # iter44 append-only: drift-refetched NEW knowledge goes AFTER the
        # history as its own system block — the [head][tail][history] prefix
        # stays byte-stable, so the cached prefix keeps climbing with history
        # while the delta (small) re-bills per round. Never mutate mid-prompt.
        if delta.strip():
            messages.append({"role": "system", "content": delta})
        return messages

    # ------------------------------------------------------------------ #
    # iter44 T6b: transition RAG prefetch. An ok transition_to_X fires a bg
    # retrieval for X's scope; the chunks land in _rag_staging and the
    # state_node rag lane consumes them at state entry (~0 ms critical path).
    # Prefetch failure degrades to today's sync fetch (never raises).
    def warm_rag_async(self, state_name: str, dvs: dict | None = None) -> None:
        """Fire-and-forget RAG staging for `state_name`. iter44: NO per-state
        latch — every ok transition into a state (re)stages its scope, so a
        revisited state's entry is prefetched too. The in-flight task dedupe
        in _spawn_stage_rag prevents duplicate concurrent embeds."""
        if not self.settings.rag_prefetch_on_transition:
            return
        # iter49 T4: transition prefetch is part of the iter48 diagnosis
        # (stale caller text staged ahead) and dead work under live-retrieve
        # (every round retrieves fresh) — inert, like the drift lane.
        if getattr(self.settings, "rag_live_retrieve", False):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._spawn_stage_rag(state_name, dvs or {})

    def _spawn_stage_rag(self, state_name: str, dvs: dict):
        """One in-flight staging task per state (a pending task is reused,
        not duplicated). Returns the task, or None outside a loop."""
        pending = self._rag_staging_tasks.get(state_name)
        if pending is not None and not pending.done():
            return pending
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None
        task = asyncio.get_running_loop().create_task(
            self._stage_rag(state_name, dvs))
        self._rag_staging_tasks[state_name] = task
        self._warm_tasks.add(task)      # cancelled in aclose()
        return task

    async def _await_staging(self, state_name: str) -> dict | None:
        """Await a pending staging task for `state_name`, if any."""
        task = self._rag_staging_tasks.get(state_name)
        if task is None or task.done():
            return self._rag_staging.get(state_name)
        try:
            return await task
        except Exception:
            return None

    async def _stage_rag(self, state_name: str, dvs: dict,
                         user_msg: str = "") -> dict | None:
        """Retrieve `state_name`'s scope and stage the chunks. Returns the
        staged dict, or None on any failure (caller falls back to the sync
        fetch — never worse than today)."""
        store = await self._resolve_kb_store()
        if store is None:
            return None
        scope = self.kb_slugs_for(state_name)
        if not scope:
            return None
        query, _dv_vals = self.refer_query(state_name, dvs, user_msg)
        if not query.strip():
            # iter48 M2: no anchor at all — never embed the empty string;
            # return None so the caller falls back to the sync fetch.
            return None
        vec = None
        embed_q = getattr(store, "embed_query", None)
        if embed_q is not None:
            try:
                vec = await embed_q(query)
            except Exception:
                vec = None
        try:
            # real stores reuse the staged vec; fakes without embed_query
            # still stage (retrieve does its own embed).
            kwargs = {"vec": vec} if vec is not None else {}
            chunks = await store.retrieve(query, scope, **kwargs)
        except Exception as exc:
            log.warning("rag prefetch failed for %s (%s); sync fallback", state_name, exc)
            return None
        staged = {"query": query, "chunks": chunks, "vec": vec}
        self._rag_staging.setdefault(state_name, staged)
        return staged

    # ------------------------------------------------------------------ #
    # iter46 T2: async drift check. Serialized per state (one task handle,
    # cancel-supersede on a newer caller message — H2b's race guard); the
    # `_drift_failed_at` backoff applies inside the task. Never raises.
    def _spawn_drift_check(self, state_name: str, dvs: dict, user_msg: str,
                           scope: list[str], kb_store) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        pending = self._drift_tasks.get(state_name)
        if pending is not None and not pending.done():
            pending.cancel()      # superseded by the newer query
        task = asyncio.get_running_loop().create_task(
            self._drift_check(state_name, dvs, user_msg, scope, kb_store))
        self._drift_tasks[state_name] = task

    async def _drift_check(self, state_name: str, dvs: dict, user_msg: str,
                           scope: list[str], kb_store) -> None:
        """BG drift lane: embed base (lazily) + round query, cosine, on drift
        re-retrieve and append NEW chunks to `_rag_delta_chunks` (rendered
        next round via rag_append_delta)."""
        try:
            embed_q = getattr(kb_store, "embed_query", None)
            if embed_q is None or (time.monotonic() - self._drift_failed_at < 60):
                return
            query, _dv_vals = self.refer_query(state_name, dvs, user_msg)
            base = self._frozen_query_vec
            t0 = time.perf_counter()
            if base is None and self._frozen_query_text:
                base = await embed_q(self._frozen_query_text)
                self._frozen_query_vec = base
            round_vec = await embed_q(query)
            # iter48 M3 (R3): base-None is NO LONGER a dead end — an empty
            # freeze has NO valid baseline, so the round FORCES a re-retrieve
            # and adopts the round vector as the new baseline (the rescue
            # happens in the existing drift path below).
            force_drift = base is None
            cos = 0.0 if force_drift else _cosine(round_vec, base)
            rag_ms = 0.0
            new_chunks: list = []
            if cos >= self.settings.rag_drift_threshold and not force_drift:
                chunks = self._frozen_chunks
            else:
                try:
                    chunks = await kb_store.retrieve(query, scope, vec=round_vec)
                except Exception:
                    return
                self._frozen_query_vec = round_vec
                self._frozen_query_text = query
                if self.settings.rag_append_delta:
                    seen = {(c.get("kb"), c.get("content")) for c in self._frozen_chunks} \
                        | {(c.get("kb"), c.get("content")) for c in self._rag_delta_chunks}
                    self._rag_delta_chunks += [
                        c for c in chunks
                        if (c.get("kb"), c.get("content")) not in seen]
                    chunks = self._frozen_chunks
                else:
                    self._frozen_chunks = chunks
                self._kb_stats["rag_turns"] += 1
                self._kb_stats["rag_chars"] += sum(len(c["content"]) for c in chunks)
            rag_ms = round((time.perf_counter() - t0) * 1000, 1)
            if self.tracer:
                self.tracer.span("rag:drift", output={
                    "state": state_name, "cosine": round(cos, 4), "ms": rag_ms,
                    "async": True, "chunks": len(chunks),
                    "force_drift": force_drift,
                    "delta_chunks": len(self._rag_delta_chunks)
                        if self.settings.rag_append_delta else 0,
                    "query": query[:200]})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._drift_failed_at = time.monotonic()
            log.warning("async drift check failed for %s (%s)", state_name, exc)

    # iter25 retrieval gate: mechanical states pull nothing (iter24: 431 wasted
    # sales-psychology pulls in Closing etc.); Closing allow-lists call-closing;
    # sales states keep general+state scope as today.
    _RETRIEVAL_ALLOW: dict[str, list[str]] = {"Closing": ["call-closing"]}
    # iter59 T4 (owner): KB EVERYWHERE — every state retrieves all call long
    # (Retell-parity: retrieval on EVERY response from the global union).
    # The four mechanical states were blind since iter25; freed states
    # retrieve via lane B over the 9-KB general union (kb_slugs_for fallback).
    # The attribute STAYS (referenced at the round gate / tests) — empty set
    # keeps the gate code alive and the revert is a one-line restore.
    _RETRIEVAL_OFF: set[str] = set()

    # iter49 T4: REAL per-state KB attachment. The iter48 union (general ∪
    # state markers) gave Intake/Discovery/Closer/Offer the SAME 9-10-KB scope
    # — per-state attachment was effectively fake (FIND-9). This map is the
    # lane-B scope and the parity target ("same KBs visible" per state).
    # Pinned from the iter49 plan T4; validated by the pre-flight replay
    # (research/surgeon/iter49p-sharp-rag/02_replay_report.md pack #9).
    _STATE_KBS: dict[str, list[str]] = {
        "Intake": ["pain-points", "call-context", "sales-language",
                   "voice-ai-capabilities", "are-you-ai", "sales-psychology",
                   "industry", "hipaa", "call-closing"],
        "Discovery": ["pain-points", "call-context", "sales-language",
                      "voice-ai-capabilities", "are-you-ai", "sales-psychology",
                      "industry", "hipaa", "call-closing", "discovery-bridge"],
        "Closer": ["sales-psychology", "industry", "sales-language",
                   "call-context", "pain-points"],
        "Offer": ["sales-psychology", "sales-language", "industry",
                  "voice-ai-capabilities"],
        "Closing": ["call-closing", "sales-language"],
    }

    def kb_slugs_for(self, state_name: str) -> list[str]:
        """iter49 T4: per-state KB attachment (_STATE_KBS). _RETRIEVAL_OFF
        keeps the mechanical-states gate (zero pulls); unmapped states keep
        the iter48 general∪state union (revert-safe fallback)."""
        if state_name in self._RETRIEVAL_OFF:
            return []
        if state_name in self._STATE_KBS:
            return list(self._STATE_KBS[state_name])
        if state_name in self._RETRIEVAL_ALLOW:
            return list(self._RETRIEVAL_ALLOW[state_name])
        general = Path(self.settings.prompts_dir) / "general_prompt.md"
        general_text = general.read_text(encoding="utf-8") if general.exists() \
            else self.llm_json.get("general_prompt", "")
        slugs = ragmod.kb_slugs_in(general_text)
        state_text = self.prompts.get(state_name, "")
        slugs += [s for s in ragmod.kb_slugs_in(state_text) if s not in slugs]
        for t in ragmod.parse_refer_tags(state_text):     # iter25-b: tag slugs join the scope
            if t["kb"] not in slugs:
                slugs.append(t["kb"])
        return slugs

    def refer_query(self, state_name: str, dvs: dict, user_msg: str) -> tuple[str, list[str]]:
        """iter48: truth-anchored retrieval query.

        iter25-b built "[state: X] whats: dv-values user msg" — the state tag
        and the empty-dv join polluted the embedding (the are-you-ai beat the
        pitch for "what do you guys do") and the empty-user branch returned a
        content-free placeholder, freezing every prefetch 0-chunk (iter40 C3).
        iter48 R1: the query is the refer whats (KB anchors, KEPT) + the
        non-empty dv values + the caller text — NO state tag, NO label join,
        NO empty-dv join. Empty user_msg = state-anchor mode (prefetch):
        whats + dv values only. Returns (query, values_used).
        """
        tags = ragmod.parse_refer_tags(self.prompts.get(state_name, ""))
        parts: list[str] = []
        values: list[str] = []
        if tags:
            parts.append("; ".join(t["what"] for t in tags))
            for t in tags:
                for dv in t["dvs"]:
                    v = str(dvs.get(dv) or "").strip()
                    if v:
                        values.append(v)
            if values:
                parts.append(" / ".join(values))
        if user_msg:
            parts.append(user_msg)
        query = " ".join(p.strip() for p in parts if p.strip())
        if not query:
            # iter48 M2 contract: no anchor at all (no tags AND no user msg)
            # → empty query; _stage_rag guards on this (caller falls back to
            # the sync fetch, never worse).
            return "", []
        return query, values

    # ------------------------------------------------------------------ #
    # iter49 T4: live multi-lane retrieval (rag_live_retrieve=True).
    def build_lanes(self, state_name: str, dvs: dict,
                    user_msg: str, pinned_tag: str | None = None
                    ) -> tuple[list[dict], list[set]]:
        """The lane plan for this round. Lane A (engine intent) = one lane per
        refer-tag: query = the tag's `what` + its non-empty dv values (leak
        values $-formatted with units), scope = [tag.kb] ONLY — the tag's KB
        can no longer be crowded out. Closing has zero tags → the hand-written
        anchor instead (booked calls recap). Lane B (user intent) = the
        caller utterance (>= rag_min_query_chars) + the industry anchor when
        extracted (right-vertical 0.589->0.788, beats Retell's 0.637), scoped
        the state's _STATE_KBS. <=3 lanes per side. Returns (lanes,
        owned_kbs_per_lane) — owned KBs are quota-exempt in the merge.

        iter52: `pinned_tag` (default None = byte-exact iter49) removes
        `industry` from the scopes — the pinned KB leaves the vector lanes
        and re-enters via render_pinned_section. The deep-dive scope trigger
        (normalized industry dv contains a trigger keyword, e.g. 'plumb')
        appends the extra KB (e.g. `plumbing`) to the lane-B scope — INDEPENDENT
        of the pin, gated by rag_pin_industry."""
        lanes: list[dict] = []
        owned: list[set] = []
        scope = self.kb_slugs_for(state_name)
        if not scope:
            return lanes, owned
        if pinned_tag:
            scope = [s for s in scope if s != "industry"]
        if getattr(self.settings, "rag_pin_industry", False):
            # iter64: pre-pin gate — the shared `industry` KB leaves the
            # scopes until the vertical is KNOWN (pin resolved or the
            # industry dv extracted); generic pre-pin turns cross-matched
            # every vertical's pain/ROI prose (iter63 battery: Susan t3/t4,
            # Marcus t3/t5, Maria t4, Danny t6). The pin block re-introduces
            # the vertical the moment it resolves.
            if not pinned_tag \
                    and not str((dvs or {}).get("industry") or "").strip():
                scope = [s for s in scope if s != "industry"]
            industry_norm = str((dvs or {}).get("industry") or "").strip().lower()
            if industry_norm:
                for kw, extra_kbs in _INDUSTRY_KB_TRIGGERS.items():
                    if kw in industry_norm:
                        for kb in extra_kbs:
                            if kb not in scope:
                                scope.append(kb)
        # ---- lane A: engine intent (refer-tags; Closing anchor) ----------
        if not self.settings.rag_multiquery:
            # middle mode: ONE combined query (iter48's refer_query shape),
            # scoped to the state — live-per-turn without the lane split.
            q, _vals = self.refer_query(state_name, dvs, user_msg)
            if q.strip():
                lanes.append({"query": q, "scope": scope})
                owned.append(set())
        elif state_name == "Closing":
            booked = bool(dvs.get("booking_verified")
                          or dvs.get("booking_confirmed"))
            lanes.append({
                "query": _CLOSING_ANCHOR_BOOKED if booked else _CLOSING_ANCHOR,
                "scope": scope})
            owned.append({"call-closing", "sales-language"})
        else:
            tags = ragmod.parse_refer_tags(self.prompts.get(state_name, ""))[:_MAX_LANES_A]
            for t in tags:
                vals = [_fmt_leak_value(dv, str(dvs.get(dv) or "").strip())
                        for dv in t["dvs"]
                        if str(dvs.get(dv) or "").strip()]
                q = t["what"] if not vals else f"{t['what']}: {', '.join(vals)}"
                lanes.append({"query": q, "scope": [t["kb"]]})
                owned.append({t["kb"]})
        # ---- lane B: user intent (the caller utterance) ------------------
        msg = (user_msg or "").strip()
        if len(msg) >= int(getattr(self.settings, "rag_min_query_chars", 0)):
            industry = str(dvs.get("industry") or "").strip()
            q = f"{msg} (their industry: {industry})" if industry else msg
            lanes.append({"query": q, "scope": scope})
            owned.append(set())       # lane B owns nothing: quota applies
        return lanes[:_MAX_LANES_A + _MAX_LANES_B], owned

    async def _retrieve_raw(self, state_name: str, dvs: dict, user_msg: str,
                            kb_store, lane: str = "full"
                            ) -> tuple[list, list, float, list[str], str, dict]:
        """iter59 T3: the retrieval pieces WITHOUT the merge/render — so the
        hybrid consume can merge lane A's landed set and lane B's awaited set
        through the SAME merge_lane_chunks (top_k=3, 1600 chars, per-KB quota
        — lane A candidates keep their owned-KB exemption).

        lane:
          "full"  — build_lanes as today (all lanes, utterance included)
          "laneA" — engine intent ONLY: build_lanes with an EMPTY utterance
                    (refer-tag lanes + dv values; utterance-independent).
                    Hybrid lane A fires at the first flux Update.
          "laneB" — the caller utterance as a SINGLE query (one embed),
                    scoped the state's KB slugs; no owned KBs (quota applies).
                    Hybrid lane B fires at EOT/EagerEOT.

        Returns (lane_results, owned, ms, lane_queries, pinned_tag, pin).
        The pinned industry vertical is ensured FIRST (unchanged iter52)."""
        t0 = time.perf_counter()
        await self._ensure_pinned_industry(dvs, kb_store)
        pin = self._pinned_industry or {}
        pinned_tag = str(pin.get("tag") or "")
        if pinned_tag and not pin.get("chunks"):
            pinned_tag = ""     # resolve hit, fetch empty -> degrade to the vector lane
        if lane == "laneB":
            msg = (user_msg or "").strip()
            lanes: list[dict] = []
            owned: list[set] = []
            if len(msg) >= int(getattr(self.settings, "rag_min_query_chars", 0)):
                industry = str(dvs.get("industry") or "").strip()
                q = f"{msg} (their industry: {industry})" if industry else msg
                scope = [s for s in self.kb_slugs_for(state_name)
                         if not (pinned_tag and s == "industry")]
                # iter64: pre-pin gate — same rule as build_lanes; pinned
                # and dv-known rounds are byte-identical to iter63.
                if getattr(self.settings, "rag_pin_industry", False) \
                        and not pinned_tag and not industry:
                    scope = [s for s in scope if s != "industry"]
                lanes.append({"query": q, "scope": scope})
                owned.append(set())       # lane B owns nothing: quota applies
        elif lane == "laneA":
            lanes, owned = self.build_lanes(state_name, dvs, "",
                                            pinned_tag=pinned_tag or None)
            # build_lanes with an empty utterance keeps ONLY lane A (the
            # utterance branch is gated by rag_min_query_chars) — exactly
            # the engine-intent half of hybrid.
        else:
            lanes, owned = self.build_lanes(state_name, dvs, user_msg,
                                            pinned_tag=pinned_tag or None)
        lane_results = await kb_store.retrieve_lanes(lanes) if lanes else []
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return (lane_results, owned, ms, [l["query"] for l in lanes],
                pinned_tag, pin)

    def _render_from_raw(self, lane_results: list, owned: list, merged: list,
                         ms: float, lane_qs: list[str], pinned_tag: str,
                         pin: dict) -> tuple[str, list[str]]:
        """iter59 T3: merge/render shared by every consume path (single
        merge_lane_chunks + render_knowledge_section + optional pin block).
        iter60 AUD-10: _kb_stats increments live in the CALLERS (_live_retrieve
        and _consume_hybrid) — a shared counter here double-counted hybrid
        (bg-task land + consume each called this)."""
        knowledge = ragmod.render_knowledge_section(merged)
        if pinned_tag:
            pin_delta = ragmod.render_pinned_section(
                pin.get("chunks") or [], pinned_tag,
                cap=int(getattr(self.settings, "rag_pin_char_budget", 1800)))
            delta = f"{pin_delta}\n\n{knowledge}" if pin_delta else knowledge
            kbs = sorted({str(c.get("kb", "")) for c in merged} | {"industry"})
        else:
            delta = knowledge
            kbs = sorted({str(c.get("kb", "")) for c in merged})
        return delta, kbs

    async def _live_retrieve(self, state_name: str, dvs: dict, user_msg: str,
                             kb_store, lane: str = "full"
                             ) -> tuple[str, list[str], float, list, list[str]]:
        """Fresh multi-lane retrieval for THIS round: build_lanes → ONE
        batched embed (all lane texts in ONE call, off-hot-path embedder) +
        CONCURRENT pgvector → merge with the per-KB quota + chunk-id dedupe.
        Returns (delta_str, kbs, ms, merged_chunks, lane_queries).

        iter52: the pinned industry vertical is ensured FIRST (once per dv
        value, inside this bg task — off the hot path); when pinned, industry
        leaves the lane scopes and the pin renders as its OWN block (own
        budget, whole-chunks) PREPENDED to the per-turn KNOWLEDGE."""
        t0 = time.perf_counter()
        lane_results, owned, _ms, lane_qs, pinned_tag, pin = \
            await self._retrieve_raw(state_name, dvs, user_msg, kb_store, lane=lane)
        merged = ragmod.merge_lane_chunks(
            lane_results, owned,
            top_k=int(self.settings.rag_top_k),
            char_budget=int(self.settings.rag_char_budget))
        ms = round((time.perf_counter() - t0) * 1000, 1)
        # iter60 AUD-10: count here (bg task land / round-sync inline) —
        # preserves iter58 counting semantics for eot/speech-window/round-sync.
        self._kb_stats["rag_turns"] += 1
        self._kb_stats["rag_ms_total"] += ms
        self._kb_stats["rag_chars"] += sum(len(str(c.get("content", ""))) for c in merged)
        delta, kbs = self._render_from_raw(lane_results, owned, merged, ms,
                                           lane_qs, pinned_tag, pin)
        return (delta, kbs, ms, merged, lane_qs)

    async def _resolve_industry_tag(self, dv: str, kb_store) -> str | None:
        """iter52 T3: the 5-step resolver ladder — normalize → stoplist →
        alias map (single tag; ≥2 distinct hits = ambiguous) → embed fallback
        (ONE store.retrieve over ['industry'], top-1 >= rag_filter_score,
        header skipped) → None (caller keeps today's vector lane). Never
        guesses, never blocks."""
        norm = " ".join(str(dv or "").split()).lower()
        if norm in _PIN_STOPLIST:
            return None
        hits = {tag for alias, tag in _PIN_ALIASES
                if re.search(rf"\b{re.escape(alias)}", norm)}
        if len(hits) == 1:
            return next(iter(hits))
        # ambiguous (>=2 distinct tags) or no alias hit -> embed fallback
        filter_score = float(getattr(self.settings, "rag_filter_score", 0.24))
        try:
            chunks = await kb_store.retrieve(norm or dv, ["industry"])
        except Exception:
            return None
        for c in chunks:                     # best-first (retrieve order)
            score = c.get("score")
            if score is None or float(score) < filter_score:
                break
            vertical = await kb_store.vertical_of(c.get("id"))
            if vertical and vertical != "header":
                return vertical
        return None

    async def _ensure_pinned_industry(self, dvs: dict, kb_store) -> None:
        """iter52 T5: resolve + fetch the pinned vertical ONCE per call (per
        dv value), lazily, inside the bg _live_retrieve task. The result is
        cached on the dv VALUE — re-resolve only when it changes; a resolved
        miss (no vertical / fetch failure) is cached too so a pinless call
        never re-embeds every round. Kill switch off -> no-op (byte-exact
        iter49). Never raises."""
        if not getattr(self.settings, "rag_pin_industry", False):
            return
        dv = str((dvs or {}).get("industry") or "").strip()
        if not dv:
            return
        pin = self._pinned_industry
        if pin and pin.get("dv") == dv:
            return                    # cached (hit or resolved miss)
        try:
            t0 = time.perf_counter()
            tag = await self._resolve_industry_tag(dv, kb_store)
            chunks = await kb_store.pinned_by_tag(tag) if tag else []
            self._pinned_industry = {"dv": dv, "tag": tag or "",
                                     "chunks": chunks}
            if self.tracer:
                self.tracer.span("rag:pin", output={
                    "dv": dv[:80], "tag": tag or "",
                    "chunks": len(chunks),
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "resolved": bool(tag and chunks)})
        except Exception as exc:
            log.warning("pin industry ensure failed: %s", exc)
            self._pinned_industry = {"dv": dv, "tag": "", "chunks": []}

    # ------------------------------------------------------------------ #
    # iter49 T3: async off the hot path. The eager EOT fires spawn_live_
    # retrieve the moment the lane queries exist (the batched arctic embed,
    # ~90-190ms multi-lane, runs inside the caller's remaining speech + the
    # speculative LLM start); the round's live branch bounded-awaits the
    # in-flight task (rag_live_await_ms) — ready: THIS turn's fresh set
    # (awaited = pgvector + merge only); not ready: the PREVIOUS turn's
    # fresh set for this state (first turn of a call: empty) — never a
    # stall; the shielded task lands for the next round.
    def spawn_live_retrieve(self, state_name: str, dvs: dict,
                            user_msg: str, lane: str = "full") -> None:
        """Fire-and-forget live retrieval for (state, dvs, user_msg). NEVER
        raises. An in-flight task for the same state+lane is superseded
        (cancelled) by the newer query, like _spawn_drift_check. No-op when
        rag_live_retrieve is off (the iter48 path never consumes these).

        iter59 T3: tasks are keyed "<state>|<lane>" and every fire bumps a
        per-state SEQUENCE number (_live_seq). The consumer matches seq (not
        msg text — fixes the same-text-rerun edge): a task from an older seq
        can never be consumed as fresh."""
        try:
            if not getattr(self.settings, "rag_live_retrieve", False):
                return
            asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            key = f"{state_name}|{lane}"
            pending = self._live_tasks.get(key)
            if pending is not None and not pending.done():
                pending.cancel()      # superseded by the newer query
            seq = self._live_seq.get(state_name, 0) + 1
            self._live_seq[state_name] = seq
            task = asyncio.get_running_loop().create_task(
                self._live_task(state_name, dict(dvs or {}), user_msg or "",
                                lane=lane, seq=seq))
            self._live_tasks[key] = task
            self._live_msgs[key] = user_msg or ""
            self._live_task_seq[key] = seq
        except Exception as exc:
            log.warning("spawn_live_retrieve(%s, %s) failed: %s",
                        state_name, lane, exc)

    async def _live_task(self, state_name: str, dvs: dict,
                         user_msg: str, lane: str = "full",
                         seq: int = 0) -> dict | None:
        """The bg body: resolve the store, run the retrieval, stash the
        result into _live_prev[state_name] and RETURN it. Exceptions are
        logged, never raised — the round degrades to the previous set.

        iter59 T3: LANE tasks (laneA/laneB) additionally stash the RAW
        piece sets (lane_results / owned / pinned pieces) so the hybrid
        consume can re-merge both lanes through ONE merge_lane_chunks."""
        try:
            store = await self._resolve_kb_store()
            if store is None:
                return None
            if lane in ("laneA", "laneB"):
                lane_results, owned, ms, lane_qs, pinned_tag, pin = \
                    await self._retrieve_raw(state_name, dvs, user_msg, store,
                                             lane=lane)
                merged = ragmod.merge_lane_chunks(
                    lane_results, owned,
                    top_k=int(self.settings.rag_top_k),
                    char_budget=int(self.settings.rag_char_budget))
                delta, kbs = self._render_from_raw(
                    lane_results, owned, merged, ms, lane_qs,
                    pinned_tag, pin)
                result = {"delta": delta, "kbs": kbs, "chunks": merged,
                          "lane_qs": lane_qs, "ms": ms, "seq": seq,
                          "lane": lane, "lane_results": lane_results,
                          "owned": owned, "pinned_tag": pinned_tag, "pin": pin}
            else:
                delta, kbs, ms, merged, lane_qs = await self._live_retrieve(
                    state_name, dvs, user_msg, store, lane=lane)
                result = {"delta": delta, "kbs": kbs, "chunks": merged,
                          "lane_qs": lane_qs, "ms": ms, "seq": seq,
                          "lane": lane}
            self._live_prev[state_name] = result
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("live retrieve task for %s failed: %s",
                        state_name, exc)
            return None

    async def _consume_live_retrieve(self, state_name: str, dvs: dict,
                                     user_msg: str, kb_store):
        """The round-path consumer, per rag_fire_mode (iter59 T3):

        "eot" (default) — iter58 surface EXACTLY: task keyed "<state>|full",
            msg-text match, bounded await, degrade to _live_prev.
        "speech-window" — the task was fired at the FIRST flux Update (the
            EOT/EagerEOT fire was SKIPPED so the landed task survives).
            Freshness matches by SEQUENCE (not msg text — the fire ran on a
            PARTIAL utterance): a fire not yet consumed = fresh for this
            round; multi-round same-turn reuse allowed when the msg matches.
            Timeout fallback degrades to _live_consumed_res (the last
            CONSUMED fresh set — never _live_prev's unconsumed overwrite).
        "round-sync" — no task at all: ONE inline retrieval on the round
            path (span label live-sync). Degraded 0% by construction.
        "hybrid" — lane A (fired at first Update, refer-tags + dvs) + lane B
            (fired at EOT/EagerEOT, single utterance query). Lane A's landed
            pieces and lane B's awaited pieces merge through the SAME
            merge_lane_chunks (lane A keeps its owned-KB exemption).
            Degraded = NEITHER lane landed fresh.

        NEVER raises into the node; only a true node cancellation propagates.
        Returns (delta, kbs, ms, chunks, lane_qs, degraded, await_ms)."""
        t_wait = time.perf_counter()
        msg = user_msg or ""
        mode = getattr(self.settings, "rag_fire_mode", "eot")

        if mode == "round-sync":
            self._live_mode_label = "live-sync"
            store = kb_store
            if store is None:
                store = await self._resolve_kb_store()
            if store is None:
                return ("", [], 0.0, [], [], True,
                        round((time.perf_counter() - t_wait) * 1000, 1))
            delta, kbs, ms, merged, lane_qs = await self._live_retrieve(
                state_name, dvs, msg, store, lane="full")
            await_ms = round((time.perf_counter() - t_wait) * 1000, 1)
            return (delta, kbs, ms, merged, lane_qs, False, await_ms)

        if mode == "hybrid":
            return await self._consume_hybrid(state_name, dvs, msg, kb_store,
                                              t_wait)

        # ---- "eot" (iter58 exact) and "speech-window" share the task path
        key = f"{state_name}|full"
        task = self._live_tasks.get(key)
        if mode == "eot":
            stale = task is None or task.cancelled() \
                or self._live_msgs.get(key) != msg
        else:
            # speech-window: fresh = a fire seq not yet consumed for this
            # state (the EOT rerun of an identical text spawns a NEW seq,
            # so cross-turn staleness is impossible); an already-consumed
            # task may still be REUSED when its msg matches (multi-round).
            tseq = self._live_task_seq.get(key, -1)
            cseq = self._live_consumed.get(state_name, -1)
            fresh = tseq > cseq
            stale = task is None or task.cancelled() or (
                not fresh and self._live_msgs.get(key) != msg)
        if stale:
            self.spawn_live_retrieve(state_name, dvs, msg, lane="full")
            task = self._live_tasks.get(key)
        res = None
        if task is not None and not task.cancelled() and task.done():
            res = task.result() or None          # landed (fresh or failure)
        elif task is not None and not task.cancelled():
            cap_ms = int(getattr(self.settings, "rag_live_await_ms", 60))
            if cap_ms > 0:
                try:
                    await asyncio.wait_for(asyncio.shield(task), cap_ms / 1000)
                except asyncio.CancelledError:
                    if not task.cancelled():
                        raise    # the NODE itself was cancelled — propagate
                    # the inner task was cancelled (superseded) — degrade
                except Exception:
                    pass        # timeout or task failure — degrade below
            if task.done() and not task.cancelled():
                res = task.result() or None
        await_ms = round((time.perf_counter() - t_wait) * 1000, 1)
        if res:
            self._live_consumed[state_name] = self._live_task_seq.get(key, 0)
            self._live_consumed_res[state_name] = res   # last consumed fresh set (C4)
            # iter62: same empty-set rule — 0 chunks is never "healthy".
            return (res["delta"], res["kbs"], res["ms"], res["chunks"],
                    res["lane_qs"], not res["chunks"], await_ms)
        # degrade: the last CONSUMED fresh set for this state (C4) — a fire
        # that never landed must NOT overwrite the degrade source
        # (_live_prev is written by the bg task at completion time, consumed
        # or not). First turn of a call: empty.
        fallback = self._live_consumed_res.get(state_name) \
            if mode != "eot" else None
        if fallback is None:
            fallback = self._live_prev.get(state_name) or {}
        return (fallback.get("delta", ""), fallback.get("kbs", []),
                fallback.get("ms", 0.0), fallback.get("chunks", []),
                fallback.get("lane_qs", []), True, await_ms)

    async def _consume_hybrid(self, state_name: str, dvs: dict, msg: str,
                              kb_store, t_wait: float):
        """iter59 T3: hybrid consume — lane A (landed during speech: refer-tag
        queries + dv values, utterance-independent) + lane B (fired at
        EOT/EagerEOT: the caller utterance as a SINGLE query, awaited within
        the cap). Both lanes' RAW piece sets merge through the SAME
        merge_lane_chunks (lane A candidates keep their owned-KB exemption;
        the pin block re-renders once). Degraded = NEITHER lane landed fresh
        this round; fallback = the last consumed fresh set (C4), else the
        previous merged set."""
        self._live_mode_label = "hybrid"
        key_a, key_b = f"{state_name}|laneA", f"{state_name}|laneB"
        # iter60 AUD-3: rounds 2+ of the SAME turn reuse the round-1 merged
        # set (the seq floor rightly rejects stale laneA, but it also
        # rejected THIS turn's laneA — knowledge silently shrank to
        # laneB-only on every tool round). Stash validity = same msg AND
        # the laneB task unchanged (a new fire bumps _live_task_seq).
        stash = self._live_merged_turn.get(state_name)
        if stash and stash.get("msg") == msg \
                and self._live_task_seq.get(key_b, -1) == stash.get("b_seq"):
            r = stash["res"]
            # iter62: same empty-set honesty on same-turn reuse rounds.
            return (r["delta"], r["kbs"], r["ms"], r["chunks"],
                    r["lane_qs"], not r["chunks"], 0.0)
        task_a = self._live_tasks.get(key_a)
        task_b = self._live_tasks.get(key_b)
        # lane A: use only if it LANDED (never await — it had the whole
        # speech window) and its fire seq was not already consumed.
        raw_a = None
        if task_a is not None and not task_a.cancelled() and task_a.done():
            ra = task_a.result() or None
            if ra and ra.get("seq", 0) > self._live_consumed.get(state_name, -1):
                raw_a = ra
        # lane B: bounded await on the in-flight single-query embed.
        # iter62: NO respawn — the fire lives in the session layer
        # (Update/EagerEOT/EOT). Missing, cancelled, or fired for a
        # DIFFERENT utterance (iter60 AUD-4 mismatch) = CLEAN FAIL this
        # round; the next fire supersedes (owner: no late-restart injection).
        res_b = None
        if task_b is None or task_b.cancelled() \
                or self._live_msgs.get(key_b) != msg:
            task_b = None
        if task_b is not None and not task_b.cancelled():
            if not task_b.done():
                cap_ms = int(getattr(self.settings, "rag_live_await_ms", 60))
                if cap_ms > 0:
                    try:
                        await asyncio.wait_for(asyncio.shield(task_b),
                                               cap_ms / 1000)
                    except asyncio.CancelledError:
                        if not task_b.cancelled():
                            raise   # the NODE itself was cancelled — propagate
                    except Exception:
                        pass        # timeout or failure — degrade below
            if task_b.done() and not task_b.cancelled():
                res_b = task_b.result() or None
        await_ms = round((time.perf_counter() - t_wait) * 1000, 1)
        # merge the RAW piece sets that landed
        lane_results: list = []
        owned: list = []
        lane_qs: list[str] = []
        ms = 0.0
        pinned_tag, pin = "", {}
        landed = False
        for r in (raw_a, res_b):
            if not r:
                continue
            lr = r.get("lane_results")
            if lr is None:            # a merged (full-shape) result — skip
                continue
            lane_results.extend(lr)
            owned.extend(r.get("owned") or [])
            lane_qs.extend(r.get("lane_qs") or [])
            ms = max(ms, float(r.get("ms") or 0.0))
            if not pinned_tag and r.get("pinned_tag"):
                pinned_tag = r["pinned_tag"]
                pin = r.get("pin") or {}
            landed = True
        if not landed:
            # iter62 — owner decision: a late/failed retrieval is a CLEAN
            # FAIL; stale-prev injection feeds the model chunks that make
            # no sense at the wrong moment. Next fire supersedes.
            return ("", [], 0.0, [], [], True, await_ms)
        merged = ragmod.merge_lane_chunks(
            lane_results, owned,
            top_k=int(self.settings.rag_top_k),
            char_budget=int(self.settings.rag_char_budget))
        delta, kbs = self._render_from_raw(lane_results, owned, merged, ms,
                                           lane_qs, pinned_tag, pin)
        # iter60 AUD-10: hybrid counts ONCE per consumed round (the bg land
        # path no longer increments inside _render_from_raw).
        self._kb_stats["rag_turns"] += 1
        self._kb_stats["rag_ms_total"] += ms
        self._kb_stats["rag_chars"] += sum(len(str(c.get("content", ""))) for c in merged)
        # record consumption: lane A's seq becomes the consumed floor
        if raw_a:
            self._live_consumed[state_name] = max(
                self._live_consumed.get(state_name, -1), raw_a.get("seq", 0))
        fresh = {"delta": delta, "kbs": kbs, "chunks": merged,
                 "lane_qs": lane_qs, "ms": ms}
        self._live_consumed_res[state_name] = fresh   # C4 fallback source
        # iter60 AUD-3: stash the merged set for same-turn rounds (rounds 2+
        # short-circuit above; a NEW turn bumps b_seq or changes msg → stale).
        self._live_merged_turn[state_name] = {
            "msg": msg, "b_seq": self._live_task_seq.get(key_b, -1),
            "res": fresh}
        # iter62: chunks=0 is an HONEST degraded round (was hard-coded False;
        # 38/67 spans hid behind that on the iter61 call).
        return (delta, kbs, ms, merged, lane_qs, not merged, await_ms)

    # ------------------------------------------------------------------ #
    def _load_prompts(self) -> dict[str, str]:
        """V2: prompts from diallux/prompts (source of truth, editable).
        Falls back to llm.json's embedded state_prompt when a file is absent."""
        prompts_dir = Path(self.settings.prompts_dir)
        out: dict[str, str] = {}
        for name, cfg in self.states.items():
            p = prompts_dir / f"{name}.md"
            out[name] = p.read_text(encoding="utf-8") if p.exists() else cfg.get("state_prompt", "")
        return out

    # ------------------------------------------------------------------ #
    def static_head(self, state_name: str, expand_kb: bool = True) -> str:
        """iter31 C3: byte-stable system head — general_prompt + state prompt
        with dvs NOT substituted ({{var}} left literal via dvs={}), cached per
        (state, expand_kb). Identical bytes across rounds AND calls, so OpenAI's
        automatic prompt cache holds the whole prefix. VOICE_OUTPUT_RULES fold
        in here (static bytes belong in the cached head, not behind the tail).
        iter56: under head_strip_vars the literal {{var}} tokens are rewritten
        to the stable anchor "CURRENT CALL STATE" AT BUILD TIME — still cached
        per (state, expand_kb), still byte-stable across rounds AND calls (no
        dvs involved). The anchor points at the delta block that ALWAYS carries
        the live values (state_in_delta) — kills the {{dv}} blindness (PT-53)."""
        key = (state_name, expand_kb)
        head = self._head_cache.get(key)
        if head is None:
            general = Path(self.settings.prompts_dir) / "general_prompt.md"
            general_text = general.read_text(encoding="utf-8") if general.exists() \
                else self.llm_json["general_prompt"]
            head = self.subst.subst(general_text + "\n\n" + self.prompts.get(state_name, ""),
                                    {}, kb=expand_kb)
            if self.settings.tts_voice_rules:
                head += VOICE_OUTPUT_RULES
            # iter56 (C2.1): head strip — dead {{var}} anchors become a stable
            # pointer to the live-values delta. Applied at head-BUILD time so
            # the result stays byte-stable (cached per key, no dvs involved).
            if getattr(self.settings, "head_strip_vars", False):
                head = _VAR_TOKEN_RE.sub("CURRENT CALL STATE", head)
            self._head_cache[key] = head
        return head

    def _lite_head(self) -> str:
        """iter43 first-turn-lite: turn-1 system = general_prompt +
        VOICE_OUTPUT_RULES only (no state prompt, no RAG) — same read path as
        static_head, no KB markers."""
        general = Path(self.settings.prompts_dir) / "general_prompt.md"
        general_text = general.read_text(encoding="utf-8") if general.exists() \
            else self.llm_json.get("general_prompt", "")
        # iter43 FIX: kb=False — do NOT inline the KBs the general prompt
        # references. kb=True ballooned the "light" turn-1 head to ~16.5k
        # tokens (every ##slug-kb## inline-expanded) — the opposite of light.
        head = self.subst.subst(general_text, {}, kb=False)
        if self.settings.tts_voice_rules:
            head += VOICE_OUTPUT_RULES
        # iter56: the strip is a head-BUILD pass — the lite head strips too
        # (same dead-{{var}} blindness on the turn-1 general prompt).
        if getattr(self.settings, "head_strip_vars", False):
            head = _VAR_TOKEN_RE.sub("CURRENT CALL STATE", head)
        return head

    def _state_block(self, dvs: dict) -> str:
        """iter31 C3: dynamic tail — every non-empty dv as `name: value`. The
        model reads live values HERE; the tail changing never invalidates the
        cached head. Schema field order (to_flat) keeps line order stable."""
        if not self.settings.state_block:
            return ""
        lines = [f"{k}: {v}" for k, v in dvs.items()
                 if v is not None and str(v).strip()]
        if not lines:
            return ""
        # iter32: recency co-emit directive — this block is the LAST message the
        # model reads; head-position rules failed 5 batteries at 60-69% silent.
        return ("\n\n# CURRENT CALL STATE (live values)\n"
                "SPEAK FIRST: every response starts with your spoken sentence, "
                "THEN fires its tools — one response, never tool calls without "
                "speech.\n" + "\n".join(lines))

    def _changed_dvs_lines(self, staged_dvs: dict, dvs: dict) -> str:
        """iter96: the TRUE per-round change — every dv whose value differs
        from the staged frame (new keys included; cleared/empty values are
        skipped — the dvs doctrine is monotonic capture, nothing deletes)."""
        return "\n".join(f"{k}: {v}" for k, v in dvs.items()
                         if v not in (None, "") and staged_dvs.get(k) != v)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _history_window(history: list[dict], n: int, step: int = 8) -> list[dict]:
        """iter43: hysteretic trim — the window START only advances every `step`
        entries past n, so the cached prefix stays byte-stable between jumps.
        len 17..24 (n=16, step=8): start pinned at 8 (window 9..16 entries,
        append-only); len 25: start jumps to 16. One re-prefill per jump
        instead of per turn. NEVER starting on an orphan tool message (an
        OpenAI 400: a tool response must follow its assistant tool_calls
        message). step=1 restores the old sliding-window behavior."""
        if n <= 0 or len(history) <= n:
            return list(history)
        start = step * ((len(history) - n + step - 1) // step)
        tail = history[start:] if start < len(history) else []
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]
        return tail

    # ------------------------------------------------------------------ #
    # iter46 D5 (H1 fix-1): per-state verbosity — states in
    # verbosity_states_medium run "medium" (late-flow ask/capture states
    # where low stalls contact_details completion); everything else rides
    # the "low" default (the 700 ms TTFT recipe). Env-able list.
    def _verbosity_for(self, state_name: str) -> str:
        if not state_name:
            return ""
        medium = {s.strip() for s in
                  str(getattr(self.settings, "verbosity_states_medium", "")).split(",")
                  if s.strip()}
        return "medium" if state_name in medium else ""

    # ------------------------------------------------------------------ #
    def _make_state_node(self, state_name: str) -> Callable[[GraphState], dict]:
        runtime = self

        async def state_node(state: GraphState) -> dict:
            writer = get_stream_writer()
            dvs = DynamicVariables.from_flat(state.get("dvs", {})).to_flat()
            history = state.get("history", [])

            # ---- V3 RAG: retrieval instead of full-KB inline expansion ----
            # iter31 C3b: STATIC head (dvs never substituted, cached bytes) is
            # messages[0]; the dynamic tail (STATE BLOCK + frozen RAG section)
            # is a TRAILING system message AFTER the history. The OpenAI prompt
            # cache holds tools + head + full append-only history; only the
            # small tail re-bills each round.
            # iter56: the tail is EMPTY under state_in_delta (default) — the
            # STATE BLOCK rides the post-history DELTA (fresh every round,
            # re-billed anyway) and the prefix [tools][head][history] is
            # truly append-only, so cache_read climbs with history. The
            # head's {{var}} tokens are stripped to the "CURRENT CALL STATE"
            # anchor (head_strip_vars) pointing at that delta.
            rag_ms = 0.0
            rag_kbs: list[str] = []
            # iter43 first-turn-lite: turn 1 = general_prompt + VOICE_OUTPUT_RULES
            # only — no state prompt, no tools, no RAG, no tail. Guard is == 1
            # EXACTLY (never <=1 — a default-0 state would go silent forever).
            # Default OFF; live enables via .env FIRST_TURN_LITE=true.
            lite = bool(runtime.settings.first_turn_lite) \
                and state.get("turn_index", 0) == 1
            # iter49 T6 (T4b): state-entry lite — the FIRST round in a state
            # whose name differs from the previous round's rendered state.
            # Covers the post-transition_to_X ack rounds AND the turn-2 entry
            # into the initial state (the turn-1 lite round renders no state
            # head, so it never counts as a visit). Turn 1 itself keeps the
            # EXISTING first_turn_lite path; per-state opt-out via
            # state_entry_lite_off. The ack can only SPEAK — extraction and
            # slots fire on the later heavy rounds.
            entry_lite = (
                not lite
                and bool(getattr(runtime.settings, "state_entry_lite", False))
                and state.get("turn_index", 0) > 1
                and state_name != runtime._prev_round_state
                and state_name not in runtime._entry_lite_off_states())
            kb_store = None if lite else await runtime._resolve_kb_store()
            if lite:
                system = runtime._lite_head()
                expand_kb = False
            elif entry_lite:
                # iter49 T6: [state head (kb=False)] — head bytes identical
                # to the heavy round's (same strip decision), never inline-
                # expanded. FULL tools (FIND-5 port — see the tools branch),
                # NO RAG lanes, NO warm await.
                system = runtime._entry_lite_head(state_name, kb_store)
                expand_kb = False
            else:
                system, expand_kb = runtime._head_for(state_name, kb_store)
                # iter44 T4: degraded (no store, inline guard OFF) — say so
                # ONCE per call instead of silently losing the KB lane.
                if kb_store is None and not expand_kb \
                        and not runtime._inline_guard_logged:
                    runtime._inline_guard_logged = True
                    log.warning("RAG fallback inline DISABLED "
                                "(fallback_inline_kb=False) — running without KB excerpts")
            # iter46 T4c: bounded await of an in-flight prewarm for this state
            # — a warm that finished in time rides cache_read at entry; on
            # timeout the warm keeps running (shield) for the next round.
            # iter48 Commit B (M11): EOT-boundary split — the FIRST round of
            # a turn (rounds_left == max_tool_rounds; turn 1 IS an EOT
            # boundary; sole writers: ingest sets max, deterministic decrements)
            # awaits with prewarm_entry_wait_eot_ms (500); mid-turn rounds keep
            # prewarm_entry_wait_ms (100). M12: the lite round awaits its own
            # `lite:<state>` warm with the EOT cap (turn-1 cache coverage).
            if runtime.settings.prewarm_entry_wait_eot_ms > 0 or \
                    runtime.settings.prewarm_entry_wait_ms > 0:
                eot = state.get("rounds_left", 0) == \
                    runtime.settings.max_tool_rounds
                if lite:
                    if eot and runtime.settings.prewarm_entry_wait_eot_ms > 0:
                        await runtime._await_warm(f"lite:{state_name}",
                                                  wait_ms=runtime.settings.prewarm_entry_wait_eot_ms)
                elif entry_lite:
                    # iter87 VERSION B hold law (supersedes the iter49 T6
                    # "NEVER awaits" rule): ordering, not timing — the ack's
                    # payload is NEVER sent before the destination warm has
                    # landed. We await a TASK, not a timer: success, error or
                    # HTTP timeout all complete it and release (shielded);
                    # 0-200 ms typical because the fire law scheduled the
                    # warm during think time / webhook idle. The cap is a
                    # fence, not the mechanism: a dead-provider day degrades
                    # to a cold round, never a frozen call. 0 = byte-exact
                    # iter85 (pass).
                    if runtime.settings.warm_await_dest_ms > 0:
                        await runtime._await_warm(
                            state_name,
                            wait_ms=runtime.settings.warm_await_dest_ms)
                        # iter87 (owner 2026-09-30): 10 ms settle after the
                        # warm lands — never race an incomplete prefix.
                        await asyncio.sleep(0.01)
                elif eot and runtime.settings.prewarm_entry_wait_eot_ms > 0:
                    await runtime._await_warm(
                        state_name, wait_ms=runtime.settings.prewarm_entry_wait_eot_ms)
                elif runtime.settings.prewarm_entry_wait_ms > 0:
                    await runtime._await_warm(state_name)
            # iter40 C3: the tail bytes were frozen at turn start (ingest) —
            # MODEL tool writes between rounds no longer mutate the prefix
            # (their results reach the model via the history tool messages).
            # EXCEPTION: a round following an engine-fired deterministic pass
            # (iter33 chain-abort surface, leak math) MUST re-render — the
            # engine's news has no other channel. Correctness > cache here.
            # iter56 (C2.2): under state_in_delta the tail is EMPTY in live
            # mode — the state-block rides the POST-HISTORY delta instead
            # (state_delta_block, joined with the RAG delta at final
            # assembly). The prefix [tools][head][history] stays append-only
            # and cache_read climbs with history. False = iter55 layout
            # (frozen/re-rendered tail pre-history — the revert path).
            state_delta_block = ""
            if lite:
                tail = ""
            elif runtime.settings.state_in_delta:
                tail = ""
                # fresh EVERY round — the delta re-bills anyway; the model
                # must read live values at the position that always mutates
                state_delta_block = runtime._state_block(dvs)
            else:
                tail = (state.get("frozen_state_block")
                        if runtime.settings.frozen_state_block
                        and state.get("frozen_state_block") is not None
                        and not state.get("engine_fired")
                        else runtime._state_block(dvs))
            # iter49 T4: live mode is decided ONCE here, BEFORE the kb_store
            # gate — rounds without a store (hermetic tests, KB disabled)
            # still need `live`/`live_delta` bound (UnboundLocalError bug,
            # fixed in iter49c T1: the bindings used to live inside the
            # kb_store branch).
            live = bool(getattr(runtime.settings, "rag_live_retrieve", False))
            live_delta = ""
            # iter49 T6: the state-entry ack round runs NO retrieval lanes —
            # the heavy rounds retrieve fresh (the eager-EOT fire supersedes).
            if kb_store is not None and not entry_lite:
                refer_tags = ragmod.parse_refer_tags(runtime.prompts.get(state_name, ""))
                scope = runtime.kb_slugs_for(state_name)
                # iter59 T4: LIVE branch gate = scope only (KB-everywhere:
                # tag-less mechanical states retrieve via lane B — Retell
                # tier). Non-live path below keeps the iter25 gate
                # (tags or allow-list) exactly.
                do_retrieve = bool(scope) if live else \
                    bool(scope) and (bool(refer_tags)
                                     or state_name in runtime._RETRIEVAL_ALLOW)
                # iter49 T4: fresh multi-lane retrieval EVERY non-lite round.
                # The merged working set REPLACES the post-history delta each
                # turn; the pre-history prefix [tools][head][history] is
                # NEVER rewritten (cache = placement, not byte-compare;
                # iter56: the state-block rides the delta too — state_in_delta).
                # Freeze/drift/prefetch are INERT under this
                # flag — the elif below is the exact iter48 path (revert
                # switch: rag_live_retrieve=False).
                if live and do_retrieve:
                    # iter62 — the session-provided transcript is the single
                    # source of truth; history re-extraction diverged under
                    # eager/barge-in and triggered empty respawns (25/67
                    # blind rounds on call cc22e0a0185e).
                    user_msg = (state.get("user_text") or "").strip()
                    # iter49 T3: bounded-await the in-flight task (fired at
                    # eager EOT); degrade = previous turn's fresh set, never
                    # a stall. The awaited portion is pgvector + merge only
                    # when the embed already ran in the speech window.
                    live_delta, rag_kbs, rag_ms, merged, lane_qs, degraded, await_ms = \
                        await runtime._consume_live_retrieve(
                            state_name, dvs, user_msg, kb_store)
                    # iter93 T1 (PT-79): last consume provenance for the
                    # session's turn report (laneA/laneB/fallback + await).
                    runtime._last_consume = {
                        "state": state_name,
                        "mode": getattr(runtime, "_live_mode_label",
                                        "live-async"),
                        "degraded": degraded, "await_ms": await_ms,
                        "ms": rag_ms, "chunks": len(merged)}
                    if runtime.tracer:
                        runtime.tracer.span("rag", output={
                            "state": state_name, "ms": rag_ms, "kbs": rag_kbs,
                            "chunks": len(merged),
                            "query": "; ".join(lane_qs)[:200],
                            "mode": getattr(runtime, "_live_mode_label",
                                            "live-async"),
                            "lanes": len(lane_qs),
                            "pinned": (getattr(runtime, "_pinned_industry", None)
                                       or {}).get("tag") or "",
                            "degraded": degraded,
                            "zero_hit": (len(merged) == 0),
                            "await_ms": await_ms,
                            "prefetched": False, "frozen_reuse": False,
                            "drifted": False})
                if do_retrieve and not live:
                    # iter31 C3: chunks frozen per state-visit — one retrieval on
                    # state entry, held until the state changes. Same-state rounds
                    # reuse the frozen bytes (no embed, no tail mutation).
                    # iter21: skip the embedding round trip on filler turns
                    # ("hello?", "yes") — zero chunks came back anyway, but the
                    # 200-1900ms embed call sat on the hot path before the LLM.
                    user_msg = next((m["content"] for m in reversed(history)
                                     if m.get("role") == "user"), "")
                    query = ""          # frozen-reuse rounds never enter a branch
                    frozen_reuse = runtime._frozen_state == state_name
                    drifted = False
                    drift_embed_ms = 0.0
                    if len(user_msg.strip()) < getattr(runtime.settings,
                                                      "rag_min_query_chars", 0):
                        # iter21: skip the embedding round trip on filler turns
                        # ("hello?", "yes") — zero chunks came back anyway, but
                        # the 200-1900ms embed sat on the hot path before the LLM.
                        query = ""
                        chunks = []
                        rag_ms = 0.0
                        runtime._frozen_state = state_name
                        runtime._frozen_chunks = chunks
                        # iter48 M4: a filler freeze has NO baseline — reset
                        # any PREVIOUS state's stale query text/vec so the
                        # next real message force-drifts instead of comparing
                        # against the wrong state's vector.
                        runtime._frozen_query_text = ""
                        runtime._frozen_query_vec = None
                    elif not frozen_reuse:
                        # iter44 T6b: consume this state's PREFETCHED chunks when
                        # the transition prewarm staged them — ~0 ms retrieval on
                        # the state-entry critical path; missing/failed staging
                        # falls back to the sync fetch (today's behavior).
                        staged = runtime._rag_staging.pop(state_name, None)
                        if staged is not None:
                            query = staged["query"]
                            chunks = staged["chunks"]
                            rag_ms = 0.0
                            runtime._frozen_state = state_name
                            runtime._frozen_chunks = chunks
                            runtime._frozen_query_text = query
                            runtime._frozen_query_vec = staged.get("vec")
                        else:
                            query, _dv_vals = runtime.refer_query(state_name, dvs, user_msg)
                            cache_key = (state_name, query)
                            chunks = runtime._rag_cache.get(cache_key)
                            cached = chunks is not None
                            if not cached:
                                t_rag = time.perf_counter()
                                chunks = await kb_store.retrieve(query, scope)
                                if len(runtime._rag_cache) > 64:
                                    runtime._rag_cache.clear()
                                runtime._rag_cache[cache_key] = chunks
                                rag_ms = round((time.perf_counter() - t_rag) * 1000, 1)
                            else:
                                rag_ms = 0.2
                            runtime._frozen_state = state_name
                            runtime._frozen_chunks = chunks
                            runtime._frozen_query_text = query
                            runtime._frozen_query_vec = None   # drift baseline embeds lazily
                            if not cached:
                                runtime._kb_stats["rag_turns"] += 1
                                runtime._kb_stats["rag_ms_total"] += rag_ms
                                runtime._kb_stats["rag_chars"] += sum(len(c["content"]) for c in chunks)
                    elif runtime.settings.rag_drift_requery:
                        # iter44 T6c: drift-triggered re-retrieval — a NEW caller
                        # message in the SAME state may need DIFFERENT knowledge
                        # (the iter43 miss: "trucking" never re-anchored the
                        # pain_points query).
                        # iter46 T2 (audit P2): the embed round-trip (234-272 ms
                        # measured) leaves the hot path. The round uses the
                        # frozen chunks with ZERO awaits; a bg task embeds
                        # base+query, computes cosine, and on drift re-retrieves
                        # and appends NEW chunks via the rag_append_delta path —
                        # they render next round (correctness lag ≤ 1 round,
                        # strictly better than pre-T6c which never re-anchored).
                        # Flag OFF → the old synchronous requery, exactly.
                        chunks = runtime._frozen_chunks
                        if runtime.settings.rag_drift_async:
                            runtime._spawn_drift_check(state_name, dvs, user_msg,
                                                       scope, kb_store)
                        else:
                            embed_q = getattr(kb_store, "embed_query", None)
                            if embed_q is None or (
                                    time.monotonic() - runtime._drift_failed_at < 60):
                                chunks = runtime._frozen_chunks   # can't drift-check now
                            else:
                                try:
                                    query, _dv_vals = runtime.refer_query(state_name, dvs, user_msg)
                                    t_e = time.perf_counter()
                                    round_vec = await embed_q(query)
                                    drift_embed_ms = round((time.perf_counter() - t_e) * 1000, 1)
                                    base = runtime._frozen_query_vec
                                    if base is None and runtime._frozen_query_text:
                                        base = await embed_q(runtime._frozen_query_text)
                                        runtime._frozen_query_vec = base
                                except Exception:
                                    runtime._drift_failed_at = time.monotonic()
                                    query = ""
                                    chunks = runtime._frozen_chunks
                                else:
                                    # iter48 M5 (R6): sync-lane parity with
                                    # the async forced-drift semantics —
                                    # base-None ⇒ forced re-retrieve, never a
                                    # silent keep-frozen.
                                    cos = 0.0 if base is None else _cosine(round_vec, base)
                                    if cos >= runtime.settings.rag_drift_threshold \
                                            and base is not None:
                                        chunks = runtime._frozen_chunks
                                    else:
                                        drifted = True
                                        try:
                                            t_rag = time.perf_counter()
                                            chunks = await kb_store.retrieve(query, scope, vec=round_vec)
                                            rag_ms = round((time.perf_counter() - t_rag) * 1000, 1)
                                        except Exception:
                                            chunks = runtime._frozen_chunks
                                        else:
                                            runtime._frozen_query_vec = round_vec
                                            runtime._frozen_query_text = query
                                            if runtime.settings.rag_append_delta:
                                                # append-only: keep the entry tail
                                                # byte-stable; only the chunks NOT
                                                # already rendered become a delta
                                                # system block after the history.
                                                seen = {
                                                    (c.get("kb"), c.get("content"))
                                                    for c in runtime._frozen_chunks
                                                } | {
                                                    (c.get("kb"), c.get("content"))
                                                    for c in runtime._rag_delta_chunks
                                                }
                                                runtime._rag_delta_chunks += [
                                                    c for c in chunks
                                                    if (c.get("kb"), c.get("content")) not in seen
                                                ]
                                                chunks = runtime._frozen_chunks
                                            else:
                                                runtime._frozen_chunks = chunks
                                            runtime._kb_stats["rag_turns"] += 1
                                            runtime._kb_stats["rag_ms_total"] += rag_ms
                                            runtime._kb_stats["rag_chars"] += sum(
                                                len(c["content"]) for c in chunks)
                                            if runtime.tracer:
                                                runtime.tracer.span("rag:drift", output={
                                                    "state": state_name,
                                                    "cosine": round(cos, 4),
                                                    "embed_ms": drift_embed_ms, "ms": rag_ms,
                                                    "chunks": len(chunks), "query": query[:200]})
                    chunks = runtime._frozen_chunks
                    # iter44 append-only: the KNOWLEDGE tail is byte-stable for
                    # the whole state visit (frozen at entry); drift re-fetches
                    # land as a DELTA system block AFTER the history — never a
                    # mid-prompt re-render, so the cached prefix keeps climbing.
                    if runtime.settings.rag_append_delta:
                        if not frozen_reuse:
                            runtime._frozen_rag_tail = ragmod.render_knowledge_section(chunks)
                            runtime._rag_delta_chunks = []
                        elif not runtime._frozen_rag_tail:
                            runtime._frozen_rag_tail = ragmod.render_knowledge_section(chunks)
                        tail += runtime._frozen_rag_tail
                        rag_kbs = sorted({c["kb"] for c in chunks}
                                         | {c["kb"] for c in runtime._rag_delta_chunks})
                    else:
                        rag_kbs = sorted({c["kb"] for c in chunks})
                        tail += ragmod.render_knowledge_section(chunks)
                    if runtime.tracer:
                        runtime.tracer.span("rag", output={
                            "state": state_name, "ms": rag_ms, "kbs": rag_kbs,
                            "chunks": len(chunks), "query": query[:200],
                            "zero_hit": (len(chunks) == 0),
                            "delta_chunks": len(runtime._rag_delta_chunks)
                                if runtime.settings.rag_append_delta else 0,
                            "cached": not frozen_reuse and rag_ms <= 0.2,
                            "frozen_reuse": frozen_reuse, "drifted": drifted,
                            "drift_async": runtime.settings.rag_drift_async,
                            "prefetched": (not frozen_reuse and rag_ms == 0.0
                                           and query != "")})

            # iter31 C3: tool descriptions/params are dvs-FREE ({{var}} literal,
            # static bytes) — the tools block is the FIRST bytes of the request;
            # cached per (state, expand_kb) too.
            # iter32 payload gate: read the PREVIOUS round's payload.
            # silent + all-mechanical → this round gets NO tool channel (must speak).
            prev_tools = [t.get("name") or "" for t in (state.get("last_round_tool_calls") or [])]
            force_speech = (not state.get("round_spoke", True)) and bool(prev_tools) \
                and all(_MECHANICAL_RE.match(n) for n in prev_tools)
            if force_speech and not entry_lite:
                # iter43: keep the FULL tool array — tools render FIRST in the
                # OpenAI cached prefix, so tools=[] on a forced round busts the
                # whole prefix (Langfuse cache_read=0 on the iter32 forced
                # rounds). The SPEECH-ONLY directive in the tail asks the model
                # to hold its fire; the executor still runs anything it calls
                # (no worse than pre-iter32) and round_spoke re-evaluates.
                # iter49 T6: entry_lite is exempt — the ack is speech-only by
                # design and holds its fire via its OWN post-history
                # directive (appending the tail HERE would mutate the
                # pre-history prefix bytes; iter56: the state lives in the
                # delta now, the prefix is [tools][head][history]).
                if runtime.settings.force_speech_keep_tools:
                    tools = runtime._tools_cache.get((state_name, expand_kb))
                    if tools is None:
                        tools = build_tool_schemas(
                            runtime.states[state_name],
                            runtime.llm_json.get("general_tools", []),
                            lambda text: runtime.subst.subst(text, {}, kb=expand_kb),
                            {},
                        )
                        runtime._tools_cache[(state_name, expand_kb)] = tools
                    tail += ("\n\nTHIS ROUND: SPEECH ONLY — do not call any tool. "
                             "Reply to the caller in one or two short sentences, "
                             "then stop.")
                else:
                    tools = []          # iter32 (revertible): channel closed — speech is the only output
            elif lite:
                # iter48b (FIND-5): the lite round carries ONE do-nothing tool —
                # gpt-5.4 only prompt-caches requests WITH tools (raw cURL,
                # OTEL-proven); without it turn 1 was structurally cache-0.
                tools = [LITE_NOOP_TOOL]
            else:
                # Heavy rounds AND the iter49 T6 state-entry ack (FIND-5 port,
                # owner 2026-09-16 "cached tokens are the speed"): the ack
                # carries the FULL tool array, byte-identical to the heavy
                # round's — same (state, expand_kb) cache key (the ack's
                # expand is False; with a store present the heavy round's is
                # False too, i.e. every live round). tools=[] is structurally
                # uncacheable on gpt-5.4 (iter48b FIND-5) and busts the whole
                # cached prefix (the iter43 force-speech lesson), so the ack
                # would re-prefill ~3k fresh tokens at EVERY state change.
                # With the full array the ack's [tools][head] prefix IS the
                # heavy/warm prefix (iter56: no pre-history state-block — it
                # rides the delta) — it rides the already-landed
                # transition/greeting heavy warm at the same cache floor and
                # EXTENDS the cache for the heavy
                # round that follows. Speech-only rides the post-history
                # directive (see the delta below); a state whose entry must
                # run lanes/tools immediately opts out via
                # state_entry_lite_off.
                tools = runtime._tools_cache.get((state_name, expand_kb))
                if tools is None:
                    tools = build_tool_schemas(
                        runtime.states[state_name],
                        runtime.llm_json.get("general_tools", []),
                        lambda text: runtime.subst.subst(text, {}, kb=expand_kb),
                        {},
                    )
                    runtime._tools_cache[(state_name, expand_kb)] = tools
            # C3b layout: [static head] + history + [dynamic tail as trailing
            # system message]. iter44 T3: with tail_before_history the tail
            # sits BEFORE the history ([head][tail][history]) so the cached
            # prefix grows round-to-round. No tail when there is nothing to
            # say (empty dvs, no RAG lane) — never an empty system message.
            # iter44 append-only: drift delta rides AFTER the history.
            # iter56 (C2.2): under state_in_delta the DELTA composition order
            # is pinned state-block → RAG/pinned → ack directive (CR-2/P2):
            # the directive is imperative = FINAL; the state anchors first,
            # adjacent to the head's "CURRENT CALL STATE" pointer. Under the
            # revert flag the state rides the tail pre-history and the delta
            # stays RAG/directive-only (the iter55 layout, byte-exact).
            if live:
                # iter49 T4: the fresh working set REPLACES the post-history
                # delta each turn (placement, never a prefix rewrite).
                rag_delta = live_delta
            else:
                # iter49 T6: entry_lite already leaves live_delta/rag lanes
                # untouched — the revert-mode drift delta stays off the ack too.
                rag_delta = ragmod.render_knowledge_section(runtime._rag_delta_chunks) \
                    if (kb_store is not None and not entry_lite
                        and runtime.settings.rag_append_delta
                        and runtime._rag_delta_chunks) else ""
            speech_directive = ""
            if entry_lite:
                # iter49 T6 (FIND-5 port): the ack's speech-only directive
                # rides the POST-HISTORY delta slot — fresh bytes every
                # round, zero cached-prefix cost (the iter43 tail placement
                # would bust the state-block out of the prefix). Same text
                # as the force-speech directive (proven live, iter43); the
                # executor still runs anything the model calls despite it
                # (no worse than pre-iter32, and state_entry_lite_off is
                # the per-state escape hatch).
                speech_directive = ("THIS ROUND: SPEECH ONLY — do not call any "
                                    "tool. Reply to the caller in one or two "
                                    "short sentences, then stop.")
            stage_msg = ""
            if entry_lite:
                # iter49 T6: the ack delta STARTS as the speech-only
                # directive (revert path keeps exactly this); the
                # stage-static branch below may shrink it further.
                rag_delta = speech_directive
                if runtime.settings.state_in_delta \
                        and runtime.settings.delta_stage_static:
                    # iter96 (stage-static ack): a stage frame for this
                    # state may already exist (re-entry) — then the state
                    # block STAYS in the prefix and the ack delta shrinks
                    # to changed values + the directive (directive stays
                    # FINAL, CR-2/P2). No stage yet → the ack MINTS the
                    # dvs-frame stage (NO RAG — the ack runs no retrieval
                    # lanes, iter49 T6); the state's FIRST heavy round
                    # then UPGRADES the frame once (adds the RAG stage
                    # frame), and after that upgrade the frame is FROZEN.
                    staged = runtime._stage_blocks.get(state_name)
                    if staged is not None:
                        changed = runtime._changed_dvs_lines(staged["dvs"], dvs)
                        stage_msg = staged["block"]
                        rag_delta = (("\n\n# LIVE VALUES (changed)\n" + changed
                                      + "\n\n") if changed else "") \
                            + speech_directive
                    else:
                        state_block = runtime._state_block(dvs)
                        stage_msg = state_block
                        if state_block:
                            runtime._stage_blocks[state_name] = {
                                "block": state_block, "dvs": dict(dvs),
                                "rag": "", "upgraded": False}
                        rag_delta = speech_directive
                elif runtime.settings.state_in_delta:
                    # iter56 (F-09): the ack delta carries the LIVE state
                    # too — the model must see current values while it
                    # acknowledges. Directive stays FINAL (it's imperative).
                    state_block = runtime._state_block(dvs)
                    if state_block:
                        rag_delta = state_block + "\n\n" + rag_delta
            elif runtime.settings.state_in_delta \
                    and runtime.settings.delta_stage_static and not lite:
                # iter96 (stage-static split): the per-state frame (state
                # block + RAG stage frame) — byte-identical across the
                # block + RAG stage frame) — byte-identical across the
                # rounds of one state on the live path (measured: 4,501
                # chars ×4) — is promoted INTO the cached prefix as its own
                # system message BEFORE the history; the trailing delta
                # shrinks to the TRUE per-round change (changed dvs values
                # + fresh RAG only when it actually changed). The frame is
                # FROZEN while the state lasts: the only upgrade it takes
                # is the ack-minted bare dvs frame growing its RAG (once,
                # at the first heavy round); a mid-state RAG change rides
                # the trailing delta instead (never a mid-state prefix
                # bust). delta_tokens keeps its iter56 meaning (chars of
                # the trailing rag_delta); delta_static_chars +
                # delta_trailing_chars ride beside it (round-usage log +
                # tracer extra_meta).
                rag_full = rag_delta
                staged = runtime._stage_blocks.get(state_name)
                if staged is None:
                    # state entry (no ack minted one): the frame renders
                    # ONCE with the RAG — nothing trails
                    block = state_delta_block + \
                        (("\n\n" + rag_full) if rag_full else "")
                    stage_msg = block
                    runtime._stage_blocks[state_name] = {
                        "block": block, "dvs": dict(dvs),
                        "rag": rag_full, "upgraded": True}
                    rag_delta = ""
                elif not staged["upgraded"]:
                    # the ack-minted bare dvs frame upgrades ONCE with the
                    # RAG stage frame (the state's first heavy round)
                    block = state_delta_block + \
                        (("\n\n" + rag_full) if rag_full else "")
                    stage_msg = block
                    staged["block"] = block
                    staged["rag"] = rag_full
                    staged["upgraded"] = True
                    rag_delta = ""
                else:
                    # frozen frame: only the true change trails
                    changed = runtime._changed_dvs_lines(staged["dvs"], dvs)
                    stage_msg = staged["block"]
                    pieces = []
                    if changed:
                        pieces.append("# LIVE VALUES (changed)\n" + changed)
                    if rag_full and rag_full != staged["rag"]:
                        pieces.append(rag_full)
                    rag_delta = "\n\n".join(pieces)
            elif runtime.settings.state_in_delta and not lite:
                # iter56 (C2.2): state FIRST, RAG after — the delta IS the
                # last message the model reads; the state block anchors the
                # live values before the knowledge section tails it.
                if state_delta_block:
                    rag_delta = state_delta_block + rag_delta
            messages = self._build_messages(state_name, history, tail, system,
                                            delta=rag_delta, stage=stage_msg)

            t0 = time.perf_counter()
            gen_obs = runtime.tracer.start_llm(state_name, runtime.settings.openai_model,
                                               history[-6:]) if runtime.tracer else None
            content = ""
            tool_calls: list[dict] = []
            usage = None
            ttft_s: float | None = None   # iter44: time to FIRST spoken token (TTS-visible)
            separated = False   # iter16: one space between state texts in a multi-state walk
            # iter41 T1 / iter42: sentence-level TTS dedupe — buffer tokens per
            # sentence (flush on . ? ! / newline, or at stream end); an exact
            # duplicate (case-insensitive, whitespace-collapsed) of ANY of the
            # last `tts_dedupe_window` flushed sentences of this turn is dropped
            # from the SPOKEN stream only (model output/history/final content
            # untouched). Kept sentences are written with their ORIGINAL token
            # boundaries. The window carries across ROUNDS of the same turn (a
            # state_node call is one round; gpt-5.x repeats whole sentence
            # blocks — the Sofia/Carlos/Jorge A,B,A,B shape). Reset at ingest.
            # Silence guard: a round whose every sentence is a window hit still
            # speaks if nothing has been spoken yet this turn.
            pending_text = ""
            pending_tokens: list[str] = []
            window: list[str] = list(state.get("tts_sentence_window") or []) \
                if runtime.settings.tts_dedupe_sentences else []
            # iter59 T4: cross-turn question stem window — round-scoped copy
            # of the state key (plain key, no reducer; survives the call),
            # mutated by the token loop and written back in `updates`.
            stem_window: list[dict] = list(state.get("question_stem_window")
                                           or []) \
                if runtime.settings.tts_dedupe_semantic else []
            spoken_any = bool(state.get("turn_spoke"))
            async for ev in runtime.llm.astream(messages, tools,
                                                verbosity=runtime._verbosity_for(state_name)):
                # iter87 T1.4: ttft anchors on the FIRST delta of ANY type
                # (token or tool_call_start) — tool-only rounds previously
                # logged ttft=None and hid the LLM time-to-first-event.
                if ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                if ev["type"] == "token":
                    if state.get("turn_spoke") and not separated:
                        writer({"tts_token": " "})
                        separated = True
                    if runtime.settings.tts_dedupe_sentences:
                        pending_text += ev["text"]
                        pending_tokens.append(ev["text"])
                        while True:
                            cut = 0
                            m = _SENTENCE_SPLIT_RE.search(pending_text)
                            if m:
                                cut = m.end()
                            elif runtime.settings.tts_gate_fast_first_flush \
                                    and not spoken_any:
                                # iter46 T5: nothing spoken this turn yet —
                                # flush the FIRST clause early (first "," after
                                # ≥12 chars, or the char threshold) so the
                                # caller hears the reply one clause sooner.
                                # The fragment joins the dedupe window exactly
                                # like a full sentence (dedupe integrity holds:
                                # a later FULL sentence can never equal an
                                # earlier fragment's bytes).
                                cut = _fast_flush_cut(pending_text, runtime.settings)
                            if not cut:
                                break
                            head_tokens: list[str] = []
                            acc = 0
                            while pending_tokens:
                                tok = pending_tokens[0]
                                if acc + len(tok) <= cut:
                                    head_tokens.append(tok)
                                    acc += len(tok)
                                    pending_tokens.pop(0)
                                else:               # token straddles the boundary
                                    k = cut - acc
                                    head_tokens.append(tok[:k])
                                    pending_tokens[0] = tok[k:]
                                    break
                            pending_text = pending_text[cut:]
                            # iter65 T2: sanitize BEFORE the dedupe window —
                            # a code/JSON artifact sentence is dropped from
                            # speech entirely (never joins the window, never
                            # sets spoken_any); a stripped sentence re-keys
                            # so the window matches what was actually spoken.
                            sent_raw = "".join(head_tokens)
                            sent = _sanitize_spoken(sent_raw, runtime.settings)
                            if sent is None:
                                continue
                            if sent != sent_raw:
                                head_tokens = [sent]
                            key = _norm_sentence(sent)
                            if key and key in window and spoken_any:
                                continue            # in-turn window double — never spoken
                            # iter59 T4: semantic cross-turn dedupe — a QUESTION
                            # whose stem is a near-verbatim re-ask of a question
                            # already spoken THIS CALL is dropped from the
                            # spoken stream (same drop path as the exact dup:
                            # model output/history untouched). ONE embed per
                            # question sentence (vec reused for the window
                            # append); store None / embed failure -> exact-only
                            # fallback, never block speech. Statements and
                            # fast-flush fragments NEVER embed (fragments join
                            # the exact window as today — G5).
                            # iter60 AUD-6: no spoken_any gate — a turn-OPENING
                            # near-verbatim re-ask must be checked too; the
                            # non-empty stem_window itself proves a question
                            # was already spoken this call.
                            qvec = None
                            is_q = runtime.settings.tts_dedupe_semantic \
                                and "?" in sent
                            if is_q and stem_window:
                                qvec = await _embed_stem(runtime, sent)
                                if qvec is not None:
                                    thr = float(runtime.settings.tts_dedupe_cos_threshold)
                                    dup = False
                                    for entry in stem_window:
                                        prev = entry.get("vec")
                                        if prev and _cosine(qvec, prev) >= thr:
                                            dup = True
                                            break
                                    if dup:
                                        continue    # cross-turn near-verbatim re-ask — never spoken
                            for tok in head_tokens:
                                writer({"tts_token": tok})
                            if key:
                                window.append(key)
                                spoken_any = True
                                max_w = runtime.settings.tts_dedupe_window
                                if max_w > 0 and len(window) > max_w:
                                    del window[:len(window) - max_w]
                                # iter59 T4: spoken questions join the cross-turn
                                # stem window ({"text","vec"}; capped FIFO).
                                if is_q:
                                    if not _spoken_entry_has(stem_window, key):
                                        if qvec is None:
                                            qvec = await _embed_stem(runtime, sent)
                                        if qvec is not None:
                                            stem_window.append({"text": key,
                                                                "vec": qvec})
                                        max_sw = runtime.settings.tts_dedupe_stem_window
                                        if max_sw > 0 and len(stem_window) > max_sw:
                                            del stem_window[:len(stem_window) - max_sw]
                    else:
                        writer({"tts_token": ev["text"]})
                elif ev["type"] == "tool_call_start":
                    # iter46 T4a: fire the destination-state prewarm the
                    # moment a transition tool NAME completes in the stream
                    # (the tool name encodes the destination) — NOT after
                    # executor.execute() returns. Banks the rest of the
                    # speaking round + TTS drain (~0.5-1.5 s) as warm-prefill
                    # head start. The post-execute warm re-fires with the
                    # post-patch dvs (force=True supersedes in _latest_warm).
                    name = ev.get("name") or ""
                    dest = name[len("transition_to_"):] if name.startswith("transition_to_") else ""
                    if dest in runtime.states:
                        runtime.warm_prompt_cache(dest, list(history), dvs=dict(dvs))
                elif ev["type"] == "final":
                    content = ev["content"]
                    tool_calls = ev["tool_calls"]
                    usage = ev.get("usage")
            # iter43 T1: per-round token/cache telemetry — cache_read is the
            # OpenAI prefix-cache hit; the T2/T3 goal is it growing past the
            # tools+static-head floor across turns.
            # iter56 (C2.7): the round usage line carries state_in_delta +
            # delta_tokens (the post-history delta's char size) so the
            # G1/G2 gates and the report can attribute the layout per round.
            if usage:
                itd = usage.get("input_token_details") or {}
                # iter96: delta_static_chars = the stage frame's chars (the
                # promoted, cacheable part of the payload); delta_trailing_chars
                # = the true per-round trailing chars. delta_tokens KEEPS its
                # iter56 meaning (chars of the trailing rag_delta).
                log.info("round usage: turn=%s state=%s input=%s cache_read=%s "
                         "output=%s ttft=%sms state_in_delta=%s delta_tokens=%s "
                         "delta_static_chars=%s delta_trailing_chars=%s",
                         state.get("turn_index"), state_name,
                         usage.get("input_tokens"), itd.get("cache_read", 0) or 0,
                         usage.get("output_tokens"),
                         round(ttft_s * 1000) if ttft_s is not None else None,
                         bool(runtime.settings.state_in_delta),
                         len(rag_delta) if rag_delta else 0,
                         len(stage_msg) if stage_msg else 0,
                         len(rag_delta) if rag_delta else 0)
            if runtime.settings.tts_dedupe_sentences and pending_tokens:
                # iter65 T2: sanitize the stream-end flush FIRST — a code/
                # JSON artifact sentence is dropped entirely (pending
                # cleared; never spoken, never joins the window).
                pending_sent = _sanitize_spoken(pending_text, runtime.settings)
                if pending_sent is None:
                    pending_tokens, pending_text = [], ""
                else:
                    if pending_sent != pending_text:
                        pending_tokens = [pending_sent]
                    pending_text = pending_sent
                    pending_key = _norm_sentence(pending_text)
                    if not (pending_key and spoken_any and pending_key in window):
                        # iter59 T4: the stream-end flush is a FULL sentence — the
                        # semantic question check applies (fragments never embed).
                        # ONE embed for check + window append.
                        # iter60 AUD-6: no spoken_any gate (turn-OPENING re-asks
                        # are checked too); non-empty stem_window = never-silent.
                        pending_qvec = None
                        pending_is_q = runtime.settings.tts_dedupe_semantic \
                            and "?" in pending_text
                        if pending_is_q and stem_window:
                            pending_qvec = await _embed_stem(runtime, pending_text)
                            if pending_qvec is not None:
                                thr = float(runtime.settings.tts_dedupe_cos_threshold)
                                dup = any(e.get("vec") and
                                          _cosine(pending_qvec, e["vec"]) >= thr
                                          for e in stem_window)
                                if dup:
                                    pending_tokens, pending_text = [], ""
                                    pending_key = None
                                    pending_is_q = False
                        if pending_key and pending_tokens:
                            for tok in pending_tokens:
                                writer({"tts_token": tok})
                            window.append(pending_key)
                            spoken_any = True
                            max_w = runtime.settings.tts_dedupe_window
                            if max_w > 0 and len(window) > max_w:
                                del window[:len(window) - max_w]
                            if pending_is_q and not _spoken_entry_has(stem_window,
                                                                      pending_key):
                                if pending_qvec is None:
                                    pending_qvec = await _embed_stem(runtime, pending_text)
                                if pending_qvec is not None:
                                    stem_window.append({"text": pending_key,
                                                        "vec": pending_qvec})
                                max_sw = runtime.settings.tts_dedupe_stem_window
                                if max_sw > 0 and len(stem_window) > max_sw:
                                    del stem_window[:len(stem_window) - max_sw]
                    pending_tokens, pending_text = [], ""
            llm_s = time.perf_counter() - t0
            if runtime.tracer:
                runtime.tracer.finish_llm(
                    gen_obs, output=content, latency_s=llm_s, usage=usage,
                    tool_calls=[{"name": tc["name"], "args": tc["arguments"]} for tc in tool_calls],
                    ttft_s=ttft_s,
                    extra_meta={"state_in_delta": bool(runtime.settings.state_in_delta),
                                "delta_tokens": len(rag_delta) if rag_delta else 0,
                                "delta_static_chars": len(stage_msg) if stage_msg else 0,
                                "delta_trailing_chars": len(rag_delta) if rag_delta else 0},
                )

            updates: dict[str, Any] = {"assistant_text": content,
                                       "turn_spoke": bool(content) or bool(state.get("turn_spoke")),
                                       "round_spoke": bool(content),   # iter29: this round's speech flag
                                       # iter41 T1/iter42: carry the last-N SPOKEN sentence
                                       # keys to the next round of this turn (cross-round dedupe)
                                       "tts_sentence_window": window,
                                       # iter59 T4: cross-turn question stems ride
                                       # the round updates (plain key — last
                                       # write wins; NOT reset at ingest).
                                       "question_stem_window": stem_window,
                                       # iter30: last non-empty spoken text (end_call goodbye memory)
                                       "last_spoken": content if content.strip()
                                       else state.get("last_spoken", "")}
            new_history: list[dict] = []
            if tool_calls:
                # iter29: dedupe identical tool calls within one round — keep first by name
                seen: set[str] = set()
                tool_calls = [tc for tc in tool_calls
                              if not (tc["name"] in seen or seen.add(tc["name"]))]
                new_history.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in tool_calls
                    ],
                })
                dvs_patch: dict[str, Any] = {}
                new_state: str | None = None
                ended = False
                executed: list[dict] = []
                t1 = time.perf_counter()
                # iter40 L3: dead-air killer — an empty-text round whose pending
                # tool is the slots query speaks the ConfirmSlots.md Start line
                # (already in the prompt, no new content) via TTS while the
                # webhook runs. Anti-repeat per call via fillers_said.
                slots_filler = ("One moment while I check availability.")
                emit_filler = (runtime.settings.one_trip and not content.strip()
                               and not state.get("turn_spoke")
                               and slots_filler not in (state.get("fillers_said") or [])
                               and any(tc["name"] == "query_livecall_slots"
                                       for tc in tool_calls))
                if emit_filler:
                    writer({"tts_token": slots_filler + " "})
                    fillers = list(state.get("fillers_said") or []) + [slots_filler]
                    updates["fillers_said"] = fillers
                    updates["turn_spoke"] = True        # round 2 gets its separator
                for tc in tool_calls:
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    outcome = await runtime.executor.execute(tc["name"], args, dvs, state_name,
                                                             spoken_text=content,
                                                             last_spoken=state.get("last_spoken", ""))
                    executed.append({"name": tc["name"],
                                     "ok": bool(outcome.response.get("ok", True)),
                                     "status": outcome.response.get("status")})
                    dvs_patch.update(outcome.dvs_patch)
                    dvs.update(outcome.dvs_patch)   # same-round visibility (engine parity)
                    if outcome.new_state:
                        new_state = outcome.new_state
                    ended = ended or outcome.ended
                    new_history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(outcome.response)[:4000],
                    })
                if new_state:
                    updates["state_name"] = new_state
                    # iter40 L2: an ok transition INTO ConfirmSlots bg-warms
                    # the ASAP payload so the model's first slots call hits cache
                    if new_state == "ConfirmSlots":
                        runtime.warm_slots_async(dvs)
                    # iter43 T3 / iter44 T1: prewarm the DESTINATION state's
                    # full payload (tools + stripped head + tail + history
                    # snapshot) while the transition round still speaks.
                    # history + new_history is the exact next-round history
                    # (only ever appends). iter46 T4a: force=True — this warm
                    # re-fires after the tool-detection warm with the
                    # POST-PATCH dvs bytes (tail-byte parity with the entry
                    # round) and becomes the _latest_warm entry await target.
                    runtime.warm_prompt_cache(new_state,
                                              list(history) + list(new_history),
                                              dvs=dict(dvs), force=True)
                    # iter87 VERSION B fire law (one-ahead): entering X also
                    # warms X's FSM successors — lands during the caller's
                    # think time (real p50 17 s). Non-force: the latch
                    # dedupes; the eventual real transition re-fires with
                    # force=True (above) which supersedes _latest_warm.
                    # GATED on the hold knob (knob law): 0 = byte-exact
                    # iter85 — successor warms stage RAG, so they must not
                    # fire at default.
                    if runtime.settings.warm_await_dest_ms > 0:
                        for succ in runtime._state_successors.get(new_state, ()):
                            runtime.warm_prompt_cache(
                                succ, list(history) + list(new_history),
                                dvs=dict(dvs))
                    # iter44 T6b: bg RAG prefetch for the destination scope —
                    # staged chunks are consumed at the next state entry.
                    runtime.warm_rag_async(new_state, dvs=dict(dvs))
                if ended:
                    updates["ended"] = True
                updates["dvs"] = dvs_patch
                updates["history"] = new_history
                updates["last_round_tool_calls"] = executed
                # iter33: after a chain abort, the model's repair speech clears
                # the abort surface (the STATE BLOCK tail stops showing it).
                if content.strip() and dvs.get("chain_aborted_step"):
                    dvs_patch.update({"chain_aborted_step": "", "chain_aborted_msg": ""})
                updates["rounds_left"] = state.get("rounds_left", 0) - 1
                updates["metrics"] = {
                    "llm_s": round(llm_s, 3),
                    "tools_s": round(time.perf_counter() - t1, 3),
                    "calls": [tc["name"] for tc in tool_calls],
                    "rag_ms": rag_ms,
                    "rag_kbs": rag_kbs,
                }
            else:
                new_history.append({"role": "assistant", "content": content or ""})
                updates["history"] = new_history
                updates["last_round_tool_calls"] = []
                # iter33: pure-speech repair round after a chain abort — clear
                # the abort surface once the model actually spoke.
                if content.strip() and state.get("dvs", {}).get("chain_aborted_step"):
                    updates["dvs"] = {"chain_aborted_step": "", "chain_aborted_msg": ""}
                updates["metrics"] = {"llm_s": round(llm_s, 3), "tools_s": 0.0,
                                      "calls": ["<speak>"], "rag_ms": rag_ms,
                                      "rag_kbs": rag_kbs}
                # iter30: the goodbye IS the end of the call — deterministic.
                # The goodbye round is a pure-speech round (20+ rounds of live
                # evidence: gpt-5.2-class models never co-emit end_call with text), so the
                # "SPOKEN goodbye = call over" rule is engine-enforced here:
                # ended=True, no end_call call required, no hammer loop.
                # GATED by the SAME booking-claim truth gate as the end_call
                # tool (iter8): a livecall-agreed caller with no verified slot/
                # booking can NOT be hung up on — no bail-out, no fake booking.
                if content.strip() and _GOODBYE_RE.search(content) and not (
                        dvs.get("livecall_agreed")
                        and not (dvs.get("slot_verified") or dvs.get("booking_confirmed"))):
                    updates["ended"] = True
            # iter49 T6: remember the state this round RENDERED. The turn-1
            # lite round is exempt — it renders no state head, so the turn-2
            # entry into the initial state still counts as a state entry.
            if not lite:
                runtime._prev_round_state = state_name
            return updates

        state_node.__name__ = f"state_{state_name}"
        return state_node

    # ------------------------------------------------------------------ #
    async def _deterministic_node(self, state: GraphState) -> dict:
        """Runs after every state round. Deterministic, no LLM, no speech.

        - leak math auto-compute (open item #1: the pitch can't be skipped)
        - /today prefetch when entering ConfirmSlots without a date (open item #3)
        """
        dvs = DynamicVariables.from_flat(state.get("dvs", {})).to_flat()
        state_name = state.get("state_name", "")
        patch: dict[str, Any] = {}
        fired = False                        # iter29: engine acted this pass

        if leak_inputs_present(dvs) and not dvs.get("weekly_leak"):
            result, leak_patch = calculate_monthly_leak(dvs)
            if result.get("ok"):
                patch.update(leak_patch)
                fired = True
                if self.tracer:
                    self.tracer.span("deterministic:leak_math", output=result)

        # iter40 L2: every ConfirmSlots pass keeps the ASAP payload warm (fresh
        # → no-op). Background warm only — never counts as engine_fired.
        if state_name == "ConfirmSlots":
            self.warm_slots_async(dvs)

        if state_name == "ConfirmSlots" and not dvs.get("today_date"):
            outcome = await self.executor.execute("check_current_date", {}, dvs, "ConfirmSlots")
            if outcome.dvs_patch.get("today_date"):
                patch["today_date"] = outcome.dvs_patch["today_date"]
                fired = True
                if self.tracer:
                    self.tracer.span("deterministic:today_prefetch", output=outcome.dvs_patch)

        # iter33 Phase B: engine-owned booking chain. The model confirmed the
        # slot in conversation; everything past slot_verified is fixed-order
        # plumbing. One deterministic pass, resumable (each step skips if its
        # flag is set), fillers between the slow webhooks, abort -> model repair.
        if (self.settings.phaseb_chain
                and state_name in ("ConfirmSlots", "VerifyLead", "Booking")
                and not dvs.get("chain_done") and not dvs.get("booking_verified")
                and not dvs.get("booking_intent")
                and dvs.get("slot_verified") and dvs.get("selected_time")
                and dvs.get("prospect_timezone") and dvs.get("callback_number")
                and dvs.get("first_name")):
            try:
                writer = get_stream_writer()
            except Exception:
                writer = None
            fillers: list[str] = list(state.get("fillers_said") or [])
            executed: list[dict] = []
            uid = dvs.get("booking_uid", "")
            plan = [("validate_lead", "ConfirmSlots", not dvs.get("slot_verified")),
                    ("transition_to_VerifyLead", "ConfirmSlots", state_name == "ConfirmSlots"),
                    ("verify_lead_data", "VerifyLead", not dvs.get("data_verified")),
                    ("transition_to_Booking", "VerifyLead", state_name in ("ConfirmSlots", "VerifyLead")),
                    ("create_livecall_booking", "Booking", not uid),
                    ("record_booking_uid", "Booking", not dvs.get("booking_verified")),
                    ("record_booking_outcome", "Booking", not dvs.get("booking_verified")),
                    ("transition_to_Closing", "Booking", True)]
            # iter87 VERSION B fire law (chain): every transition_to_*
            # destination in the plan warms at chain start — the warm lands
            # inside the webhook idle (1.5-4.5 s) the caller already
            # tolerates. ALL destinations (not just Closing) so an abort's
            # repair round in a mid-chain state holds on a landed warm too
            # (owner-ratified 2026-09-30). state["history"] is byte-exact
            # for the post-chain ack (the chain appends no history).
            if self.settings.warm_await_dest_ms > 0:
                for _step, _home, _needs in plan:
                    if _step.startswith("transition_to_"):
                        _dest = _step[len("transition_to_"):]
                        if _dest in self.states:
                            self.warm_prompt_cache(
                                _dest, list(state.get("history") or []),
                                dvs=dict(dvs))
            cur_state = state_name
            patch["chain_done"] = True        # idempotency latch (popped on abort)
            emitted_filler = False            # iter41 T2: max ONE filler per pass
            for step, home, needs in plan:
                if not needs:
                    continue
                if step == "validate_lead" and dvs.get("slot_verified"):
                    continue                  # model already validated this pass
                if writer and step in _PHASEB_SLOW and not emitted_filler:
                    for line in _ENGINE_FILLERS:
                        if line not in fillers:
                            writer({"tts_token": line + " "})
                            fillers.append(line)
                            emitted_filler = True
                            break
                args = _chain_args(step, dvs, uid)
                if step == "record_booking_outcome":
                    args = {"booking_verified": True}   # engine asserts only on this path
                outcome = await self.executor.execute(step, args, dvs, home)
                status = outcome.response.get("status", "ok" if outcome.response.get("ok") else "error")
                executed.append({"name": step, "ok": status == "ok"})
                patch.update(outcome.dvs_patch)
                dvs.update(outcome.dvs_patch)   # same-round visibility
                if outcome.new_state:
                    cur_state = outcome.new_state
                if step == "create_livecall_booking" and outcome.response.get("booking_uid"):
                    uid = outcome.response["booking_uid"]
                if status in _PHASEB_ABORT:
                    patch.pop("chain_done", None)
                    patch["chain_aborted_step"] = step
                    patch["chain_aborted_msg"] = json.dumps(outcome.response)[:600]
                    if self.tracer:
                        self.tracer.span("phaseb:chain_abort", output={"step": step, "resp": outcome.response})
                    return {"dvs": patch, "engine_fired": True, "state_name": cur_state,
                            "fillers_said": fillers, "last_round_tool_calls": executed}
            # iter87 VERSION B: re-fire the final destination with
            # POST-PATCH dvs bytes (force=True supersedes the chain-start
            # warm in _latest_warm — the Closing ack's hold target).
            # GATED on the hold knob (knob law: 0 = byte-exact iter85).
            if self.settings.warm_await_dest_ms > 0:
                self.warm_prompt_cache(
                    "Closing", list(state.get("history") or []),
                    dvs=dict(dvs), force=True)
            if self.tracer:
                self.tracer.span("phaseb:chain_ok", output={"steps": [e["name"] for e in executed]})
            return {"dvs": patch, "engine_fired": True, "state_name": "Closing",
                    "fillers_said": fillers, "last_round_tool_calls": executed,
                    "metrics": {"phaseb_chain": [e["name"] for e in executed]}}

        # iter29: engine_fired MUST be reset every pass (a stale True from an
        # earlier pass would mask fire-and-forget one-trip finalization)
        if not patch and not fired:
            return {"engine_fired": False}
        return {"dvs": patch, "engine_fired": fired}

    # ------------------------------------------------------------------ #
    def _build(self, checkpointer=None):
        async def ingest(state: GraphState) -> dict:
            # iter38 F4 (defense-in-depth): idempotent user-append. Any
            # cancel/rerun path in the session layer must never duplicate the
            # user utterance in history again (2026-09-09 disaster call: ~40
            # duplicated user turns). Bookkeeping (turn_index etc.) still
            # advances so orchestration reruns stay visible in metrics.
            patch: dict[str, Any] = {
                "turn_active": True,
                "turn_spoke": False,
                "round_spoke": False,
                "engine_fired": False,
                "rounds_left": self.settings.max_tool_rounds,
                "last_round_tool_calls": [],
                # iter62: keep the turn transcript — state_node consumes
                # state["user_text"]; the next turn's payload overwrites it.
                "user_text": state.get("user_text", ""),
                # iter41 T1: turn-scoped reset — the cross-round dedupe
                # window never leaks across turns.
                "tts_sentence_window": [],
                "turn_index": state.get("turn_index", 0) + 1,
            }
            text = state.get("user_text", "")
            last_user = next((m for m in reversed(state.get("history", []))
                              if m.get("role") == "user"), None)
            if not (text and last_user is not None
                    and last_user.get("content", "") == text):
                patch["history"] = [{"role": "user", "content": text}]
            # iter40 C3: freeze the state-block bytes for THIS turn — every
            # round of the turn reuses them (prefix shared across rounds →
            # prompt-cache hits). Refresh happens at the NEXT turn's ingest.
            # iter56 (C2.4): INERT under state_in_delta — the pre-history
            # tail is empty in the new layout and nothing reads the frozen
            # bytes (the state rides the live delta every round). Kept for
            # the revert path (STATE_IN_DELTA=false); gated to skip the
            # dead write.
            if self.settings.frozen_state_block \
                    and not getattr(self.settings, "state_in_delta", False):
                dvs_flat = DynamicVariables.from_flat(state.get("dvs", {})).to_flat()
                patch["frozen_state_block"] = self._state_block(dvs_flat)
            return patch

        async def finalize(state: GraphState) -> dict:
            return {"turn_active": False}

        def route_state(state: GraphState) -> str:
            if state.get("ended"):
                return "finalize"
            name = state.get("state_name", self.llm_json.get("starting_state", "Intake"))
            return name if name in self.states else "finalize"

        def after_state(state: GraphState) -> str:
            if state.get("ended"):
                return "finalize"
            executed = state.get("last_round_tool_calls") or []
            if not executed:
                return "finalize"                    # pure speech round -> turn done
            if state.get("rounds_left", 0) <= 0:
                return "finalize"
            return "deterministic"                   # iter29: tool rounds ALWAYS pass the engine

        # iter29 ONE-TRIP routing: speech + all fire-and-forget bookkeeping tools
        # finalizes the turn in a single LLM trip; result-dependent tools and the
        # silence guard loop back into the state node.
        FIRE_AND_FORGET = re.compile(
            r"^(extract_|record_reach_details|set_callback_number|intake_completed|"
            r"discovery_completed|closer_completed|offer_completed|"
            r"contact_details_completed|record_booking_outcome|memory_note)")

        def route_after_deterministic(state: GraphState) -> str:
            if state.get("ended"):
                return "finalize"
            if state.get("engine_fired"):
                return route_state(state)            # engine has news -> model speaks it
            executed = state.get("last_round_tool_calls") or []
            # iter40 L3: an ok transition_to_* counts as terminal bookkeeping —
            # a spoken round that ONLY books/transitions finalizes in one trip.
            # gate_failed keeps the repair loop (status != "ok" → loops below).
            if self.settings.one_trip and executed and all(
                    FIRE_AND_FORGET.match(t["name"])
                    or (t["name"].startswith("transition_to_")
                        and t.get("status") == "ok")
                    for t in executed):
                if state.get("round_spoke"):
                    return "finalize"                # ONE TRIP: speech + bookkeeping/ok-transition
            if state.get("round_spoke") and executed and \
                    all(FIRE_AND_FORGET.match(t["name"]) for t in executed):
                return "finalize"                    # ONE TRIP: speech + bookkeeping, done
            return route_state(state)                # silence guard / result-dependent loop

        builder = StateGraph(GraphState)
        builder.add_node("ingest", ingest)
        for name in self.states:
            builder.add_node(name, self._make_state_node(name))
        builder.add_node("deterministic", self._deterministic_node)
        builder.add_node("finalize", finalize)

        builder.add_edge(START, "ingest")
        builder.add_conditional_edges("ingest", route_state,
                                      {**{n: n for n in self.states}, "finalize": "finalize"})
        for name in self.states:
            builder.add_conditional_edges(name, after_state,
                                          {**{n: n for n in self.states},
                                           "deterministic": "deterministic",
                                           "finalize": "finalize"})
        # deterministic layer routes back into the (possibly new) state node
        # (iter29: route_after_deterministic owns the one-trip decision)
        builder.add_conditional_edges("deterministic", route_after_deterministic,
                                      {**{n: n for n in self.states}, "finalize": "finalize"})
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=checkpointer or MemorySaver())

    # ------------------------------------------------------------------ #
    def initial_state(self, call_id: str, persona_dvs: dict | None = None) -> GraphState:
        self._rag_cache.clear()             # iter29: cache is per call, never across calls
        # iter31 C3: the RAG freeze is per call too (a fresh call re-retrieves
        # on its first state visit)
        self._frozen_state = None
        self._frozen_chunks = []
        # iter44: drift stash + staging are per call as well
        self._frozen_query_vec = None
        self._frozen_query_text = ""
        self._rag_staging.clear()
        self._rag_staging_tasks.clear()
        self._latest_warm.clear()          # iter46: warm tasks are per call
        self._warm_diag.clear()            # iter55 T0: warm failure diag is per call
        self._drift_tasks.clear()
        # iter49 T6: state-entry trigger baseline is per call
        self._prev_round_state = None
        # iter52: the pinned industry vertical is per call too
        self._pinned_industry = {}
        # iter96: the stage frames are per call too
        self._stage_blocks = {}
        return initial_state(call_id, self.llm_json, persona_dvs)


def build_checkpointer(settings: Settings):
    """Optional Postgres checkpointer (multi-worker / restart durability).
    Set LANGGRAPH_CHECKPOINT=postgres and DATABASE_URL=postgresql://...
    Falls back to per-call MemorySaver when the package or DB is unavailable."""
    if settings.checkpoint_backend != "postgres":
        return None
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async def _make():
            saver = AsyncPostgresSaver.from_conn_string(settings.database_url)
            await saver.setup()                      # one-time migrations
            return saver
        return _make()
    except Exception as exc:
        print(f"[checkpoint] postgres unavailable ({exc}); using MemorySaver")
        return None
