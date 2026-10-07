"""iter96 — stage-static delta split (delta-cache layout).

Pins the NEW payload contract (default flags: state_in_delta=True,
delta_stage_static=True):
  1. the per-state frame (state block + RAG) rides the CACHED PREFIX as its
     own system message BEFORE the history; round-2 of the same state reuses
     the BYTE-IDENTICAL stage block (stable [tools][head][stage] prefix)
  2. a dvs value change lands ONLY in the trailing live delta
     (# LIVE VALUES (changed) block) — the stage bytes never mutate
  3. a state transition replaces the stage message (the old state's frame is
     gone — not orphaned adjacent to the new one)
  4. delta_stage_static=False reproduces the iter56 layout byte-exactly
     ([head][history][delta = state + RAG])
  5. the tools array is unchanged across all rounds (FIND-5 cache floor)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diallux.config import Settings
from diallux.graph.builder import CallRuntime
from tests.fake_llm import FakeLLM, tool_call
from tests.mock_webhooks import mock_client

ROOT = Path(__file__).resolve().parents[1]
LLM_JSON = json.loads((ROOT / "agent" / "llm.json").read_text())
BLOCK_HEADING = "# CURRENT CALL STATE (live values)"
CHANGED_HEADING = "# LIVE VALUES (changed)"
SPEECH_DIRECTIVE = "THIS ROUND: SPEECH ONLY"


def _settings(**kw) -> Settings:
    base = dict(openai_api_key="test", retell_api_key="test",
                langfuse_enabled=False, first_turn_lite=False,
                rag_fire_mode="eot")
    base.update(kw)
    return Settings(**base)


def make_runtime(fake, settings=None, kb_store=None) -> CallRuntime:
    return CallRuntime(settings or _settings(), LLM_JSON, tracer=None, llm=fake,
                       http_client=mock_client(), kb_store=kb_store)


async def _turns(rt: CallRuntime, call_id: str, texts: list[str]) -> dict:
    config = {"configurable": {"thread_id": call_id}}
    payload = {"user_text": texts[0], **rt.initial_state(call_id)}
    for i, text in enumerate(texts):
        if i > 0:
            payload = {"user_text": text}
            await asyncio.gather(*rt._warm_tasks, return_exceptions=True)
        async for _m, _d in rt.graph.astream(payload, config=config,
                                             stream_mode=["updates"]):
            pass
    return (await rt.graph.aget_state(config)).values


class EmptyStore:
    async def retrieve_lanes(self, lanes):
        return [[] for _ in lanes]

    async def retrieve(self, query, slugs, vec=None):
        return []


class StableChunkStore:
    """Every retrieval returns THE SAME chunk — the live RAG delta is
    byte-stable across rounds (the observed phone-path shape: byte-identical
    4,501-char deltas across consecutive rounds)."""

    async def retrieve_lanes(self, lanes):
        return [[{"kb": "pain-points", "content": "stable stage chunk",
                  "score": 0.9}] for _ in lanes]

    async def retrieve(self, query, slugs, vec=None):
        return []


EXTRACT_ROUND = {
    "tokens": ["Got", " it", "."],
    "tool_calls": [tool_call("extract_intake_details", {
        "inbound_channel": "voicemail", "interest_topic": "missed calls",
        "pain_frame": "losing leads"})],
}


# --------------------------------------------------------------------------- #
# 1: stage block byte-stability across rounds of the same state
# --------------------------------------------------------------------------- #
def test_stage_block_byte_identical_round2_same_state():
    """iter96 (a): the per-state frame renders ONCE and rides pre-history;
    round-2 of the SAME state reuses the byte-identical stage message, so the
    [tools][head][stage] prefix is cache-stable and the trailing delta is
    empty (nothing changed)."""
    store = StableChunkStore()
    fake = FakeLLM([{"tokens": ["Hi", " there", "."]},
                    {"tokens": ["And", " more", "?"]}])
    rt = make_runtime(fake, _settings(), kb_store=store)
    asyncio.run(_turns(rt, "c96-1", ["hi, saw your ad", "yeah go on"]))
    r1, r2 = fake.seen_messages[0], fake.seen_messages[1]
    # layout: [head][stage][history...] — the stage is messages[1], pre-history
    assert r1[1]["role"] == "system" and BLOCK_HEADING in r1[1]["content"], \
        "the stage message (with the state frame) must ride BEFORE the history"
    assert "stable stage chunk" in r1[1]["content"], \
        "the stage carries the RAG frame too (state block + RAG)"
    assert r2[1]["content"] == r1[1]["content"], \
        "round-2 stage block must be BYTE-IDENTICAL to round-1 (stable prefix)"
    assert r2[0] == r1[0]
    # the trailing delta is EMPTY on round-2 (nothing changed) — no system
    # message after the history
    system_positions_r2 = [i for i, m in enumerate(r2) if m["role"] == "system"]
    assert system_positions_r2 == [0, 1], \
        f"round-2 layout must be [head][stage][history] with NO trailing delta, got {system_positions_r2}"
    # history grew append-only between the rounds
    hist1 = [m for m in r1[2:] if m["role"] != "system"]
    hist2 = [m for m in r2[2:] if m["role"] != "system"]
    assert hist2[:len(hist1)] == hist1


def test_stage_renders_once_not_per_round():
    """iter96 (a, companion): the stage frame is stored per state — a THIRD
    round in the same state reuses the same bytes (the stage is not
    re-rendered per round; it renders at state entry only)."""
    store = StableChunkStore()
    fake = FakeLLM([{"tokens": ["Hi", " there", "."]},
                    {"tokens": ["Got", " more", "."]},
                    {"tokens": ["And", " so", "?"]}])
    rt = make_runtime(fake, _settings(), kb_store=store)
    asyncio.run(_turns(rt, "c96-1b", ["hi, saw your ad", "yeah go on",
                                      "sure keep going"]))
    r1, r2, r3 = fake.seen_messages[0], fake.seen_messages[1], fake.seen_messages[2]
    assert r2[1]["content"] == r1[1]["content"]
    assert r3[1]["content"] == r1[1]["content"], \
        "every same-state round must carry the byte-identical stage block"


# --------------------------------------------------------------------------- #
# 2: dvs change lands ONLY in the trailing delta
# --------------------------------------------------------------------------- #
def test_dvs_change_lands_only_in_trailing_delta():
    """iter96 (b): after a capture the NEW values ride the trailing live
    delta (# LIVE VALUES (changed)) while the stage block bytes stay put."""
    store = EmptyStore()
    fake = FakeLLM([EXTRACT_ROUND, {"tokens": ["And", " what", " else", "?"]}])
    rt = make_runtime(fake, _settings(), kb_store=store)
    asyncio.run(_turns(rt, "c96-2", ["hi, saw your ad", "yeah go on"]))
    r1, r2 = fake.seen_messages[0], fake.seen_messages[1]
    # round-1: the stage mints the frame with the PRE-capture dvs; the extract
    # writes land in the graph AFTER the round — round-2's stage is unchanged
    assert r1[1]["content"] == r2[1]["content"], "stage bytes never mutate"
    trailing = r2[-1]["content"]
    assert r2[-1]["role"] == "system" and CHANGED_HEADING in trailing, \
        "the changed values must ride the trailing live delta"
    assert "inbound_channel: voicemail" in trailing
    assert "interest_topic: missed calls" in trailing
    assert "pain_frame: losing leads" in trailing
    # the values are NOT duplicated into the stage block
    assert "inbound_channel: voicemail" not in r2[1]["content"]


# --------------------------------------------------------------------------- #
# 3: state transition replaces the stage message
# --------------------------------------------------------------------------- #
def test_state_transition_replaces_stage_block():
    """iter96 (c): on a transition the NEW state's frame replaces the stage
    message — exactly one pre-history stage system message, carrying the new
    state's frame; the old frame is not orphaned adjacent to it."""
    calls = {"n": 0}

    class TwoChunkStore:
        async def retrieve_lanes(self, lanes):
            calls["n"] += 1
            chunk = "intake chunk" if calls["n"] <= 2 else "discovery chunk"
            return [[{"kb": "pain-points", "content": chunk, "score": 0.9}]
                    for _ in lanes]

        async def retrieve(self, query, slugs, vec=None):
            return []

    store = TwoChunkStore()
    fake = FakeLLM([
        EXTRACT_ROUND,                                     # turn 1 r1: Intake extract
        {"tokens": ["Got", " it", "."]},                   # turn 1 r2: speak
        {"tokens": ["Let", "'s", " dig", " in", "."],
         "tool_calls": [
             tool_call("intake_completed", {"intake_completed": True}),
             tool_call("transition_to_Discovery", {
                 "intake_completed": True, "inbound_channel": "voicemail",
                 "interest_topic": "missed calls",
                 "pain_frame": "losing leads"})]},         # turn 2: ok transition
        {"tokens": ["Sure", "."]},                         # filler rounds until the
        {"tokens": ["Sure", "."]},                         # walk lands on Discovery
        {"tokens": ["What", " makes", " you", " say", " that", "?"]},
    ])
    rt = make_runtime(fake, _settings(), kb_store=store)
    asyncio.run(_turns(rt, "c96-3", ["hi, saw your ad", "go on", "more",
                                     "yeah", "sure", "okay"]))
    d_r = fake.seen_messages[-1]                       # the Discovery heavy round
    system_msgs = [m for m in d_r if m["role"] == "system"]
    # exactly [head][stage] before the history — no orphaned old stage
    assert len(system_msgs) == 2, \
        f"post-transition layout must be [head][stage][history], got {len(system_msgs)} system msgs"
    assert "discovery chunk" in system_msgs[1]["content"], \
        "the stage message must carry the NEW state's frame"
    assert "intake chunk" not in system_msgs[1]["content"], \
        "the old state's frame must not ride the new stage"
    assert not any("intake chunk" in m["content"]
                   for m in d_r if m["role"] == "system"), \
        "the old stage must not sit adjacent to the new one"
    # the stage store now owns BOTH frames, distinct bytes, per state
    stages = rt._stage_blocks
    assert set(stages) >= {"Intake", "Discovery"}
    assert stages["Discovery"]["block"] != stages["Intake"]["block"]
    assert "discovery chunk" in stages["Discovery"]["block"]
    assert "intake chunk" not in stages["Discovery"]["block"]


# --------------------------------------------------------------------------- #
# 4: revert — delta_stage_static=False reproduces the iter56 layout
# --------------------------------------------------------------------------- #
def test_revert_flag_reproduces_iter56_layout():
    """iter96 (d): delta_stage_static=False = byte-exact iter56: NO stage
    message; [head][history...][delta(state + RAG)] with the state block
    FIRST in the delta."""
    store = StableChunkStore()
    fake = FakeLLM([EXTRACT_ROUND, {"tokens": ["And", " what", " else", "?"]}])
    rt = make_runtime(fake, _settings(delta_stage_static=False), kb_store=store)
    asyncio.run(_turns(rt, "c96-4", ["hi, saw your ad", "yeah go on"]))
    for msgs in (fake.seen_messages[0], fake.seen_messages[1]):
        system_positions = [i for i, m in enumerate(msgs) if m["role"] == "system"]
        assert system_positions == [0, len(msgs) - 1], \
            "revert layout must be [head][history...][delta]"
    delta = fake.seen_messages[1][-1]["content"]
    assert BLOCK_HEADING in delta and "stable stage chunk" in delta
    assert delta.index(BLOCK_HEADING) < delta.index("stable stage chunk")


# --------------------------------------------------------------------------- #
# 5: tools array unchanged across rounds (FIND-5 cache floor)
# --------------------------------------------------------------------------- #
def test_tools_array_stable_across_rounds():
    """iter96 (e): the stage split must NOT touch the tool channel — every
    round of the same state carries the FULL byte-identical tool array
    (iter48b FIND-5: gpt-5.4 only caches tool-bearing requests)."""
    store = StableChunkStore()
    fake = FakeLLM([EXTRACT_ROUND, {"tokens": ["And", " more", "?"]}])
    rt = make_runtime(fake, _settings(), kb_store=store)
    asyncio.run(_turns(rt, "c96-5", ["hi, saw your ad", "yeah go on"]))
    t1, t2 = fake.seen_tools[0], fake.seen_tools[1]
    assert t1 and t1 == t2, f"tools must be byte-identical across rounds, got {t1} vs {t2}"
