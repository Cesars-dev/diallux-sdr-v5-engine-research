"""LangGraph state — V2 typed build.

Same shape as V1 (state_name / dvs / history / turn scratch), but dvs are
seeded through schema.DynamicVariables: typed defaults ("" / False), numeric
coercion, and no null passthrough — the Retell null-dv bug class is
structurally impossible, which is why V2 needs no "verify no null dvs" tool.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .schema import DynamicVariables


def merge_dict(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class GraphState(TypedDict, total=False):
    # identity
    call_id: str

    # Retell runtime replica
    state_name: str
    dvs: Annotated[dict[str, Any], merge_dict]
    history: Annotated[list[dict], operator.add]

    # per-turn scratch
    turn_active: bool
    turn_spoke: bool
    round_spoke: bool          # iter29: the LAST round produced speech (one-trip routing)
    engine_fired: bool         # iter29: the deterministic engine acted this pass
    last_spoken: str           # iter30: last NON-EMPTY agent speech (end_call goodbye memory)
    rounds_left: int
    last_round_tool_calls: list[dict]
    assistant_text: str
    ended: bool
    fillers_said: list[str]    # iter33: engine filler lines already spoken this call (complete-list overwrite)
    frozen_state_block: str    # iter40 C3: state-block bytes rendered ONCE at turn start (ingest)
    tts_sentence_window: list[str]  # iter42: last N SPOKEN sentence keys this turn (sliding dedupe window; reset at ingest)
    question_stem_window: list[dict]  # iter59 T4: cross-turn spoken QUESTION stems {"text","vec"} (plain key — NO reducer, NOT reset at ingest; survives the whole call; seeded empty in initial_state)

    # user input channel
    user_text: str

    # observability
    turn_index: int
    metrics: Annotated[dict[str, Any], merge_dict]


def begin_message(llm_json: dict) -> str:
    msg = (llm_json.get("begin_message") or "").strip()
    return msg or "Hello! You have reached Dialux, this is Linda speaking, how may I help you today?"


def initial_state(call_id: str, llm_json: dict, persona_dvs: dict | None = None) -> GraphState:
    """Typed seed: defaults come from DynamicVariables (no stale dates, no nulls)."""
    flat_defaults = DynamicVariables().to_flat()
    # only accept known keys from persona overrides (e.g. Twilio <Parameter> seeds)
    persona = {k: v for k, v in (persona_dvs or {}).items() if k in DynamicVariables.model_fields}
    dvs = DynamicVariables.from_flat({**flat_defaults, **persona}).to_flat()
    return GraphState(
        call_id=call_id,
        state_name=llm_json.get("starting_state", "Intake"),
        dvs=dvs,
        history=[{"role": "assistant", "content": begin_message(llm_json)}],
        turn_active=False,
        turn_spoke=False,
        rounds_left=0,
        ended=False,
        turn_index=0,
        question_stem_window=[],
        metrics={},
    )
