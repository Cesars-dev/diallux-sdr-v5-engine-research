"""iter95 pins — savage piece 1: warm cancel-not-wait (the abort cap).

Disease (savage README Proof 2, iter92b audit): 169 await_warm events —
163 already_done (96%), 6 lost races waited 1,218–2,324 ms of the 2,500 cap.
The wait sits in the S→L window (the caller's ear), not in TTFT.

Fix (ratified, owner 2026-10-06: cap 400 ms): at state entry a warm that is
NOT yet done gets up to `prewarm_abort_cap_ms` to land; at the cap it is
CANCELLED — the prefix cache it wrote at prefill survives server-side
(savage Proof 1: after-abort TTFT 857 ms ≈ hit-class 914 vs cold 1,421) and
the real round fires against it. cap=0 = legacy iter46 wait-shield (timeout
lets the warm keep running). Completed warms are adopted before any cap.

Fake tracer records `await_warm` spans for outcome assertions; fake warm
tasks are plain asyncio tasks so cancellation is real.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diallux.config import Settings
from diallux.graph.builder import CallRuntime


class FakeTracer:
    def __init__(self):
        self.spans = []

    def span(self, name, **k):
        self.spans.append((name, k.get("metadata", k)))


def _settings(**kw) -> Settings:
    return Settings(openai_api_key="test", retell_api_key="test",
                    langfuse_enabled=False, metrics_enabled=False, **kw)


def _rt(**kw) -> tuple[CallRuntime, FakeTracer]:
    tr = FakeTracer()
    rt = object.__new__(CallRuntime)
    rt.settings = _settings(**kw)
    rt._latest_warm = {}
    rt.tracer = tr
    return rt, tr


def _outcome(tr, key: str) -> dict:
    """Latest await_warm span for key (re-awaits supersede earlier spans)."""
    found = [meta for name, meta in tr.spans
             if name == "await_warm" and meta.get("key") == key]
    if not found:
        raise AssertionError(f"no await_warm span for {key}: {tr.spans}")
    return found[-1]


def _warm_body(delay: float, writes: list):
    async def body():
        await asyncio.sleep(delay)
        writes.append("history")          # adoption-only-on-completion
    return body()


def test_default_cap_is_400():
    """SWITCH LAW 4: code default = the ratified tested-on value."""
    assert Settings(openai_api_key="t", retell_api_key="t",
                    langfuse_enabled=False).prewarm_abort_cap_ms == 400


def test_completed_warm_adopts_unchanged():
    """A done warm is adopted before any cap applies (already_done, 0 ms),
    and a warm that lands within the cap still adopts (landed)."""
    rt, tr = _rt()

    async def scenario():
        rt._latest_warm["Discovery"] = _create(_warm_body(0.0, []))
        await asyncio.sleep(0.01)         # let it finish
        await rt._await_warm("Discovery")
        meta = _outcome(tr, "Discovery")
        assert meta["outcome"] == "already_done" and meta["waited_ms"] == 0
        assert not rt._latest_warm["Discovery"].cancelled()

        rt._latest_warm["Discovery2"] = _create(_warm_body(0.05, []))
        await rt._await_warm("Discovery2")          # cap 400 » 50 ms sleep
        meta = _outcome(tr, "Discovery2")
        assert meta["outcome"] == "landed"
        assert not rt._latest_warm["Discovery2"].cancelled()

    asyncio.run(scenario())


def test_midcompletion_warm_aborts_at_cap_round_proceeds():
    """A warm just past the bound (mid-completion) is cancelled; the call
    returns cleanly and the real round proceeds (no exception, no re-raise
    of the warm's cancellation). The abort cap is a CEILING: a context with
    its own tighter bound keeps it (entry 100); a long-tier context (dest
    2500) clamps to the 400 ceiling — the 1.2-2.4 s lost races die at 400."""
    writes: list = []
    rt, tr = _rt()

    async def scenario():
        rt._latest_warm["Discovery"] = _create(_warm_body(0.9, writes))
        await rt._await_warm("Discovery")           # entry bound 100
        meta = _outcome(tr, "Discovery")
        assert meta["outcome"] == "aborted"
        assert meta["cap_ms"] == 100                # ceiling keeps 100
        assert meta["waited_ms"] < 300
        rt._latest_warm["Closing"] = _create(_warm_body(0.9, writes))
        await rt._await_warm("Closing", wait_ms=2500)   # dest tier
        meta = _outcome(tr, "Closing")
        assert meta["outcome"] == "aborted"
        assert meta["cap_ms"] == 400                # clamped to the ceiling
        assert meta["waited_ms"] < 700              # not 1.2-2.4 s
        await asyncio.sleep(0)                      # let the cancels land
        assert rt._latest_warm["Discovery"].cancelled()
        assert rt._latest_warm["Closing"].cancelled()
        assert writes == []                         # no-garbage (see below)

    asyncio.run(scenario())


def test_preinflight_abort_is_baseline_round():
    """A warm far from done (prefill not even started) aborts at the cap —
    the bounded downside is today's baseline round. A re-await of the dead
    task reports `aborted`, never a false `already_done`."""
    rt, tr = _rt()

    async def scenario():
        rt._latest_warm["Intake"] = _create(_warm_body(30.0, []))
        await rt._await_warm("Intake", wait_ms=2500)    # long-tier context
        meta = _outcome(tr, "Intake")
        assert meta["outcome"] == "aborted" and meta["cap_ms"] == 400
        for _ in range(3):                          # let the cancel fully land
            await asyncio.sleep(0)
        await rt._await_warm("Intake")              # second entry, dead task
        meta2 = _outcome(tr, "Intake")
        # two spans: first aborted-after-wait, second aborted-immediately
        assert [m["outcome"] for m in
                [m for n, m in tr.spans if n == "await_warm"
                 and m.get("key") == "Intake"]] == ["aborted", "aborted"]
        assert meta2["waited_ms"] == 0

    asyncio.run(scenario())


def test_cap_zero_is_legacy_wait_shield():
    """cap=0: byte-legacy behavior — timeout outcome, the warm is NOT
    cancelled (shield keeps it running for the next round)."""
    rt, tr = _rt(prewarm_abort_cap_ms=0, prewarm_entry_wait_ms=80)

    async def scenario():
        rt._latest_warm["Discovery"] = _create(_warm_body(30.0, []))
        t = rt._latest_warm["Discovery"]
        await rt._await_warm("Discovery")
        meta = _outcome(tr, "Discovery")
        assert meta["outcome"] == "timeout"
        assert meta["cap_ms"] == 80                 # legacy bound = entry wait
        assert not t.cancelled() and not t.done()   # shield kept it alive
        t.cancel()                                  # test cleanup
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_aborted_warm_writes_nothing_to_history():
    """No-garbage invariant (savage README): an aborted warm never reaches
    its completion — adoption only accepts completed warm runs, so an
    aborted warm leaves zero writes behind."""
    writes: list = []
    rt, tr = _rt()

    async def scenario():
        rt._latest_warm["ConfirmSlots"] = _create(_warm_body(30.0, writes))
        await rt._await_warm("ConfirmSlots")
        assert _outcome(tr, "ConfirmSlots")["outcome"] == "aborted"
        await asyncio.sleep(0.01)
        assert writes == []                         # the completion write never ran

    asyncio.run(scenario())


def _create(coro) -> asyncio.Task:
    return asyncio.get_running_loop().create_task(coro)
