# DEEP RESEARCH PROMPT — Voice-agent latency & dual-model architecture
(paste-ready for a deep-research/web-surfing LLM)

---

## ROLE

You are a senior voice-AI infrastructure engineer and researcher. You have shipped
production telephony agents (Twilio Media Streams, SIP, WebRTC) and you understand
asyncio internals, OpenAI streaming APIs, and the commercial voice-agent platforms at
the architecture level, not the marketing level. You will research the web AND read a
real codebase, then produce engineering decisions, not a survey.

## SUBJECT CODEBASE (read it first)

https://github.com/Cesars-dev/diallux-sdr-v5-engine-research

Start with `RESEARCH_CONTEXT.md`, then read: `diallux/app.py` (WS endpoints),
`diallux/graph/builder.py` (state-node loop: astream consumption, ttft measurement,
TTS token writer), `diallux/graph/llm.py`, `diallux/media/session.py`,
`diallux/media/deepgram_stt.py`, `diallux/media/elevenlabs_tts.py`,
`diallux/media/sentence_gate.py`, `diallux/config.py`, `agent/llm.json` (9-state machine,
tools, gates, dynamic variables). This is a LangGraph voice SDR engine ("Linda"):
Twilio Media Streams + browser-mic WS transports, Deepgram STT (streaming WS, eager EOT),
ElevenLabs turbo TTS, gpt-5.4 via langchain ChatOpenAI `.astream()`.

## MEASURED GROUND TRUTH (do not re-derive, do not contradict without evidence)

- TTFT here = LLM request dispatch → first stream event observed (engine-side wall clock).
- Chat path (idle asyncio loop, same model/payloads): p50 ~728 ms (EU box) / ~650 ms (US box).
- Mic/phone path (either box): p50 ~900–1,065 ms. Hypothesis (not yet proven): ~250–350 ms
  is single-threaded asyncio event-loop contention — STT WS frames in, TTS streaming out,
  per-frame base64/mulaw work — delaying pickup of the first LLM stream event.
- TTFT is FLAT vs prompt size and prefix-cache depth: 8–22 ms per 1,000 uncached tokens,
  R² ≈ 0. Caching deeper buys tens of ms, not hundreds. The model/API first-token floor
  (~650–730 ms) dominates. Prefill is NOT the bottleneck.
- gpt-5.4 runs with reasoning_effort="none"; it is still a reasoning-family model
  (temperature ignored server-side).
- e2e turn p50 today ≈ 1.9 s. Goal ≤ 1.2 s on real telephony.
- Historical reference: the same agent logic on Retell with gpt-4.1 (2025) felt ~1.2 s e2e.

## RESEARCH QUESTIONS

### Q1 — Media endpoint architecture: how do the platforms do it, and is ours good enough?

Research the ACTUAL production architectures of Retell AI, Vapi, Bland, LiveKit Agents
/Pipecat, and OpenAI Realtime API voice mode. For each, find (engineering blogs, docs,
open-source code, conference talks — not landing pages):
- Where does their audio I/O loop live relative to the LLM call? Dedicated process/thread?
  How do they isolate STT in, LLM pickup, and TTS out from each other?
- Do they consume the LLM as a stream (`astream`/SSE) or via a lower-level path? OpenAI
  Realtime uses a single WS for audio+tokens — when is that architecture strictly better
  than chat-completions streaming, and can a chat-completions stack ever match it?
- What TTFT / e2e response-latency numbers do they publish or benchmark (e.g. Vapi's
  "end-of-speech to first audio" benchmarks, LiveKit agents benchmarks)?
- VERDICT NEEDED: given our codebase (single asyncio loop per session, per-frame handlers
  in app.py, astream consumption in builder.py), is our endpoint architecture sound, and
  is `astream` the right consumption primitive — or is the pattern itself the ceiling?

### Q2 — Why can Retell be faster than our self-hosted infra? What do they have that we don't?

We run Twilio Media Streams → our VPS → Deepgram/ElevenLabs/OpenAI. Retell presumably runs
the same vendors. Research and enumerate concretely:
- Telephony ingress: do they terminate media closer to Twilio (multi-region media
  servers, SIP forks, pre-warmed connections)? Impact of Twilio's own media relay hops.
- Do they colocate with STT/TTS/LLM (same region/AZ, persistent connections, connection
  pooling, HTTP/2 or WS multiplexing)? What does cross-region RTT cost at each hop?
- Model serving: do platforms get priority/low-latency OpenAI endpoints, cached sessions,
  or speculative streaming? Any evidence of distillation or "first-token SLO" products?
- Anything structural we cannot replicate on a single VPS (and what it would take:
  e.g., a US media gateway, a second box, dedicated loop per session)?
- VERDICT NEEDED: rank the realistic gap-closers for OUR stack by ms-saved per dollar
  and engineering effort.

### Q3 — The dual-model "fast mouth, smart hands" architecture (the core question)

Proposal: a fast non-reasoning model (gpt-4.1) generates the SPOKEN stream (fast TTFT,
streamed to TTS immediately); gpt-5.4 runs in the background and owns: dynamic-variable
extraction (typed dvs set from caller speech), state-transition decisions (a single
transition boolean + tool calls), gate enforcement, KB/RAG decisions, booking actions.
State transitions happen at schema level — the real signal is "dvs x,y,z = true + one
transition boolean" — which is gpt-5.4's job, NOT the mouth's.

Research prior art and production patterns:
- LLM cascades / draft-and-verify / speculative generation in conversational agents;
  "fast responder + slow reasoner" patterns; any production voice agents doing exactly
  this (search: dual model voice agent, draft model streaming, fast slow path LLM,
  speculative dialogue, cascade serving, FractalGPT/FrugalGPT-style routing for latency).
- How OpenAI Realtime's own "speculative responses" or Gemini Live's architecture
  handle the fast-path/slow-path split, if documented.
- Constrained generation techniques to MUZZLE the fast mouth precisely:
  logit/constrained decoding (guidance, outlines, JSON schema enforcement), system-prompt
  hardening, "may only speak from provided facts" patterns, grounding the mouth strictly
  in: current state block + extracted dvs + RAG chunks — i.e., the mouth may only
  verbalize supplied facts, never invent prices/availability/confirmations.
  What works in production? What fails? Hallucination-rate evidence for
  4.1-class models on grounded verbalization tasks.
- Reconciliation design: the mouth speaks BEFORE the brain's gate verdict exists.
  Patterns for asynchronous verification of already-streamed speech (post-hoc correction,
  barge-in correction, "hold the last sentence" buffering, sentence-level gating of
  commitments while free chat streams raw). Our engine already has a sentence-level
  gate/sanitizer before TTS first byte (`sentence_gate.py`) — evaluate using it as the
  muzzle point: mouth streams freely for non-committal sentences; money/date/booking
  sentences pass a cheap validator (rules or 4.1 self-check) before TTS release.
- Same prompt for both models, or two prompts? Evidence on shared-prompt vs
  role-specialized prompts when one model verbalizes and another decides. Risks of
  drift between mouth and brain on the SAME state.
- VERDICT NEEDED: a production-grade architecture for this codebase — which model sees
  what, prompt strategy, the exact muzzle mechanism, the reconciliation protocol, failure
  modes with mitigations, and what to prototype first. Be specific to LangGraph
  (parallel branches / background tasks) and our files.

## OUTPUT FORMAT

For each Q: findings with sources (links) → comparison to OUR code (cite file/line) →
concrete verdict + recommended changes ranked by (ms saved)/(effort). End with a single
"first prototype" spec: the smallest change that tests the biggest assumption.
Flag anything where public information is thin and you are extrapolating — mark it
EXTRAPOLATION. No generic advice ("use caching", "optimize network") without numbers.
