# Diallux SDR v5 — Voice Agent Engine (research snapshot)

LangGraph-based voice-agent engine for an outbound/inbound SDR ("Linda"). A 9-state
state machine with hard-gated transitions, typed dynamic variables (dvs), pgvector RAG,
deterministic leak math, and an LLM-to-LLM acceptance harness. This is a **single-commit
snapshot** of the `engine/` subtree published for a latency-research effort; it is not the
live repo (no git history, no secrets, `.env` values removed — see `.env.example`).

## Stack

- Python 3.12, LangGraph state machine (`diallux/graph/builder.py` — 3k lines, the heart)
- LLM: gpt-5.4 via `langchain-openai` `ChatOpenAI.astream()` (`diallux/graph/llm.py`)
- STT: Deepgram streaming WS, eager-EOT (`diallux/media/deepgram_stt.py`)
- TTS: ElevenLabs turbo_v2_5 streaming / Cartesia sonic (`diallux/media/*_tts.py`)
- Transports: Twilio Media Streams (mulaw 8k) + browser mic WS (`diallux/app.py`,
  `diallux/media/session.py`) — same graph, zero transcoding on the Twilio path
- Observability: OTEL → Langfuse (`diallux/observability/tracer.py`)

## The latency problem being researched

Measured (engine round-usage logs, `ttft=` = LLM dispatch → first stream event):

| path | TTFT p50 | notes |
|---|---|---|
| chat harness, idle event loop, gpt-5.4 | ~728 ms | 43 traces, same model/payloads |
| chat harness, US box, idle loop | ~650 ms | |
| mic/phone path (either box) | ~900–1,065 ms | includes ~250–350 ms of asyncio event-loop contention |
| uncached-prefill slope | 8–22 ms / 1k tokens | R² ≈ 0 — prefill is NOT the bottleneck |

TTFT is flat vs prompt size and cache depth; the gap between chat (~730) and mic (~950)
is hypothesized to be single-threaded asyncio event-loop contention from co-running
media tasks (STT WS frames in, TTS streaming out, base64/mulaw frame work) delaying
pickup of the first LLM stream event. e2e turn p50 today ≈ 1.9 s; goal ≤ 1.2 s.

## Key files for the research questions

- `diallux/app.py` — WS endpoints (`/mic/ws`, Twilio media streams), per-frame handlers
- `diallux/graph/builder.py` — the state-node loop: `astream()` consumption, `ttft_s`
  measurement (t0 at line ~2404, first-event anchor ~2439), TTS token writer, per-round
  usage logging (~2573)
- `diallux/graph/llm.py` — ChatOpenAI config, shared httpx client, boot prewarm,
  reasoning_effort/verbosity model_kwargs
- `diallux/media/session.py` — session orchestration, codec paths
- `diallux/media/deepgram_stt.py`, `elevenlabs_tts.py`, `cartesia_tts.py` — streaming media clients
- `diallux/media/sentence_gate.py` — sentence-level gate/sanitizer before TTS first byte
- `diallux/config.py` — pydantic Settings, all behavior switches
- `agent/llm.json` — the 9-state machine definition (states, tools, gates, dvs)
- `engine/diallux/prompts/*.md` — live prompts
- `tests/llm2llm/` — LLM-to-LLM acceptance harness + 25 personas
