# Agentic workflow — architecture notes (`agentic_workflow` branch)

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).

## What this branch is

A LangChain-based agentic re-implementation of the EDA judging. Same three
judgements (Q0 human-or-AI, Q1 personality, Q2 content), same fixed
within-person split, same prompts, same Pydantic schemas, same output CSVs —
but each judge model runs as its **own agent** with its **own call log**, so
judges can be tracked, monitored and scaled independently.

```
run_agentic_eda.py                 entrypoint (mirrors run_eda.py outputs)
  └─ src/agents/orchestrator.py    plain functions: fan Q0/Q1/Q2 to each agent
       └─ src/agents/judge_agent.py  JudgeAgent — one per model in eda.judges
            ├─ LCEL chain: ChatOllama (temp 0, seed, format=json)
            │              | RunnableLambda(extract text + token usage)
            ├─ cache/agentic/<hash>.json   disk cache, (model, prompt hash)
            └─ src/agents/monitor.py       AgentMonitor → JSONL per agent
                                           output/agent_logs/<model>.jsonl
```

## One agent per model — why

- **Per-model tracking.** Each `JudgeAgent` owns an `AgentMonitor` writing
  one JSONL line per call (timestamp, prompt hash, latency, status
  success/repair/fail/cache_hit, token counts). `monitor.summary()`
  aggregates per agent: calls, mean latency, failure rate. A slow or flaky
  judge is visible immediately, in its own file, without untangling a shared
  log.
- **Independent scaling.** Agents share nothing but the input DataFrame.
  Adding a judge is one line in `config.yaml eda.judges`; removing one loses
  nothing else. When scale demands it, each agent can be fanned out
  separately (see limitations).
- **Judge diversity.** The evaluation deliberately uses three model families
  (gpt-oss:120b, deepseek-v4-flash, glm-5.2) to guard against self-preference
  — the generator model is unknown. One agent per family keeps their
  verdicts, caches and logs strictly separate, so inter-judge agreement is
  computed on genuinely independent tracks.

## LangChain, but LCEL only — no LangGraph

Plan §1 is explicit: *"LangChain LCEL chains. No LangGraph, no multi-agent
choreography. If a step can be a function, it is a function."* The chain
inside an agent is exactly two runnables: `ChatOllama | RunnableLambda`.
Everything around it — caching, retry-with-repair, logging, fan-out — is
plain Python functions and loops (`orchestrator.py`). The orchestrator is
three loops, not a graph.

## What is reused (deliberately, so the two pipelines cannot drift)

| Reused | From |
|---|---|
| Pydantic response schemas (`IsAI`, …) | `src/eda_models.py` |
| Prompt templates (no wording in code) | `user_prompts/eda/` |
| Prompt loading, fence stripping, schema block | `src/llm.py` |
| Fixed split + split saving | `src/eda.py` |
| Metrics, reasons, style features | `run_eda.py` |
| Judge-input lowercasing rule | `run_eda.normalise()` (mirrored with the flag passed in) |

Cache behaviour mirrors `src/llm.py`: keyed on (model, prompt hash), only
validated replies are cached, one retry-with-repair on `ValidationError` —
but under `cache/agentic/` so agentic runs never collide with the EDA cache.

## LIMITATIONS — deliberately deferred until scaling

Pinned here so nobody mistakes omission for oversight. At 3 people × 10
questions × 3 judges (~84 calls) none of these pay for their complexity;
at 1,000 people × 50 questions they all become necessary.

- **No memory optimization.** Each call is stateless and self-contained; the
  full prompt (context answers included) is rebuilt and resent every time.
  No conversation memory, no shared persona state between calls.
- **No context-window optimization.** Q1 context answers are concatenated
  verbatim with no token counting, truncation, summarisation or packing. At
  this data size every prompt fits comfortably; at scale long interviews
  would overflow.
- **No batching or parallelism.** Calls are sequential, one agent at a time,
  synchronous `invoke()`. Simple to read and debug; slow at scale.

What scaling would need (in order):

1. **Async fan-out** — `chain.abatch()` / `asyncio.gather` per agent, with a
   per-agent concurrency limit; agents are already independent, so this is
   mechanical.
2. **Token budgeting** — count prompt tokens before sending, cap or
   summarise Q1 context to a budget, and track spend per agent in the
   monitor (the JSONL already records token counts where the backend
   reports them).
3. **Shared context store** — persona context built once per person and
   referenced by every agent/question, instead of re-embedding it in each
   prompt string; at scale, a keyed store (person_id → context block) with
   cache-friendly prompt prefixes.
