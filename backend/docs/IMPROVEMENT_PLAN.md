# DeerFlow Improvement Plan

Based on forensic source code analysis (2026-03-10). Ordered by impact.

---

## Phase 1 — Memory System Fixes (highest impact, isolated changes)

### 1.1 Inject facts into prompt
**Gap:** Facts are extracted, scored, stored — but `format_memory_for_injection()` skips them entirely.
**Fix:** Add facts injection in `src/agents/memory/prompt.py`. Format top-N facts (by confidence) into the `<memory>` block. Respect the 2000 token budget.
**Files:** `src/agents/memory/prompt.py`
**Risk:** Low — additive change, token budget already enforced.

### 1.2 Inject longTermBackground into prompt
**Gap:** `history.longTermBackground` is stored and updated but never read back.
**Fix:** Add it to `format_memory_for_injection()` alongside `recentMonths` and `earlierContext`.
**Files:** `src/agents/memory/prompt.py`
**Risk:** Low — same as above.

### 1.3 Fix 1000-char message truncation
**Gap:** `format_conversation_for_update()` truncates messages to 1000 chars before the LLM sees them. Long tool outputs and detailed responses lose important content.
**Fix:** Increase to 4000 chars or make configurable. The memory update LLM can handle it.
**Files:** `src/agents/memory/updater.py`
**Risk:** Low — slightly more tokens per memory update call.

### 1.4 Wire the FACT_EXTRACTION_PROMPT
**Gap:** A standalone fact extraction prompt exists in `prompt.py` but is never used. The memory update prompt handles facts as a side-product of summarization.
**Fix:** Evaluate if dedicated fact extraction improves quality. If so, add a second pass after memory update. If not, remove dead code.
**Files:** `src/agents/memory/prompt.py`, `src/agents/memory/updater.py`
**Risk:** Low — research + small change.

---

## Phase 2 — Subagent Competence (high impact, moderate scope)

### 2.1 Enrich subagent system prompts
**Gap:** Subagents get ~15 lines vs lead agent's 280+. Missing: thinking_style, date/time, response_style, critical_reminders, working directory details.
**Fix:** Build a `_build_subagent_prompt()` function that includes essential sections from the lead agent prompt. Not everything — skip memory, soul, clarification, subagent_system — but add thinking_style, date, response_style, critical_reminders, working directory.
**Files:** `src/subagents/builtins/general_purpose.py`, `src/subagents/builtins/bash_agent.py`, new shared prompt builder
**Risk:** Medium — larger prompts = more tokens per subagent. Must balance quality vs cost.

### 2.2 Enable thinking for subagents (configurable)
**Gap:** Thinking is hardcoded to disabled in `executor.py` for all subagents.
**Fix:** Make it configurable per subagent type in `SubagentConfig`. Default: disabled for bash, enabled for general-purpose.
**Files:** `src/subagents/executor.py`, `src/subagents/config.py`, `src/subagents/builtins/general_purpose.py`
**Risk:** Medium — more tokens, but general-purpose tasks need reasoning.

### 2.3 Add DanglingToolCallMiddleware to subagents
**Gap:** Subagents don't get this middleware. If interrupted mid-tool-call, next iteration sees orphaned tool calls.
**Fix:** Add `DanglingToolCallMiddleware()` to the subagent middleware chain in `executor.py`.
**Files:** `src/subagents/executor.py`
**Risk:** Low — proven middleware, just adding it to the chain.

---

## Phase 3 — Exploration Budget & Loop Control (high impact, prompt changes)

### 3.1 Add exploration budget guidance to lead agent prompt
**Gap:** No instruction about managing exploration depth. Causes runaway exploration (GraphRecursionError at 150) or timid exploration (reads 1 file and stops).
**Fix:** Add an `<exploration_budget>` section to the system prompt with rules: estimate scope before starting, set a file/step budget, checkpoint progress, know when to stop.
**Files:** `src/agents/lead_agent/prompt.py`
**Risk:** Low — prompt-only change, testable immediately.

### 3.2 Add exploration budget to subagent prompts
**Gap:** Same issue in subagents but worse — they have max_turns as the only limit.
**Fix:** Include budget guidance in the enriched subagent prompt (depends on 2.1).
**Files:** `src/subagents/builtins/general_purpose.py`
**Risk:** Low — prompt change.

---

## Phase 4 — Configuration & Limits (medium impact, small changes)

### 4.1 Fix SubagentLimitMiddleware silent clamping
**Gap:** Configured values outside [2,4] are silently clamped. No warning logged.
**Fix:** Log a warning when clamping occurs. Consider widening the range or making it configurable.
**Files:** `src/agents/middlewares/subagent_limit_middleware.py`
**Risk:** Low — logging + minor logic change.

### 4.2 Make thread pool sizes configurable
**Gap:** `_scheduler_pool` and `_execution_pool` are hardcoded at 3 workers each.
**Fix:** Read from config or SubagentConfig. Default 3, allow override.
**Files:** `src/subagents/executor.py`
**Risk:** Low — initialization change.

### 4.3 Fix str_replace single-occurrence validation
**Gap:** When `replace_all=False`, `str_replace` doesn't verify the match is unique. Could edit the wrong occurrence.
**Fix:** Count occurrences. If > 1 and `replace_all=False`, return error asking user to provide more context.
**Files:** `src/sandbox/tools.py`
**Risk:** Low — validation addition.

---

## Phase 5 — Research Quality (medium impact)

### 5.1 Increase web_fetch content limit
**Gap:** All community providers hard-truncate at 4KB (4096 chars). Research tasks get shallow content.
**Fix:** Make the limit configurable per provider via `config.yaml` tool config. Default to 8KB or 16KB.
**Files:** `src/community/tavily/tools.py`, `src/community/jina_ai/tools.py`, `src/community/searxng/tools.py`, `src/community/firecrawl/tools.py`, `src/community/infoquest/tools.py`
**Risk:** Low — more tokens in tool results, but research quality improves.

### 5.2 Sandbox file conflict protection for concurrent subagents
**Gap:** Multiple subagents share the same sandbox. No file-level coordination beyond OS.
**Fix:** Add advisory locking or per-subagent work directories within the shared sandbox.
**Files:** `src/subagents/executor.py`, `src/sandbox/tools.py`
**Risk:** Medium — requires careful design to not break existing tool paths.

---

## Phase 6 — Code Quality (low impact, good hygiene)

### 6.1 Replace datetime.utcnow() with datetime.now(timezone.utc)
**Gap:** `datetime.utcnow()` is deprecated since Python 3.12.
**Fix:** Find and replace across memory system.
**Files:** `src/agents/memory/updater.py`, `src/agents/memory/queue.py`
**Risk:** None — drop-in replacement.

### 6.2 Remove or wire FACT_EXTRACTION_PROMPT dead code
**Gap:** Dead code in `prompt.py` (depends on 1.4 evaluation).
**Files:** `src/agents/memory/prompt.py`
**Risk:** None.

---

## Estimated Effort

| Phase | Tasks | Complexity | Dependencies |
|-------|-------|-----------|-------------|
| 1 | 4 | Low | None |
| 2 | 3 | Medium | None |
| 3 | 2 | Low | Phase 2.1 for 3.2 |
| 4 | 3 | Low | None |
| 5 | 2 | Medium | None |
| 6 | 2 | None | Phase 1.4 for 6.2 |

Phases 1-3 are the highest value. Phases 1 and 3 are pure prompt/injection fixes with immediate impact.

---

*Plan created 2026-03-10 from forensic source code analysis.*
