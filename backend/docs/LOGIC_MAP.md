# DeerFlow Logic Map

Complete execution logic mapped from source code by Opus forensic analysis.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                                 │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ LangGraph    │  │ DeerFlow     │  │ IM Channels  │             │
│  │ Server :2024 │  │ Client       │  │ (Telegram,   │             │
│  │ (HTTP API)   │  │ (embedded)   │  │  Slack, etc) │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│                            ▼                                        │
│              ┌─────────────────────────┐                            │
│              │    create_agent()       │                            │
│              │    (langchain.agents)   │                            │
│              │                         │                            │
│              │  model + tools +        │                            │
│              │  middlewares +           │                            │
│              │  system_prompt +        │                            │
│              │  state_schema           │                            │
│              └────────────┬────────────┘                            │
│                           │                                         │
│                           ▼                                         │
│              ┌─────────────────────────┐                            │
│              │   Compiled LangGraph    │                            │
│              │   (ReAct Agent Loop)    │                            │
│              └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

**Key files:**
- `langgraph.json` — registers `make_lead_agent` as the graph factory
- `src/agents/lead_agent/agent.py` — `make_lead_agent()` builds the agent
- `src/client.py` — `DeerFlowClient` embeds the same agent without HTTP
- `src/channels/manager.py` — IM bridges call LangGraph Server via `langgraph-sdk`

---

## 2. The ReAct Loop (Graph Nodes & Edges)

DeerFlow does NOT use a custom StateGraph. It uses LangChain's `create_agent()` which builds a standard ReAct loop:

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
         ┌─────────────────────────────────┐
         │         AGENT NODE              │
         │                                 │
         │  1. before_model middlewares     │
         │  2. wrap_model_call middlewares  │
         │  3. LLM CALL                    │
         │  4. after_model middlewares      │
         │                                 │
         └─────────────┬───────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  tool_calls?   │
              └───┬────────┬───┘
                  │        │
            YES   │        │  NO
                  │        │
                  ▼        ▼
   ┌──────────────────┐  ┌──────┐
   │   TOOLS NODE     │  │ END  │
   │                   │  └──────┘
   │  wrap_tool_call   │
   │  middlewares →     │
   │  execute each     │
   │  tool call        │
   └────────┬──────────┘
            │
            │  ToolMessages added to state
            │
            └──────────► back to AGENT NODE
```

**Conditional edge logic:**
- `AIMessage.tool_calls` non-empty → go to TOOLS node
- `AIMessage.tool_calls` empty (final text) → go to END
- `Command(goto=END)` from middleware → go to END (interrupt)

---

## 3. State Schema (ThreadState)

```
ThreadState (extends AgentState)
├── messages: list[BaseMessage]          ← standard message list with add_messages reducer
├── sandbox: SandboxState | None         ← {sandbox_id: str}
├── thread_data: ThreadDataState | None  ← {workspace_path, uploads_path, outputs_path}
├── title: str | None                   ← auto-generated thread title
├── artifacts: list[str]                 ← deduplicated output file paths
├── todos: list | None                  ← plan mode task list
├── uploaded_files: list[dict] | None   ← current upload metadata
└── viewed_images: dict[str, ViewedImageData]  ← path → {base64, mime_type}
```

**File:** `src/agents/thread_state.py`

---

## 4. Middleware Chain (Lead Agent)

Middlewares execute in strict order. Each has a specific hook type that determines WHEN it fires.

```
REQUEST IN ─────────────────────────────────────────────────────────►

═══ BEFORE_AGENT (fires once at start) ═══════════════════════════

  1. ThreadDataMiddleware     → computes per-thread directory paths
  2. UploadsMiddleware        → injects <uploaded_files> block into last HumanMessage
  3. SandboxMiddleware        → acquires sandbox (lazy by default)

═══ BEFORE_MODEL (fires before each LLM call) ════════════════════

  10. ViewImageMiddleware*    → injects base64 image HumanMessage if view_image completed

═══ WRAP_MODEL_CALL (wraps each LLM call) ════════════════════════

  4. DanglingToolCallMiddleware → patches missing ToolMessages in history
  5. OutputRepairMiddleware     → retries if LLM output malformed (max 2 retries)
  13. SelfEvaluationMiddleware* → evaluates final text, requests revision if needed

═══ [LLM CALL HAPPENS HERE] ══════════════════════════════════════

═══ AFTER_MODEL (fires after each LLM call) ══════════════════════

  11. SubagentLimitMiddleware*  → truncates excess task calls to max 3 (clamped [2,4])
  12. LoopDetectionMiddleware   → warn at 3 repeats, hard-stop at 5 repeats

═══ AAFTER_MODEL (async, fires after each LLM call) ═════════════

  8. TitleMiddleware*          → generates thread title after first exchange

═══ WRAP_TOOL_CALL (wraps each tool execution) ═══════════════════

  14. ClarificationMiddleware  → intercepts ask_clarification → Command(goto=END)

═══ AFTER_AGENT (fires once at end) ══════════════════════════════

  9. MemoryMiddleware          → queues conversation for async memory update

  * = conditional (see notes below)

RESPONSE OUT ◄──────────────────────────────────────────────────────
```

**Conditional middlewares:**
- 6\. SummarizationMiddleware — only if `summarization.enabled`
- 7\. TodoListMiddleware — only if `is_plan_mode`
- 8\. TitleMiddleware — only if `title.enabled`
- 10\. ViewImageMiddleware — only if model `supports_vision`
- 11\. SubagentLimitMiddleware — only if `subagent_enabled`
- 13\. SelfEvaluationMiddleware — only if `self_evaluation_enabled`

**Key files:** `src/agents/middlewares/*.py`, `src/sandbox/middleware.py`

---

## 5. Subagent System

### 5.1 Spawning Decision

The LLM decides to spawn subagents based on **prompt instructions** in `<subagent_system>` (from `prompt.py`). The prompt tells it:

1. **DECOMPOSE** complex tasks into parallel sub-tasks
2. **DELEGATE** via parallel `task` tool calls
3. **SYNTHESIZE** results

Rules in the prompt:
- Use subagents when 2+ independent parallel sub-tasks exist
- Do NOT use for single-step ops, sequential deps, or meta-conversation
- Max N task calls per response (default 3)

### 5.2 Subagent Lifecycle

```
Lead Agent LLM generates N task() tool calls
          │
          ▼
┌─────────────────────────────────┐
│  SubagentLimitMiddleware        │
│  Truncates to max 3 (clamped   │
│  [2,4]) task calls              │
└────────────┬────────────────────┘
             │
             ▼ (for each task call, in parallel)
┌─────────────────────────────────────────────────────────┐
│  task_tool()                                            │
│                                                         │
│  1. Look up SubagentConfig from registry                │
│  2. Append skills prompt to system prompt               │
│  3. Extract parent context (sandbox, thread_data)       │
│  4. Get tools with subagent_enabled=False (no nesting)  │
│  5. Create SubagentExecutor                             │
│  6. execute_async() → submit to thread pools            │
│  7. Poll every 5s, stream SSE events                    │
│  8. Return "Task Succeeded. Result: ..."                │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│  SubagentExecutor (dual thread pools)                   │
│                                                         │
│  _scheduler_pool (3 workers)                            │
│       │                                                 │
│       ▼                                                 │
│  _execution_pool (3 workers)                            │
│       │                                                 │
│       ▼                                                 │
│  asyncio.run(_aexecute())                               │
│       │                                                 │
│       ▼                                                 │
│  create_agent() with MINIMAL middlewares:                │
│    1. ThreadDataMiddleware(lazy_init=True)               │
│    2. SandboxMiddleware(lazy_init=True)                  │
│    3. OutputRepairMiddleware(max_retries=1)              │
│    4. LoopDetectionMiddleware(warn=2, hard=4)            │
│       │                                                 │
│       ▼                                                 │
│  agent.astream([HumanMessage(task)])                    │
│  → ReAct loop until completion or timeout               │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Subagent Types

| Property | general-purpose | bash |
|----------|----------------|------|
| **Tools** | All parent tools minus denylist | Only: bash, ls, read_file, write_file, str_replace |
| **Denied tools** | task, ask_clarification, present_files | task, ask_clarification, present_files |
| **Max turns** | 50 | 30 |
| **Timeout** | 900s (15 min) | 900s (15 min) |
| **Model** | inherit from parent | inherit from parent |
| **Prompt** | 15 lines: autonomous researcher | 15 lines: command execution specialist |

### 5.4 Lead Agent vs Subagent — What's Inherited

```
                          Lead Agent          Subagent
                          ──────────          ────────
System prompt             280+ lines          ~15 lines
  ├── <role>              ✓                   ✓ (minimal)
  ├── <soul>              ✓                   ✗
  ├── <memory>            ✓                   ✗
  ├── <thinking_style>    ✓                   ✗
  ├── <clarification>     ✓                   ✗
  ├── <skill_system>      ✓                   ✓ (appended)
  ├── <subagent_system>   ✓                   ✗
  ├── <response_style>    ✓                   ✗
  ├── <citations>         ✓                   ✗
  ├── <critical_reminders>✓                   ✗
  └── date/time           ✓                   ✗

Middlewares               14 (up to)          4
Thinking                  configurable        ALWAYS disabled
Conversation history      full thread         fresh (task prompt only)
Sandbox                   shared              shared (same sandbox_id)
Thread data               shared              shared (same paths)
Scratchpad                shared              shared (via tools)
Memory                    loaded at start     ✗ none
Recursion limit           config              max_turns (50 or 30)
```

**Key files:**
- `src/subagents/executor.py` — execution engine, thread pools, middleware chain
- `src/subagents/builtins/general_purpose.py` — general-purpose config + prompt
- `src/subagents/builtins/bash_agent.py` — bash config + prompt
- `src/tools/builtins/task_tool.py` — task tool (spawning + polling + SSE)
- `src/agents/middlewares/subagent_limit_middleware.py` — concurrency enforcement

---

## 6. Memory System

### 6.1 Memory Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY UPDATE PIPELINE                        │
│                                                                  │
│  Agent execution completes                                       │
│        │                                                         │
│        ▼                                                         │
│  MemoryMiddleware.after_agent()                                  │
│        │                                                         │
│        ├── Filter: keep only HumanMessage + final AIMessage      │
│        ├── Strip: remove <uploaded_files> blocks                 │
│        ├── Drop: upload-only turns                               │
│        │                                                         │
│        ▼                                                         │
│  MemoryUpdateQueue.add(thread_id, messages, agent_name)          │
│        │                                                         │
│        ├── Deduplication: replaces older entry for same thread   │
│        ├── Debounce: 30 second timer                             │
│        │                                                         │
│        ▼  (daemon thread)                                        │
│  MemoryUpdater.update_memory()                                   │
│        │                                                         │
│        ├── Load current memory from JSON                         │
│        ├── Format conversation (TRUNCATE messages to 1000 chars) │
│        ├── Build MEMORY_UPDATE_PROMPT                            │
│        ├── LLM call (thinking disabled)                          │
│        ├── Parse JSON response                                   │
│        ├── _apply_updates():                                     │
│        │     ├── Update user context sections                    │
│        │     ├── Update history sections                         │
│        │     ├── Add new facts (confidence >= 0.7)               │
│        │     ├── Remove flagged facts                            │
│        │     └── Cap at 100 facts (sorted by confidence)         │
│        ├── Strip upload mentions                                 │
│        └── Atomic save (.tmp → rename)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Memory Injection at Prompt Time

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY INJECTION                              │
│                                                                  │
│  make_lead_agent() → apply_prompt_template()                     │
│        │                                                         │
│        ▼                                                         │
│  _get_memory_context(agent_name)                                 │
│        │                                                         │
│        ▼                                                         │
│  format_memory_for_injection(memory_data, max_tokens=2000)       │
│        │                                                         │
│        ├── INJECTED:                                             │
│        │     ├── user.workContext.summary      → "Work: ..."     │
│        │     ├── user.personalContext.summary   → "Personal: ..."│
│        │     ├── user.topOfMind.summary         → "Focus: ..."   │
│        │     ├── history.recentMonths.summary   → "Recent: ..."  │
│        │     └── history.earlierContext.summary  → "Earlier: ..." │
│        │                                                         │
│        ├── ⚠️ NOT INJECTED:                                      │
│        │     ├── history.longTermBackground     ← STORED BUT     │
│        │     ├── facts[] array                  ← NEVER READ     │
│        │     └── timestamps                     ← BACK           │
│        │                                                         │
│        ▼                                                         │
│  Wrapped in <memory>...</memory> tags                            │
│  Token-limited via tiktoken (2000 tokens)                        │
│  Interpolated into system prompt at {memory_context}             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Memory File Format

```json
{
  "version": "1.0",
  "lastUpdated": "ISO-8601",
  "user": {
    "workContext":     {"summary": "...", "updatedAt": "..."},   ← INJECTED
    "personalContext": {"summary": "...", "updatedAt": "..."},   ← INJECTED
    "topOfMind":       {"summary": "...", "updatedAt": "..."}    ← INJECTED
  },
  "history": {
    "recentMonths":      {"summary": "...", "updatedAt": "..."},  ← INJECTED
    "earlierContext":     {"summary": "...", "updatedAt": "..."},  ← INJECTED
    "longTermBackground": {"summary": "...", "updatedAt": "..."}  ← ⚠️ NOT INJECTED
  },
  "facts": [                                                       ← ⚠️ NOT INJECTED
    {"id": "fact_xxx", "content": "...", "category": "...",
     "confidence": 0.9, "createdAt": "...", "source": "thread_id"}
  ]
}
```

**Key files:**
- `src/agents/memory/prompt.py` — injection + update prompts
- `src/agents/memory/updater.py` — update pipeline, file I/O
- `src/agents/memory/queue.py` — debounced async queue
- `src/agents/middlewares/memory_middleware.py` — conversation filtering

---

## 7. Tool System

### 7.1 Tool Assembly Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  get_available_tools(groups, include_mcp, model_name,           │
│                      subagent_enabled)                           │
│        │                                                         │
│        ├── SOURCE 1: Config tools (config.yaml)                  │
│        │     Resolved via importlib: "src.sandbox.tools:bash_tool│
│        │     Filtered by tool_groups if specified                │
│        │                                                         │
│        ├── SOURCE 2: Built-in tools (hardcoded)                  │
│        │     ALWAYS: present_files, ask_clarification,           │
│        │             save_finding, read_findings, scratchpad_stats│
│        │     IF subagent_enabled: + task                         │
│        │     IF supports_vision:  + view_image                   │
│        │     IF bootstrap:        + setup_agent                  │
│        │                                                         │
│        ├── SOURCE 3: MCP tools (extensions_config.json)          │
│        │     Lazily loaded, mtime-cached                         │
│        │                                                         │
│        └── RESULT: loaded_tools + builtin_tools + mcp_tools      │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Complete Tool Inventory

**Always available (BUILTIN):**

| Tool | Purpose |
|------|---------|
| `present_files` | Make output files visible to user |
| `ask_clarification` | Interrupt to ask user question (via ClarificationMiddleware) |
| `save_finding` | Write to per-thread shared scratchpad |
| `read_findings` | Read from shared scratchpad with filters |
| `scratchpad_stats` | Scratchpad statistics |

**Config-loaded (sandbox + community):**

| Tool | Purpose |
|------|---------|
| `bash` | Execute shell commands in sandbox |
| `ls` | List directory (2 levels deep) |
| `read_file` | Read file contents (optional line range) |
| `write_file` | Write/append to file |
| `str_replace` | Replace substring in file |
| `web_search` | Search web (Tavily/SearxNG/Firecrawl/InfoQuest) |
| `web_fetch` | Fetch page content (Tavily/Jina/SearxNG/Firecrawl/InfoQuest) |
| `image_search` | Search images (DuckDuckGo) |

**Conditional:**

| Tool | Condition |
|------|-----------|
| `view_image` | model.supports_vision = True |
| `task` | subagent_enabled = True |
| `setup_agent` | bootstrap mode only |
| `write_todos` | plan mode only (injected by TodoListMiddleware) |

### 7.3 Subagent Tool Filtering

```
Lead Agent tool set
       │
       ▼
get_available_tools(subagent_enabled=False)   ← removes task tool
       │
       ▼
_filter_tools(tools, config.tools, config.disallowed_tools)
       │
       ├── general-purpose: tools=None (all), deny=[task, ask_clarification, present_files]
       │     → gets everything except those 3
       │
       └── bash: tools=[bash,ls,read_file,write_file,str_replace], deny=[...]
             → gets only 5 sandbox tools
```

**Key files:**
- `src/tools/tools.py` — `get_available_tools()`, tool groups
- `src/sandbox/tools.py` — 5 sandbox tools
- `src/tools/builtins/` — built-in tools
- `src/community/` — web_search, web_fetch, image_search providers
- `src/mcp/` — MCP tool loading and caching

---

## 8. Skills System

Skills are NOT tools. They are **prompt-injected documents** the agent reads on demand.

```
┌─────────────────────────────────────────────────────────────────┐
│  Skills Loading (src/skills/loader.py)                          │
│                                                                  │
│  Scan deer-flow/skills/{public,custom}/                          │
│       │                                                          │
│       ▼                                                          │
│  Find SKILL.md files with YAML frontmatter                       │
│       │                                                          │
│       ▼                                                          │
│  Check enabled state in extensions_config.json                   │
│       │                                                          │
│       ▼                                                          │
│  get_skills_prompt_section()                                     │
│       │                                                          │
│       ▼                                                          │
│  <skill_system> XML block in system prompt                       │
│  Lists each skill name + description + container path            │
│       │                                                          │
│       ▼                                                          │
│  Agent uses read_file to load SKILL.md when task matches         │
│  (progressive loading — not preloaded)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Full Request Lifecycle

```
USER MESSAGE
    │
    ▼
[Entry: LangGraph Server / DeerFlowClient / IM Channel]
    │
    ▼
[Checkpointer loads existing thread state]
    │
    ▼
══ BEFORE_AGENT ══════════════════════════════
 1. ThreadDataMiddleware  → set thread paths
 2. UploadsMiddleware     → inject <uploaded_files>
 3. SandboxMiddleware     → acquire sandbox (lazy)
    │
    ▼
══ AGENT NODE (iteration 1) ══════════════════
    │
    ├── before_model:
    │   ViewImageMiddleware → inject image if ready
    │
    ├── wrap_model_call:
    │   DanglingToolCallMiddleware → patch missing ToolMessages
    │   OutputRepairMiddleware     → retry on malformed output
    │   SelfEvaluationMiddleware   → evaluate + revise if needed
    │
    ├── [LLM CALL]
    │
    ├── after_model:
    │   SubagentLimitMiddleware    → truncate excess task calls
    │   LoopDetectionMiddleware    → detect repetition
    │
    └── aafter_model:
        TitleMiddleware            → generate title (first exchange only)
    │
    ▼
[DECISION: tool_calls?]
    │
    ├── YES ─────────────────────────────────
    │   │
    │   ▼
    │   ══ TOOLS NODE ═══════════════════════
    │   │
    │   ├── wrap_tool_call:
    │   │   ClarificationMiddleware
    │   │     → if ask_clarification: Command(goto=END) [INTERRUPT]
    │   │     → else: pass through to handler
    │   │
    │   ├── Execute tools:
    │   │   ├── bash/ls/read_file/write_file/str_replace → Sandbox
    │   │   ├── web_search/web_fetch → External API
    │   │   ├── task → SubagentExecutor (async, polls 5s)
    │   │   │         ├── scheduler_pool (3 workers)
    │   │   │         ├── execution_pool (3 workers)
    │   │   │         └── Subagent ReAct loop (4 middlewares, 50/30 turns)
    │   │   ├── view_image → base64 encode, update state
    │   │   ├── save_finding/read_findings → Scratchpad (fcntl-locked JSON)
    │   │   └── MCP tools → External MCP servers
    │   │
    │   └── ToolMessages added to state
    │       │
    │       └── → back to AGENT NODE (next iteration)
    │
    └── NO ──────────────────────────────────
        │
        ▼
    ══ AFTER_AGENT ══════════════════════════
     MemoryMiddleware → filter messages → queue for async update
        │
        ▼
    [END — response returned to caller]
```

---

## 10. Configuration Reference

| Config | File | Key Settings |
|--------|------|-------------|
| Graph registration | `langgraph.json` | `graphs.lead_agent`, `checkpointer.path` |
| App config | `config.yaml` | models, tools, tool_groups, sandbox, skills, subagents |
| Extensions | `extensions_config.json` | MCP servers, skill enable/disable |
| Model factory | `src/models/factory.py` | `resolve_class()`, thinking settings |
| Subagent config | `src/subagents/builtins/*.py` | type, tools, max_turns, timeout |
| Subagent overrides | `src/config/subagents_config.py` | config.yaml timeout overrides |
| Memory config | `src/config/memory_config.py` | enabled, injection, model, thresholds |
| Paths | `src/config/paths.py` | thread dirs, memory files, uploads/outputs |

---

*Generated by Opus forensic analysis of DeerFlow backend source code, 2026-03-10.*
