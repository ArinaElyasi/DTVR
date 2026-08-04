# Build Specification — DTV Requirements Agent (DTV-REA)

**Audience:** the implementing engineer/agent.
**Status:** Authoritative blueprint. Every design decision below is grounded in the project's Step 1 (input derivation) and Step 2 (architecture) documents; where those were silent, decisions are made here and justified inline.
**Language:** Python 3.11+. **LLM:** Llama 3.3 70B served via Groq. **Orchestration:** LangGraph (evaluation below).

---

## 0. What this agent is

An interviewing agent that elicits **digital-twin requirements** from a single stakeholder by conversation, and produces a structured requirements document aligned with the DTV framework (Bitencourt et al., 2025, *Int. J. Production Research*, DOI 10.1080/00207543.2025.2524516). It operationalizes DTV steps 1–2 only (Problem Definition & Conceptual Design → DT Requirements). Standards alignment (IEEE 29148) is **out of scope** by prior decision.

**The governing rule of the whole architecture:**

> **Decisions in code. Language in the model.**
> The LLM does exactly two jobs: phrase questions, and structure answers. Every decision — what to ask next, whether an extraction is acceptable, whether the session is finished — is deterministic Python.

**The central research claim the implementation must make provable:**

> The agent **never invents a number**. Every design goal (tolerance, threshold, accuracy, percentage) must trace to the exact conversation turn where the stakeholder said it. This is enforced by a code-level validator, not by prompt instructions. Target metric: fabricated-goal rate = **0** (zero, not "low").

Domain grounding: the reverse-engineered FDM 3D-printer case study (12 requirements, UR1.1–UR1.12) is the reference scenario and first ground-truth test case. Two real defects found in the published paper — UR1.11's unverifiable "minimize", and the UR1.2/UR1.7 duplicate obligation — are mandatory validator test cases.

---

## 1. Agent orchestration and reasoning core

### 1.1 Framework evaluation: LangGraph vs. LangChain — commit to **LangGraph**

| Requirement of this agent | LangChain (chains/LCEL) | LangGraph |
|---|---|---|
| **Cyclic control flow** — the seven-step turn cycle repeats until a code-computed termination condition | Chains are linear/DAG; cycles require awkward workarounds | Cycles are the core primitive (graph with conditional edges) |
| **Explicit, typed, persistent state** — one running `SessionState` object is the agent's memory; requirements are never re-derived from the transcript | State is implicit in message history / memory classes | First-class typed state (`TypedDict`/Pydantic), reducers, checkpointing |
| **Conditional branching in code** — "next gap?", "validator verdict?", "complete?" are Python decisions routing the flow | Router chains exist but are LLM-centric | `add_conditional_edges` with plain Python predicates — exactly "decisions in code" |
| **Human-in-the-loop pauses** — the stakeholder answers every question; flags need human resolution | No native pause/resume | `interrupt()` + checkpointer gives durable pause/resume at any node |
| **Turn-by-turn audit trail** — the snapshot log is the evaluation dataset | Manual | Checkpointer persists every state transition for free |

**Decision: LangGraph.** This agent *is* a state machine with cycles, code-gated transitions, and human interrupts — LangGraph's exact shape. LangChain is used only incidentally (the `langchain-groq` chat model client); no chains, no LangChain memory, no LangChain agents.

**Tradeoff accepted:** LangGraph adds a dependency and a learning curve over a hand-rolled `while` loop (the existing prototype). We take it for durable checkpointing, native interrupts, and a graph structure that maps 1:1 to the documented architecture — the hand-rolled loop's logic ports directly into nodes.

### 1.2 Graph topology

State machine. Model-touching nodes are marked **[LLM]**; everything else is pure Python.

```
START
  └─▶ capture_purpose            [LLM: P2 phrases the opener; interrupt→stakeholder]
  └─▶ propose_maturity           [LLM: P3 maps purpose→4R + plain description]
  └─▶ confirm_maturity           [interrupt→stakeholder confirms/overrides]  ← HITL-1
  └─▶ decide_gap                 [code: checklist lookup]
        ├─ gap found ──────────▶ generate_question   [LLM: P4]
        │                         └─▶ await_answer   [interrupt→stakeholder] ← HITL-2
        │                         └─▶ extract        [LLM: P5, JSON mode]
        │                         └─▶ validate       [code: V1–V5]
        │                         └─▶ commit_or_flag [code]
        │                               └─▶ decide_gap          (the cycle)
        ├─ no gap, open flags ──▶ resolve_flags      [LLM: P6 phrases; interrupt] ← HITL-3
        │                         └─▶ decide_gap
        └─ complete ───────────▶ finalize            [code: write outputs]
END
```

**Node contracts (implement exactly):**

| Node | Type | Input → Output | Rules |
|---|---|---|---|
| `capture_purpose` | LLM + interrupt | ∅ → `state.purpose` | One open question (P2). Store purpose `{statement, rationale, user_roles, context_of_use}`. |
| `propose_maturity` | LLM | purpose → proposal | P3. Must output the 4R level **and** a plain-language description of what the twin will/won't do. Never the label alone. |
| `confirm_maturity` | interrupt | proposal → `state.maturity_level` | Record `{value, provenance:"agent_proposed_human_confirmed", agent_reasoning, description_shown_to_human, human_response: confirmed|overridden, overridden_to}`. |
| `decide_gap` | code | state → route | First dimension with status `uncovered`, in fixed order. Enforce attempt cap (see §1.4/Q3): if attempts[gap] > 3 → mark dimension `parked`, continue. Route: `ask` / `resolve_flags` / `finalize`. |
| `generate_question` | LLM | gap + state → one question | P4. Exactly one question, plain language, no framework vocabulary. If the last committed performance requirement lacks a goal, the number-question takes priority over a new topic. |
| `await_answer` | interrupt | question → turn appended | Append stakeholder turn with index. |
| `extract` | LLM | answer turn + gap + state → `list[Requirement]` | P5, Groq JSON mode, temperature 0, validated against the Pydantic schema. **Schema-parse failure → retry extraction up to 2× (parse retry is allowed; fabrication retry is NOT — see validator).** |
| `validate` | code | candidates → verdicts | Run V1–V5 per candidate (§1.3). Deterministic; no model call. |
| `commit_or_flag` | code | verdicts → state | Hard-fail → reject candidate, append flag, **do not regenerate** (route back so the *human* is asked for the value via P4 rule 3). Soft flags → commit + append flag. Snapshot after every turn. |
| `resolve_flags` | LLM + interrupt | open flag → resolution | P6 phrases the problem + one resolving question; human answer updates/merges/parks requirement, flag → resolved. Duplicates: human merges; agent never auto-merges. |
| `finalize` | code | state → files | Write `session_<id>.json` (full state), `requirements.md` (human-readable), snapshot log. If terminated early (see Q2), mark output `"status": "partial"` prominently. |

### 1.3 The validator (port from prototype `validator.py` — behavior is contractual)

Two severities. **HARD** blocks commit; **FLAG** commits but blocks *termination* until a human resolves.

| ID | Check | Severity | Rule |
|---|---|---|---|
| V1 | **Fabrication** | HARD | A `designGoal` must carry `stakeholder_utterance_ref` → a real turn → whose `role == "stakeholder"` → whose text contains every numeric token of the goal (normalized: `90%`≈`90`, `±2.5`≈`2.5`). Any failure ⇒ `fabricated_goal`, reject. Resolution is always "ask the human"; **never regenerate** — a regenerated number is still an invented number. |
| V2 | Vague predicate | FLAG | Requirement text contains a term from the vague-word list (minimize, maximize, optimize, improve, reduce, reliable, fast, accurate, sufficient, …) **and** `designGoal is None` ⇒ `unverifiable_predicate` (the UR1.11 case). Documented limitation: keyword heuristic, first pass. |
| V3 | Duplicate obligation | FLAG | Same dimension, content-word overlap ≥ 0.6 after stop-word removal + light stemming (`collected`→`collect`) ⇒ `duplicate_obligation` (the UR1.2/UR1.7 case). Human merges. |
| V4 | Orphan goal | HARD | Goal attached to a nonexistent requirement id ⇒ reject. |
| V5 | Maturity consistency | FLAG | Requirement lands in a dimension marked `not_applicable` ⇒ `maturity_inconsistency`; surface which statement stands. |

### 1.4 State model (Pydantic; LangGraph state wraps it)

Port the prototype's `SessionState`, `Requirement`, `Flag`, `Turn` to Pydantic v2. Key fields (unchanged semantics):

- `coverage_status: dict[dim → covered|uncovered|not_applicable]` over the four DTV dimensions, fixed order: `data_collection_integration`, `virtual_environment`, `intelligence_layer`, `automation_feedback`. `not_applicable` requires a **confirming stakeholder turn index** (or `-1` when *parked* by the attempt cap — record `parked_reason`). The maturity level never sets `not_applicable`; only the stakeholder does (the FDM case is Replication yet pauses the machine — levels don't partition dimensions).
- `Requirement`: `id (UR1.<n> sequential)`, `text ("The DT shall …", solution-free)`, `type functional|performance`, `dimension`, `verifyMethod` + `verifyMethod_provenance:"agent_proposed_human_confirmed"` (pattern: functional→Inspection, performance→Test; human confirms at finalize), `designGoal|null`, `designGoal_provenance:"stated"|null` (**no `agent_derived` option exists — the schema enforces the research claim structurally**), `stakeholder_utterance_ref`, `priority`, `rationale`, `status complete|needs_clarification`.
- Snapshots after every turn: `{turn, n_requirements, n_open_flags, coverage}` — this log **is** the evaluation dataset.

**Locked edge-case rules** (supervisor-approved): **Q1** "I don't know" to a number → `status:"needs_clarification"`, `designGoal:null`, move on. **Q2** early stop → emit partial output, clearly marked. **Q3** attempt cap N=3 per dimension → park and continue.

**Termination** (`decide_gap` routes to `finalize`) iff: every dimension `covered`/`not_applicable`/parked **and** no committed performance requirement silently lacks a goal (parked `needs_clarification` is acceptable) **and** no open flags.

### 1.5 LLM configuration (Groq)

- Model: `llama-3.3-70b-versatile` via `langchain_groq.ChatGroq`. API key from `GROQ_API_KEY` env var.
- `extract` (P5): `temperature=0`, JSON mode (`response_format={"type":"json_object"}`), output parsed into Pydantic; on parse error retry ≤2 with the error appended, then park the answer as a `needs_clarification` note rather than guessing.
- `generate_question`/`propose_maturity`/`resolve_flags` (P4/P3/P6): `temperature=0.3` (mild variation in phrasing; content is constrained by prompt rules).
- Rate-limit/backoff: `tenacity` exponential retry on 429/5xx, max 5 attempts. All calls logged (prompt hash, latency, tokens) to the run log.

---

## 2. Backend tools, data, and integrations

**Tools the agent calls:** none external. This agent's "tools" are internal pure functions — the checklist (`decide_gap`), the validator (V1–V5), the state store, and the output writer. No web, no databases, no third-party APIs beyond Groq. This is deliberate: the deterministic core must be auditable and testable offline (it already runs against a scripted model with zero network access).

**Integrations:**
- **Groq** (LLM inference) — the only network dependency.
- **Checkpointer:** `SqliteSaver` (`langgraph-checkpoint-sqlite`) at `./runs/checkpoints.db` — durable pause/resume across process restarts (a stakeholder can leave mid-interview and return).
- **Filesystem outputs** per session under `./runs/<session_id>/`: `session.json`, `requirements.md`, `snapshots.jsonl`, `llm_calls.jsonl`.
- **Interface:** CLI first (`python -m dtv_rea.cli --session <id>`), reading stakeholder answers from stdin at interrupts. The graph is UI-agnostic; a thin web UI can wrap the same interrupts later without touching the graph.

**Dataset decision — hybrid, and here is why:**

1. **Real ground truth (primary):** the reverse-engineered FDM case from the published paper — the expected conversation mapped to the 12 known requirements UR1.1–UR1.12 with their exact design goals (±2.5 °C, ±10%, ≥90%, ≥90%, ≤10%, 100%). This is *real published data*, not invented; it anchors extraction-fidelity evaluation to something citable. Ship it as `data/ground_truth/fdm.json` (expected requirements) + `data/personas/fdm_stakeholder.json` (scripted answers).
2. **Synthetic stakeholder personas (secondary):** scripted answer sets that simulate interview dynamics the real case can't exercise — an evasive stakeholder (triggers Q3 cap), an "I don't know" stakeholder (Q1), a contradictory one (V5), one who volunteers no numbers (V1 pressure), one who quits early (Q2). Synthetic is correct here because *no public corpus of DT requirements interviews exists*, and personas give reproducible, targeted coverage of edge paths.
3. **Known limitation to carry into the paper:** n=1 real scenario. The build must make adding ground-truth cases trivial (drop a new `<case>.json` pair in `data/` — no code changes), because 2–3 further reverse-engineered DT cases are the planned fix.

---

## 3. Retrieval / knowledge grounding

**Decision: no RAG. Full context-stuffing.** (This is a prior, documented architecture decision — uphold it.)

The knowledge base is small and fixed: DTV §3.1–3.2 (input specification), the four 4R level definitions, the four requirement dimensions, the three-layer conceptual pattern, and the worked FDM exemplar (Figures 3–4). ≈10 pages total against Llama 3.3's 128k context. It lives verbatim inside P1 (system prompt) in `prompts.py`.

**Why not a vector store:** retrieval adds a *silent* failure mode — if the 4R definitions fail to retrieve on the turn the agent proposes a maturity level, the proposal is unguided and the output still looks normal. For a corpus this size, retrieval is risk with zero benefit. **Revisit trigger (state in README):** adopt retrieval only if the knowledge base outgrows the context window (e.g., many case exemplars or standards documents are added later).

No structured-data lookup either: the only "data" the agent grounds in is the conversation itself — which is the point. Grounding = the utterance-reference discipline enforced by V1.

---

## 4. Human-in-the-loop

**Determination: HITL is not an add-on here — it is the product.** The stakeholder is the *only* legitimate source of purposes, tolerances and authorizations (Step 1's load-bearing finding: every design goal in the reference case was stakeholder-authorized; none was derivable). Three mandatory checkpoints, implemented as LangGraph `interrupt()` points with the SQLite checkpointer:

1. **HITL-1 — maturity confirmation** (`confirm_maturity`): the agent proposes a 4R level *with a plain-language description*; the human confirms or overrides. The description, response, and any override are recorded — a bare "OK" to an unexplained label would be rubber-stamping, and the record lets us detect that in evaluation.
2. **HITL-2 — every interview answer** (`await_answer`): each loop iteration pauses for the stakeholder. Includes fabrication recovery: when V1 rejects a goal, the *next* question asks the human for the value (P4 rule 3) — the model is never re-rolled for a number.
3. **HITL-3 — flag resolution** (`resolve_flags`): open flags (vague wording, duplicates, inconsistencies) block termination until the human resolves them. Duplicates are merged only by the human. Final `verifyMethod` confirmations happen here too, batched before `finalize`.

There is deliberately **no** autonomous path around any of these: an agent that completes a requirements document with no human contact has, by this project's definition, fabricated it.

---

## 5. Evaluation and metrics

Harness: `eval/run_eval.py` replays each persona through the real graph with the real Groq model (and a `--stub` mode with the scripted model for CI). Metrics computed from `session.json` + `snapshots.jsonl` + ground truth.

| Metric | Definition | Measurement | Target |
|---|---|---|---|
| **Fabricated-goal rate** | committed design goals with no valid stakeholder utterance ref | audit every committed goal against transcript (independent re-check, not the validator's own verdict) | **0** — headline claim; any nonzero is the headline negative result |
| Validator interception rate | fabricated candidates produced by the LLM that V1 caught pre-commit | rejected/(rejected+slipped) on the no-numbers persona | 100% caught; report LLM fabrication-attempt rate separately (it measures the model; interception measures us) |
| Coverage completeness | dimensions resolved (covered/n-a) at session end | final `coverage_status` | 4/4 on cooperative personas |
| Extraction fidelity (FDM) | committed requirements matching ground truth | per-requirement match on {text semantic match (manual/rubric), type, dimension, goal exact, ref correct} vs `fdm.json` | ≥ 10/12; goal values exact-match 100% of those present |
| Defect detection | the two seeded paper defects flagged | UR1.11 → `unverifiable_predicate`; UR1.2/1.7 → `duplicate_obligation` | 2/2 |
| Elicitation efficiency | stakeholder turns to completion | `snapshots.jsonl` | ≤ 12 turns (FDM persona); report distribution |
| Maturity proposal quality | correct 4R proposal + description shown | vs. case's known level; `description_shown_to_human` non-empty | correct on FDM (Replication); 100% description presence |
| Requirement well-formedness | fields populated, "shall" form, solution-free (no tool/sensor names) | schema check + banned-solution-word lint | ≥ 95% |
| Edge-rule conformance | Q1 parks, Q2 partial-marked, Q3 caps at 3 | targeted personas | 3/3 behaviors exact |
| Robustness/ops | JSON-parse retry rate, latency p50/p95, tokens/session | `llm_calls.jsonl` | report; parse-retry < 10% of extractions |

Requirement-quality scoring (semantic text match, clarity) uses a small human rubric adapted from Ronanki et al. (2023) rather than an invented one. Statistical claims wait for n≥3 ground-truth cases — say so in the eval README; do not overclaim from n=1.

---

## 6. Step-by-step build logic (implementation guide)

Build in this order; each phase ends green before the next starts. An existing prototype (`dtv_agent/`: `state.py`, `validator.py`, `loop.py`, `prompts.py`, 16 passing tests, scripted demo) is the porting source for phases 1–2 — reuse its logic and test cases verbatim where possible.

**Phase 0 — Scaffold.**
`dtv_rea/` package; `pyproject.toml`; deps: `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-groq`, `pydantic>=2`, `tenacity`, `pytest`. Repo layout: `dtv_rea/{state.py, validator.py, prompts.py, graph.py, llm.py, cli.py}`, `data/{ground_truth/, personas/}`, `eval/`, `tests/`. `.env` handling for `GROQ_API_KEY`.

**Phase 1 — State model (pure, tested).**
Pydantic models: `Requirement`, `Flag`, `Turn`, `SessionState` (+ `coverage_status`, `not_applicable_confirmations`, `snapshots`, attempt counters). Methods: `add_turn`, `commit_requirement`, `mark_not_applicable(dim, confirming_turn)`, `next_gap()`, `open_flags()`, `is_complete()`, `snapshot()`. **Constraint to encode in the schema itself:** `designGoal_provenance` is `Literal["stated"] | None` — no other value can exist. Tests: checklist ordering, three coverage states, termination logic (flags block completion).

**Phase 2 — Validator (pure, tested).**
Port V1–V5 exactly (numeric-token normalization; stop-words + light stemming; 0.6 overlap threshold). Tests must include, verbatim: goal-with-valid-ref passes; goal-with-no-ref rejected; number-absent-from-cited-turn rejected; ref-to-agent-turn rejected; **UR1.11 "minimize" flagged**; **UR1.2/UR1.7 duplicate flagged**; requirement-in-n/a-dimension flagged. All validator tests run with zero network.

**Phase 3 — Prompts.**
`prompts.py` with P1–P6 as module constants, knowledge base embedded in P1 (4R definitions, four dimensions, DTV §3.1–3.2 summary, FDM exemplar as few-shot for P5). P5 must instruct JSON-only output matching the Pydantic schema, `designGoal:null` when no number was said, `status:"needs_clarification"` on "I don't know", and the utterance-ref citation duty.

**Phase 4 — Graph with stub model.**
`graph.py`: `StateGraph` with the §1.2 topology, conditional edges as plain functions, `interrupt()` at HITL-1/2/3, `SqliteSaver`. `llm.py` defines the `ModelPort` protocol (`propose_maturity`, `ask`, `extract`, `phrase_flag`); implement `StubModel` (scripted) first. Integration tests: full FDM persona run end-to-end on the stub — commits, one seeded fabricated candidate rejected, both paper defects flagged, termination blocked until flags resolved, snapshots accumulate. This proves the graph before any API call exists.

**Phase 5 — Groq model port.**
`GroqModel` implementing `ModelPort` per §1.5 (JSON mode, temp 0 extraction, parse-retry ≤2 then park, tenacity backoff, call logging). Unit-test the JSON-repair path with canned malformed outputs. Then run the FDM persona once live and diff against the stub run.

**Phase 6 — CLI + finalize.**
`cli.py`: create/resume session by id (checkpointer), print agent turns, read stakeholder input at interrupts, `--persona <file>` to auto-answer from a script. `finalize` writes `session.json`, `requirements.md` (grouped by dimension, goals with their utterance quotes, open/parked items marked), `snapshots.jsonl`. Early-exit (Ctrl-D / "stop") → partial output, `"status":"partial"` at top of both files (Q2).

**Phase 7 — Personas + evaluation.**
Author the five personas (§2) + FDM ground truth files. `eval/run_eval.py` computes every §5 metric into `eval/report.md` (one table per persona + aggregate); `--stub` for CI. Wire `pytest` + stub-eval into CI so the deterministic core and edge rules are regression-guarded.

**Phase 8 — Hardening + docs.**
README: run instructions, architecture-decision record (LangGraph rationale, no-RAG with revisit trigger, hybrid dataset, HITL placement), honest-limitations section (V2 keyword heuristic, V3 threshold tuned to the known case, single stakeholder, n=1). Config knobs in one `settings.py`: model name, temperatures, attempt cap N (default 3), duplicate threshold (0.6), vague-word list.

**Definition of done:** stub-mode eval green on all personas; live FDM run achieves fabricated-goal rate 0, both defects caught, ≥10/12 fidelity; a fresh clone runs the demo with only `GROQ_API_KEY` set.

---

*Prior-work anchors for the implementer: Step 1 derivation document (input taxonomy, schema, fabrication finding), Step 2 architecture document (loop, checklist, validator, no-RAG, memory), prototype `dtv_agent/` (porting source). Framework source: Bitencourt et al. 2025, DOI 10.1080/00207543.2025.2524516.*
