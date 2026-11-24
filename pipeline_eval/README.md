# pipeline_eval — scorecard of the whole pipeline through the API

`scorecard.py` sends the requests a grader would send to a running API and scores each phase
with that phase's own metrics (verifier pass@1, phase-2 workflow metrics, Recall@5/MRR and
citations on the phase-3 gold set, agent bench per brain, two-turn memory), plus HTTP errors and
requests slower than 300 s (ngrok's limit).

- `run_api_scorecard.sh <code_dir> <out_dir> <label>` starts the API from `code_dir` on port
  8899, runs the scorecard from the current checkout (same benches for every variant) and stops
  the API. Needs the GPU alone.
- Branch `eval/pipeline-baseline`: job 300 measures the deployed state.
- Branch `eval/pipeline-fixes`: fixes for the problems the baseline shows, the same scorecard on
  both branches in one DGX session, and the delta table.

## Problems found and fixes (branch `eval/pipeline-fixes`)

| # | Problem (where it was seen) | Fix | Scorecard metric that should move |
|---|---|---|---|
| 1 | `/rag` serves the **repealed** RDL 20/2020 (3rd hit in the first GPU query, 2026-11-24) | Repealed documents (`in_force: false` in `MANIFEST.json`) stay indexed but are never served; `ARCA_RAG_INCLUDE_REPEALED=1` restores them | phase3 recall@5 / cited_relevant |
| 2 | RAG chunking, λ and abstention gate are **provisional** (800/100, 0.5, gate off) | Job 305 sweeps λ on two recursive chunkings and calibrates the gate; `tune_rag.py` picks them and job 310 serves with them | phase3 mrr, unanswerable_abstained, false_abstentions |
| 3 | Agent **invents figures** it never retrieved (`rlm`: "15 días", real answer 30; first GPU runs) | A `final_answer` with figures that appear in no observation nor in the task gets one verification step | phase4 success (rlm above all) |
| 4 | Agent answers from the knowledge base **without citing** (all three brains) | The ids of the consulted passages are appended when the answer cites none | — (trace quality; graded by hand) |
| 5 | Agent can run past **ngrok's 300 s** (thinking brain, 2048 tokens per step) | 240 s time budget (`ARCA_AGENT_TIME_BUDGET`): stop and report what was obtained | phase4 over_300s, http_errors |
| 6 | Plain-text agent answers start with the label "Pensamiento:" (thinking brain) | Label stripped | — |
| 8 | The full API takes **11.4 GB of the 16 GB slice**: `/rag` loads a private copy of the same Qwen3-1.7B that `/tools` holds (baseline scorecard, OOM traces) | The RAG generator takes its weights from the shared model pool | phase3/phase4 http_errors (OOM) |
| 9 | **First request 341 s**, second 122 s, with the weights already loaded (baseline phase 1): the grader's first call would get ngrok's HTML 504 | `arca-api` warms every phase up in the background at start-up (`ARCA_WARMUP=0` disables) | phase1 over_300s, p90_s |
| 10 | `/reasoning` samples (T 0.6) **without a seed**: identical phase-1 code scored pass@1 0.625 (job 300) and 0.475 (job 310) on the same 40 problems | Sampling seeded per question (`ARCA_REASONING_SEED`, default 0): same question, same answer | reproducibility (compare's 2σ column) |
| 11 | `search_knowledge_base` **cut every passage to 900 characters**; the tuned chunking is 1200, so the sentence that answers could be cut off | Passages up to 1600 characters (whole chunks) | phase4 success (KB tasks) |
| 7 | Phase-2 tool loop **gives up** after a byte-identical failed output (greedy decoding: natural_calculation 0/6 in v4) | One resample (T 0.7, fixed seed) before giving up | phase2 end_to_end (natural_calculation) |

Not fixable without retraining (documented, not attempted): phase-1 errors on CAPI age bands and
`residence_fail` (0/20 in every model), and the rule-sheet v2 drop without retraining (job 030
of `feat/night-improvements` failed with an OOM).

Run order on the DGX (one session, GPU free): runner on `eval/pipeline-fixes` executes
300 (baseline API from a worktree), 305 (RAG tuning), 310 (fixes API) and 320 (`delta.md`).

## Baseline scorecard (jobs 300 + 302, 2026-11-24, `eval/pipeline-baseline`)

Phases 1-2 from job 300; phases 3-4 from job 302 (job 300's ran with a demo API on the same 16 GB
slice and died with CUDA OOM). Job 302 started with 16.3 GB free and had 0 HTTP errors.

| phase | result |
|---|---|
| health | 4/4 phases ready |
| 1 `/reasoning` (40 test problems) | pass@1 **0.625**, format 0.975, truncated 0.025; p50 21 s; **1 request over 300 s** (the first, 341 s) |
| 2 `/tools` (40 cases) | selection 0.95, arguments 0.85, sequence 0.90, **end to end 0.725**, amounts 0.81; weakest: missing_facts 0.40, natural_calculation 0.50, bdns 0.62 |
| 3 `/rag` (50 gold + 8 unanswerable) | Recall@5 **0.70**, MRR 0.455, relevant chunk cited 0.42, citations exist 1.00, false abstentions 0.12, unanswerable abstained 7/8 |
| 4 `/agent` (20 chained + 3 impossible) | success, re-scored with the corrected checks: **base 0.60, rlm 0.45, thinking 0.55**; impossible: base **0/3**, rlm 3/3, thinking 3/3; tool chain complete: base 0.65, rlm 0.75, thinking 0.35; ~3 steps and ~15 800 tokens per task; 0 requests over 300 s |
| memory | 1/2 two-turn conversations (the failed one remembered the fact but did not compute 23 - 19) |

Reading:
- **Agent checks were too strict.** Right answers were scored as wrong: "es ineligible", "seis
  meses" in words, IMV and CAPI given separately. Raw success was base 0.45 / rlm 0.40 / thinking
  0.40. The checks are corrected and `compare.py` re-scores both variants with them.
- **The base brain never says it cannot.** It answers the three impossible tasks (it explains how
  to apply instead of saying it cannot apply, and gives the IMV amount when asked about another
  benefit). The trained (rlm) and the thinking brains abstain on all three. Trained reasoning
  helps exactly where the course asks for it: saying so instead of inventing.
- **Real agent failures** in all three brains: T19 says "compatible" where the law says
  incompatible, T10 computes the second case wrong, T15 omits the asset limit, and the prose tasks
  T13/T14 fail (extraction, as in phase 2).
