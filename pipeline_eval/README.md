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
| 7 | Phase-2 tool loop **gives up** after a byte-identical failed output (greedy decoding: natural_calculation 0/6 in v4) | One resample (T 0.7, fixed seed) before giving up | phase2 end_to_end (natural_calculation) |

Not fixable without retraining (documented, not attempted): phase-1 errors on CAPI age bands and
`residence_fail` (0/20 in every model), and the rule-sheet v2 drop without retraining (job 030
of `feat/night-improvements` failed with an OOM).

Run order on the DGX (one session, GPU free): runner on `eval/pipeline-fixes` executes
300 (baseline API from a worktree), 305 (RAG tuning), 310 (fixes API) and 320 (`delta.md`).
