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
