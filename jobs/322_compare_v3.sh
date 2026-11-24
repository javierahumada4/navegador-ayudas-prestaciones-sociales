# est-minutes: 2
# Delta tables, every phase-4 answer re-scored with the corrected bench: baseline -> fixes v2
# (job 311) and baseline -> fixes v3 (job 312, the final variant).
set -u
OUT="$(realpath "$OUT")"
S="$HOME/arca-runs/scorecard"
python3 -m pipeline_eval.compare "$S/baseline/scorecard.json" "$S/fixes-v2/scorecard.json" --out "$OUT/delta_v2.md" || true
python3 -m pipeline_eval.compare "$S/baseline/scorecard.json" "$S/fixes-v3/scorecard.json" --out "$OUT/delta.md"
