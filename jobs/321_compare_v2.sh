# est-minutes: 2
# Delta table baseline (jobs 300 + 302) -> fixes (job 311), with the 2-sigma noise column.
set -u
OUT="$(realpath "$OUT")"
S="$HOME/arca-runs/scorecard"
python3 -m pipeline_eval.compare "$S/baseline/scorecard.json" "$S/fixes-v2/scorecard.json" --out "$OUT/delta.md"
