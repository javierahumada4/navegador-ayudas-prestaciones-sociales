# est-minutes: 2
# Delta table baseline -> fixes (CPU only).
set -u
OUT="$(realpath "$OUT")"
S="$HOME/arca-runs/scorecard"
python3 -m pipeline_eval.compare "$S/baseline/scorecard.json" "$S/fixes/scorecard.json" --out "$OUT/delta.md"
