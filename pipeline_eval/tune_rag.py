"""Choose the phase-3 serving parameters from data instead of the provisional defaults.

Reads the reports that ``rag.evaluate`` writes (``phase3_lambda_<chunking>.json`` for each
candidate chunking and ``phase3_calibrate.json``) and writes ``rag_tuned.env`` (``export`` lines
the scorecard job sources) plus ``rag_tuned.json`` with the reasons:

- **chunking and λ**: the (chunking, λ) with the best MRR on the 50 gold questions. Only recursive
  chunkings are candidates: the scorecard resolves gold ids with them. The 5-fold
  cross-validated MRR is reported next to it, because choosing λ on the same 50 questions inflates
  the in-sample number.
- **abstention gate** (top-1 dense cosine): the highest threshold that keeps at least
  ``--keep`` of the answerable questions (default 0.96 = at most 2 of 50 wrongly refused); 0
  (gate off) if no threshold rejects any unanswerable question at that cost. Calibrated and
  scored on the same 8 negatives: optimistic, and said so in the notebook.

    python -m pipeline_eval.tune_rag --reports reports --out-dir reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def best_lambda(reports: Path) -> dict:
    best = None
    for path in sorted(reports.glob("phase3_lambda_recursive_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        chunking = path.stem.removeprefix("phase3_lambda_").replace("_", ":")
        weight = report["best"]["lambda"]
        candidate = {"chunking": chunking, "lambda": weight, "mrr": report[str(weight)]["mrr"],
                     "recall@5": report[str(weight)].get("recall@5"),
                     "cv_mrr": report["best"].get("cv_mrr")}
        if best is None or candidate["mrr"] > best["mrr"]:
            best = candidate
    if best is None:
        raise SystemExit(f"no phase3_lambda_recursive_*.json in {reports}")
    return best


def best_gate(reports: Path, keep: float) -> dict:
    report = json.loads((reports / "phase3_calibrate.json").read_text(encoding="utf-8"))
    answerable, negatives = report["answerable"], report["unanswerable"]
    best = {"min_score": 0.0, "kept": 1.0, "rejected": 0.0}
    for step in range(30, 91):
        gate = step / 100
        kept = sum(s >= gate for s in answerable) / len(answerable)
        rejected = sum(s < gate for s in negatives) / len(negatives)
        if kept >= keep and rejected > best["rejected"]:
            best = {"min_score": gate, "kept": kept, "rejected": rejected}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--keep", type=float, default=0.96)
    args = parser.parse_args()
    lam = best_lambda(args.reports)
    gate = best_gate(args.reports, args.keep)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rag_tuned.json").write_text(
        json.dumps({"retrieval": lam, "gate": gate}, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "rag_tuned.env").write_text(
        f"export ARCA_RAG_CHUNKING={lam['chunking']}\n"
        f"export ARCA_RAG_LAMBDA={lam['lambda']}\n"
        f"export ARCA_RAG_MIN_SCORE={gate['min_score']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"retrieval": lam, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
