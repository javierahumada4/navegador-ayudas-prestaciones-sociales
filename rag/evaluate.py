"""Phase 3 evaluation: Recall@k and MRR of the three retrievers on the gold set.

The gold set (``rag/gold/gold.jsonl``, see ``rag/gold.py``) stores the evidence that answers each
question; ``resolve_gold`` turns it into chunk ids for whichever chunking is evaluated. Every
sub-command prints a Markdown table and writes its numbers to ``reports/phase3_<name>.json``::

    uv run python -m rag.evaluate chunking            # recursive vs semantic, 3 retrievers
    uv run python -m rag.evaluate lambda              # sweep of the hybrid weight, with CV
    uv run python -m rag.evaluate instructions        # dense retrieval vs query instruction
    uv run python -m rag.evaluate tokenizer           # BM25 tokenisation ablation
    uv run python -m rag.evaluate failures            # where each retriever misses
    uv run python -m rag.evaluate plot-distances      # sentence distances + threshold (figure)

Chunking specs: ``recursive:<size>:<overlap>`` and ``semantic:<threshold>:<max_chars>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag.chunking import (
    Chunk,
    consecutive_distances,
    describe,
    recursive_chunks,
    semantic_chunks,
    sentence_spans,
)
from rag.embed import DEFAULT_QUERY_INSTRUCTION, Embedder
from rag.gold import GoldItem, item_recall, item_reciprocal_rank, load_gold, resolve_gold
from rag.ingest import Document, load_corpus
from rag.retriever import (
    BM25Retriever,
    DenseRetriever,
    Hit,
    fuse,
    tokenize,
    tokenize_es,
)

REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("docs/fase3_figs")
KS = (1, 3, 5, 10)
DEPTH = 20  # ranking depth used for Recall@k and MRR
DEFAULT_CHUNKING = "recursive:800:100"
LAMBDA_GRID = [round(0.1 * i, 1) for i in range(11)]


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top k."""
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / position of the first relevant chunk (0 when none is retrieved)."""
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def evaluate_ranking(rankings: list[list[str]], items: list[GoldItem]) -> dict[str, float]:
    """Recall@k (k in ``KS``) and MRR of one retriever, given its ranked chunk ids per question."""
    result = {
        f"recall@{k}": float(
            np.mean([item_recall(r, i, k) for r, i in zip(rankings, items, strict=True)])
        )
        for k in KS
    }
    result["mrr"] = float(
        np.mean([item_reciprocal_rank(r, i) for r, i in zip(rankings, items, strict=True)])
    )
    result["n_questions"] = len(items)
    return result


def rank_ids(search, items: list[GoldItem], depth: int = DEPTH) -> list[list[str]]:
    """Ranked chunk ids of ``search(question, top_k)`` for every gold question."""
    return [[hit.chunk.id for hit in search(item.question, depth)] for item in items]


def evaluate_retriever(retriever, items: list[GoldItem]) -> dict[str, float]:
    return evaluate_ranking(rank_ids(retriever.search, items), items)


# ------------------------------------------------------------------------------ setup helpers


class CachedEmbedFn:
    """Sentence embeddings are the same for every threshold: embed each document only once."""

    def __init__(self, embedder: Embedder):
        self.embedder, self.cache = embedder, {}

    def __call__(self, sentences: list[str]) -> np.ndarray:
        key = hash(tuple(sentences))
        if key not in self.cache:
            self.cache[key] = self.embedder.embed_passages(sentences, batch_size=64)
        return self.cache[key]


def build_chunks(spec: str, documents: list[Document], embed_fn) -> list[Chunk]:
    kind, first, second = spec.split(":")
    if kind == "recursive":
        return recursive_chunks(documents, chunk_size=int(first), chunk_overlap=int(second))
    if kind == "semantic":
        return semantic_chunks(
            documents, embed_fn, distance_threshold=float(first), max_chunk_chars=int(second)
        )
    raise ValueError(f"unknown chunking {spec!r}")


def make_embedder() -> Embedder:
    import torch

    return Embedder(device="cuda" if torch.cuda.is_available() else "cpu")


def markdown_table(headers: list[str], rows: list[list]) -> str:
    def cell(value) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def save_report(name: str, payload: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"phase3_{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nnumbers written to {path}")


def pools(bm25, dense, items: list[GoldItem], pool: int = 20) -> list[tuple[list[Hit], list[Hit]]]:
    """BM25 and dense top-``pool`` hits per question, computed once and reused to fuse."""
    return [(bm25.search(i.question, pool), dense.search(i.question, pool)) for i in items]


def hybrid_rankings(question_pools, weight_bm25: float) -> list[list[str]]:
    return [
        [h.chunk.id for h in fuse(bm25_hits, dense_hits, weight_bm25, DEPTH)]
        for bm25_hits, dense_hits in question_pools
    ]


# ------------------------------------------------------------------------------- experiments


def run_chunking(specs: list[str], weight_bm25: float, persist_dir: str) -> None:
    documents = load_corpus()
    embedder = make_embedder()
    embed_fn = CachedEmbedFn(embedder)
    rows, report = [], {}
    for spec in specs:
        chunks = build_chunks(spec, documents, embed_fn)
        items = resolve_gold(load_gold(), chunks)
        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, embedder, persist_dir)
        question_pools = pools(bm25, dense, items)
        scores = {
            "bm25": evaluate_retriever(bm25, items),
            "dense": evaluate_retriever(dense, items),
            "hybrid": evaluate_ranking(hybrid_rankings(question_pools, weight_bm25), items),
        }
        stats = describe(chunks)
        report[spec] = {"chunks": stats, "scores": scores}
        rows.append(
            [spec, stats["n_chunks"], round(stats["mean_chars"])]
            + [scores[r][m] for r in ("bm25", "dense", "hybrid") for m in ("recall@5", "mrr")]
        )
    headers = ["chunking", "n", "mean chars"] + [
        f"{r} {m}" for r in ("bm25", "dense", "hybrid") for m in ("R@5", "MRR")
    ]
    print(f"hybrid weight λ = {weight_bm25}\n")
    print(markdown_table(headers, rows))
    save_report("chunking", report)


def cross_validated_lambda(question_pools, items, folds: int = 5) -> dict[str, float]:
    """Choose λ on 4 folds, score on the 5th: the honest estimate of tuning λ on 50 questions."""
    order = np.random.default_rng(0).permutation(len(items))
    held_out = []
    for fold in range(folds):
        test = set(order[fold::folds].tolist())
        train = [i for i in range(len(items)) if i not in test]

        def mrr(weight, indices):
            ranked = hybrid_rankings([question_pools[i] for i in indices], weight)
            return np.mean(
                [item_reciprocal_rank(r, items[i]) for r, i in zip(ranked, indices, strict=True)]
            )

        best = max(LAMBDA_GRID, key=lambda w: (round(mrr(w, train), 9), -abs(w - 0.5)))
        held_out.append(mrr(best, sorted(test)))
    return {"cv_mrr": float(np.mean(held_out))}


def run_lambda(chunking: str, persist_dir: str) -> None:
    documents = load_corpus()
    embedder = make_embedder()
    chunks = build_chunks(chunking, documents, CachedEmbedFn(embedder))
    items = resolve_gold(load_gold(), chunks)
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks, embedder, persist_dir)
    question_pools = pools(bm25, dense, items)
    rows, report = [], {}
    for weight in LAMBDA_GRID:
        scores = evaluate_ranking(hybrid_rankings(question_pools, weight), items)
        report[str(weight)] = scores
        rows.append([weight] + [scores[f"recall@{k}"] for k in KS] + [scores["mrr"]])
    print(f"chunking {chunking}; λ = weight of BM25 (λ=0 is dense alone, λ=1 is BM25 alone)\n")
    print(markdown_table(["λ"] + [f"R@{k}" for k in KS] + ["MRR"], rows))
    best = max(LAMBDA_GRID, key=lambda w: (report[str(w)]["mrr"], -abs(w - 0.5)))
    cv = cross_validated_lambda(question_pools, items)
    print(
        f"\nbest λ by MRR: {best}; 5-fold cross-validated MRR with λ chosen on 4 folds: "
        f"{cv['cv_mrr']:.3f} (vs {report[str(best)]['mrr']:.3f} in-sample)"
    )
    report["best"] = {"lambda": best, **cv}
    save_report("lambda", report)


INSTRUCTIONS = {
    "none": "",
    "default": DEFAULT_QUERY_INSTRUCTION,
    "generic": "Given a web search query, retrieve relevant passages that answer the query",
    "legal-en": (
        "Given a question about the Spanish minimum income benefit (IMV), retrieve the legal "
        "provision that answers it"
    ),
    "legal-es": (
        "Dada una pregunta sobre el ingreso mínimo vital, recupera el artículo de la norma que "
        "la responde"
    ),
}


def run_instructions(chunking: str, persist_dir: str) -> None:
    documents = load_corpus()
    embedder = make_embedder()
    chunks = build_chunks(chunking, documents, CachedEmbedFn(embedder))
    items = resolve_gold(load_gold(), chunks)
    dense = DenseRetriever(chunks, embedder, persist_dir)
    rows, report = [], {}
    for name, instruction in INSTRUCTIONS.items():
        ranked = [
            [h.chunk.id for h in dense.search(i.question, DEPTH, instruction=instruction)]
            for i in items
        ]
        scores = evaluate_ranking(ranked, items)
        report[name] = {"instruction": instruction, **scores}
        rows.append([name] + [scores[f"recall@{k}"] for k in KS] + [scores["mrr"]])
    print(
        f"chunking {chunking}; dense retriever; the passages never change, only the query prompt\n"
    )
    print(markdown_table(["instruction"] + [f"R@{k}" for k in KS] + ["MRR"], rows))
    save_report("instructions", report)


def run_tokenizer(chunking: str) -> None:
    documents = load_corpus()
    chunks = build_chunks(chunking, documents, None)
    items = resolve_gold(load_gold(), chunks)
    rows, report = [], {}
    for name, tokenizer in (
        ("lower + \\w+", tokenize),
        ("accents folded + stop words", tokenize_es),
    ):
        scores = evaluate_retriever(BM25Retriever(chunks, tokenizer), items)
        report[name] = scores
        rows.append([name] + [scores[f"recall@{k}"] for k in KS] + [scores["mrr"]])
    print(markdown_table(["BM25 tokenisation"] + [f"R@{k}" for k in KS] + ["MRR"], rows))
    save_report("tokenizer", report)


def run_failures(chunking: str, weight_bm25: float, persist_dir: str, k: int) -> None:
    documents = load_corpus()
    embedder = make_embedder()
    chunks = build_chunks(chunking, documents, CachedEmbedFn(embedder))
    items = resolve_gold(load_gold(), chunks)
    by_id = {c.id: c for c in chunks}
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks, embedder, persist_dir)
    question_pools = pools(bm25, dense, items)
    rankings = {
        "bm25": rank_ids(bm25.search, items),
        "dense": rank_ids(dense.search, items),
        "hybrid": hybrid_rankings(question_pools, weight_bm25),
    }
    report = []
    for position, item in enumerate(items):
        ranks = {}
        for name, ranked in rankings.items():
            relevant = item.relevant_chunk_ids
            ranks[name] = next(
                (r for r, cid in enumerate(ranked[position], 1) if cid in relevant), None
            )
        report.append(
            {
                "id": item.id,
                "question": item.question,
                "category": item.category,
                "style": item.style,
                "first_relevant_rank": ranks,
            }
        )
        if any(r is None or r > k for r in ranks.values()):
            print(f"\n{item.id} [{item.category}/{item.style}] {item.question}")
            print(f"  rank of the first relevant chunk: {ranks}")
            for name in ("bm25", "dense"):
                top = rankings[name][position][0]
                print(f"  {name} top-1: {top}: {by_id[top].text[:110]!r}")
    save_report("failures", {"k": k, "questions": report})


def run_plot_distances(doc_id: str, thresholds: list[float], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    documents = [d for d in load_corpus() if d.id == doc_id]
    if not documents:
        raise SystemExit(f"no document {doc_id!r} in the corpus")
    text = documents[0].text
    sentences = [text[a:b] for a, b in sentence_spans(text)]
    distances = consecutive_distances(sentences, CachedEmbedFn(make_embedder()))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    top.plot(distances, linewidth=0.6, color="#4c72b0")
    for threshold in thresholds:
        top.axhline(threshold, linestyle="--", linewidth=1, label=f"threshold {threshold}")
    top.set(title=f"Cosine distance between consecutive sentences: {doc_id}", ylabel="distance")
    top.set_xlabel("sentence index")
    top.legend()
    bottom.hist(distances, bins=60, color="#4c72b0")
    for threshold in thresholds:
        bottom.axvline(threshold, linestyle="--", linewidth=1)
    bottom.set(xlabel="distance", ylabel="pairs of sentences")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130)
    print(f"{len(sentences)} sentences; median distance {np.median(distances):.3f}; figure: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index-dir", default="rag/index")
    sub = parser.add_subparsers(dest="command", required=True)

    chunking = sub.add_parser("chunking")
    chunking.add_argument("--specs", nargs="+", default=[DEFAULT_CHUNKING, "semantic:0.65:1000"])
    chunking.add_argument("--weight", type=float, default=0.5)
    for name in ("lambda", "instructions", "tokenizer"):
        sub.add_parser(name).add_argument("--chunking", default=DEFAULT_CHUNKING)
    failures = sub.add_parser("failures")
    failures.add_argument("--chunking", default=DEFAULT_CHUNKING)
    failures.add_argument("--weight", type=float, default=0.5)
    failures.add_argument("-k", type=int, default=5)
    plot = sub.add_parser("plot-distances")
    plot.add_argument("--doc", default="ley-19-2021")
    plot.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.65])
    plot.add_argument("--out", type=Path, default=FIGURES_DIR / "distances.png")

    args = parser.parse_args()
    if args.command == "chunking":
        run_chunking(args.specs, args.weight, args.index_dir)
    elif args.command == "lambda":
        run_lambda(args.chunking, args.index_dir)
    elif args.command == "instructions":
        run_instructions(args.chunking, args.index_dir)
    elif args.command == "tokenizer":
        run_tokenizer(args.chunking)
    elif args.command == "failures":
        run_failures(args.chunking, args.weight, args.index_dir, args.k)
    else:
        run_plot_distances(args.doc, args.thresholds, args.out)


if __name__ == "__main__":
    main()
