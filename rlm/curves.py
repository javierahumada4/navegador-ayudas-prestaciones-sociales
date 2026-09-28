"""Training curves for phase 1 from TRL's log history.

Reads ``log_history.json`` (written by ``train_grpo.py`` at the end of a run) or the
``trainer_state.json`` inside any checkpoint, so a run that died halfway can still be
plotted. Several runs can be overlaid to compare hyperparameters:

    uv run python -m rlm.curves --runs grpo=rlm/weights/grpo_lora --out reports/grpo_curves.png
    uv run python -m rlm.curves --runs beta0=rlm/weights/grpo_b0 beta004=rlm/weights/grpo_b004
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (title, [(log key, legend label)]). Panels whose keys never appear are skipped.
PANELS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Recompensa total", [("reward", "total")]),
    (
        "Recompensas",
        [
            ("rewards/format_reward/mean", "formato"),
            ("rewards/imv_accuracy_reward/mean", "exactitud"),
            ("rewards/accuracy_reward/mean", "exactitud"),
            ("rewards/domain_reward/mean", "dominio"),
        ],
    ),
    (
        "Longitud de la respuesta (tokens)",
        [
            ("completions/mean_length", "media"),
            ("completions/mean_terminated_length", "media sin truncadas"),
        ],
    ),
    (
        "Fracciones",
        [
            ("completions/clipped_ratio", "truncadas"),
            ("rewards/zero_answer_rate/mean", "responde 0"),
            ("frac_reward_zero_std", "grupos sin señal"),
        ],
    ),
    (
        "Actualización de la política",
        [("clip_ratio/region_mean", "fracción recortada"), ("kl", "KL")],
    ),
    ("Entropía", [("entropy", "entropía")]),
]


def find_history(path: str | Path) -> Path:
    """Accept a JSON file, a run output dir, or a checkpoint dir."""
    path = Path(path)
    if path.is_file():
        return path
    if (path / "log_history.json").exists():
        return path / "log_history.json"
    if (path / "trainer_state.json").exists():
        return path / "trainer_state.json"
    checkpoints = sorted(
        path.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1]),
    )
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"no log_history.json or trainer_state.json under {path}")


def load_history(path: str | Path) -> list[dict]:
    """Per-step training logs, without the final summary entry."""
    data = json.loads(find_history(path).read_text(encoding="utf-8"))
    history = data["log_history"] if isinstance(data, dict) else data
    return [entry for entry in history if "step" in entry and "train_runtime" not in entry]


def smooth(values: list[float], window: int) -> list[float]:
    """Trailing moving average; window 1 returns the values unchanged."""
    if window <= 1:
        return list(values)
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(e["step"], e[key]) for e in history if e.get(key) is not None]
    return [s for s, _ in points], [v for _, v in points]


def plot_runs(runs: dict[str, list[dict]], out: Path, window: int) -> list[str]:
    """Draw one panel per metric group; returns the titles that were drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    panels = [
        (title, [(k, label) for k, label in keys if any(series(h, k)[0] for h in runs.values())])
        for title, keys in PANELS
    ]
    panels = [(title, keys) for title, keys in panels if keys]
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.6 * rows), squeeze=False)
    for ax, (title, keys) in zip(axes.flat, panels, strict=False):
        for run_name, history in runs.items():
            for key, label in keys:
                steps, values = series(history, key)
                if not steps:
                    continue
                name = label if len(runs) == 1 else f"{run_name} · {label}"
                ax.plot(steps, smooth(values, window), label=name, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("paso")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(panels) :]:
        ax.axis("off")
    if window > 1:
        fig.suptitle(f"media móvil de {window} pasos", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [title for title, _ in panels]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="name=path pairs; path is a run dir, checkpoint dir or JSON file",
    )
    parser.add_argument("--window", type=int, default=5, help="moving-average window")
    parser.add_argument("--out", default="reports/grpo_curves.png")
    args = parser.parse_args()

    runs = {}
    for pair in args.runs:
        name, path = pair.split("=", 1) if "=" in pair else (Path(pair).name, pair)
        runs[name] = load_history(path)
        print(f"{name}: {len(runs[name])} logged steps from {find_history(path)}")
    drawn = plot_runs(runs, Path(args.out), args.window)
    print(f"{len(drawn)} panels -> {args.out}")


if __name__ == "__main__":
    main()
