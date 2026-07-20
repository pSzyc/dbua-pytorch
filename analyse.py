"""Build the MAE comparison table from the accuracy experiment in ``run.py``.

The SoS / B-mode figures are rendered by the backends themselves (``run.py``
runs with ``plot=True``); this file only turns ``results/mae-{backend}.json``
into a MAE +/- SE table over the phantom x loss grid, both backends.
"""

import json
from pathlib import Path

LOSSES = ["sb", "lc", "cf", "pe"]

RESULTS_DIR = Path("results")
COMPARISON_DIR = Path("comparison")


def _fmt_mae(entry: dict) -> str:
    if not entry or "error" in entry:
        return "err"
    mae, se = entry.get("mae"), entry.get("mae_se")
    if mae is None or se is None:
        return "--"
    return f"{mae:.2f}±{se:.2f}"


def mae_table(jax: dict, torch: dict) -> str:
    """MAE +/- SE over the phantom x loss grid; each cell is jax / torch."""
    samples = sorted(set(jax) | set(torch))
    col_w = 20
    header = f"{'phantom':<10} " + " ".join(f"{l:<{col_w}}" for l in LOSSES)
    sub = f"{'(jax / torch)':<10}"
    lines = [header, sub, "-" * len(header)]
    for s in samples:
        cells = []
        for loss in LOSSES:
            j = _fmt_mae(jax.get(s, {}).get(loss, {}))
            t = _fmt_mae(torch.get(s, {}).get(loss, {}))
            cells.append(f"{j + ' / ' + t:<{col_w}}")
        lines.append(f"{s:<10} " + " ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    COMPARISON_DIR.mkdir(exist_ok=True)

    with open(RESULTS_DIR / "mae-jax.json") as f:
        jax_mae = json.load(f)
    with open(RESULTS_DIR / "mae-torch.json") as f:
        torch_mae = json.load(f)

    table = mae_table(jax_mae, torch_mae)
    print(table)
    with open(COMPARISON_DIR / "mae_table.txt", "w") as f:
        f.write(table + "\n")
