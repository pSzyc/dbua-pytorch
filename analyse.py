"""Build the MAE comparison table from ``run.py``'s proposed-loss sweep.

The SoS / B-mode figures are rendered by the backends themselves (``run.py``
runs with ``plot=True``); this file only turns ``results/phantoms-{backend}.json``
into a MAE (+/- SE) table over every phantom whose ground-truth speed is a known
constant (homogeneous phantoms), one column per backend. Heterogeneous phantoms
have no single ground-truth speed and are dropped.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("results")
COMPARISON_DIR = Path("comparison")


def _fmt_mae(entry: dict) -> str:
    if not entry or "error" in entry:
        return "err"
    mae, se = entry.get("mae"), entry.get("mae_se")
    if mae is None or se is None:
        return "--"
    return f"{mae:.2f}±{se:.2f}"


def _has_mae(entry: dict) -> bool:
    return bool(entry) and "error" not in entry and entry.get("mae") is not None


def mae_table(jax: dict, torch: dict) -> str:
    """MAE +/- SE per phantom, one column each for jax and torch."""
    samples = sorted(
        s for s in set(jax) | set(torch)
        if _has_mae(jax.get(s, {})) or _has_mae(torch.get(s, {}))
    )
    col_w = 14
    header = f"{'phantom':<16} {'jax':<{col_w}} {'torch':<{col_w}}"
    lines = [header, "-" * len(header)]
    for s in samples:
        j = _fmt_mae(jax.get(s, {}))
        t = _fmt_mae(torch.get(s, {}))
        lines.append(f"{s:<16} {j:<{col_w}} {t:<{col_w}}")
    return "\n".join(lines)


if __name__ == "__main__":
    COMPARISON_DIR.mkdir(exist_ok=True)

    with open(RESULTS_DIR / "phantoms-jax.json") as f:
        jax_phantoms = json.load(f)
    with open(RESULTS_DIR / "phantoms-torch.json") as f:
        torch_phantoms = json.load(f)

    table = mae_table(jax_phantoms, torch_phantoms)
    print(table)
    with open(COMPARISON_DIR / "mae_table.txt", "w") as f:
        f.write(table + "\n")
