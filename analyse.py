import json

import numpy as np
import matplotlib.pyplot as plt


CTRUE = {
    "1420": 1420,
    "1465": 1465,
    "1480": 1480,
    "1510": 1510,
    "1540": 1540,
    "1555": 1555,
    "1570": 1570,
    "inclusion": 0,
    "inclusion_layer": 0,
    "four_layer": 0,
    "two_layer": 0,
    "checker2": 0,
    "checker8": 0
}

LOSSES = ["sb", "lc", "cf", "pe"]

# Focusing losses shown as jax / torch / |diff| triplets in the all-loss table.
TABLE_METRICS = ["sb", "lc", "cf", "pe"]

# Cost function driving the all-phantoms plots.
PLOT_LOSS = "pe"

# Homogeneous phantom used for the all-loss tables (must match run.ALL_LOSS_SAMPLE).
ALL_LOSS_SAMPLE = "1540"


def load_results(name: str) -> dict:
    """Load a results JSON by file stem (``results/{name}.json``)."""
    with open(f"results/{name}.json") as f:
        return json.load(f)


def plot_differences(sample: str, c_true: float, loss: str = PLOT_LOSS):
    try:
        result_jax = np.load(f"results/{sample}-{loss}-jax.npy")
        result_torch = np.load(f"results/{sample}-{loss}-torch.npy")
    except Exception as e:
        print(f"Problems with sample {sample} ({loss})\n{e}")
        print("=======")
        return

    diff_jax_torch = result_jax - result_torch
    fig_diff, ax_diff = plt.subplots()
    im_diff = ax_diff.imshow(diff_jax_torch, cmap="RdBu_r")
    ax_diff.set_title(f"JAX - PyTorch ({loss})")
    fig_diff.colorbar(im_diff, ax=ax_diff)
    fig_diff.savefig(f"comparison/diff_jax_torch_{sample}.png")
    plt.close(fig_diff)

    if c_true > 0:
        diff_jax_true = result_jax - c_true
        diff_torch_true = result_torch - c_true
        vmax = max(np.abs(diff_jax_true).max(), np.abs(diff_torch_true).max())
        fig_true, (ax_jax_true, ax_torch_true) = plt.subplots(1, 2, figsize = (10, 3))
        im_jax_true = ax_jax_true.imshow(diff_jax_true, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_jax_true.set_title("JAX - c_true")
        fig_true.colorbar(im_jax_true, ax=ax_jax_true)
        im_torch_true = ax_torch_true.imshow(diff_torch_true, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_torch_true.set_title("PyTorch - c_true")
        fig_true.colorbar(im_torch_true, ax=ax_torch_true)
        fig_true.suptitle(f"{sample} ({loss})")
        fig_true.savefig(f"comparison/diff_ctrue_{sample}.png", dpi=300)
        plt.close(fig_true)


def losses_table(jax: dict, torch: dict, sample: str) -> str:
    """All-loss comparison for one homogeneous phantom.

    ``jax``/``torch`` are ``{driving_loss: metrics}``; each row is a driving
    loss, each metric column shows jax / torch / |diff| on the converged map.
    """
    header = (
        f"phantom: {sample}\n"
        + f"{'driving':<10} "
        + " ".join(f"{m}(jax) {m}(torch) {m}(|diff|)" for m in TABLE_METRICS)
        + f" {'MAE±SE(jax)':>18} {'MAE±SE(torch)':>18}"
    )
    lines = [header, "-" * len(header.splitlines()[-1])]
    for dl in LOSSES:
        j, t = jax.get(dl, {}), torch.get(dl, {})
        if "error" in j or "error" in t:
            lines.append(f"{dl:<10} (error)")
            continue
        cells = []
        for m in TABLE_METRICS:
            jv, tv = j.get(m), t.get(m)
            cells.append(
                f"{jv:8.4f} {tv:9.4f} {abs(jv - tv):9.4f}"
                if jv is not None and tv is not None
                else f"{'--':>8} {'--':>9} {'--':>9}"
            )
        cells.append(f"{_fmt_mae(j):>18} {_fmt_mae(t):>18}")
        lines.append(f"{dl:<10} " + " ".join(cells))
    return "\n".join(lines)


def _fmt_mae(metrics: dict) -> str:
    """Format ``MAE ± SE`` (m/s), or ``--`` when the phantom has no ground truth."""
    mae, se = metrics.get("mae"), metrics.get("mae_se")
    if mae is None or se is None:
        return "--"
    return f"{mae:.3f}±{se:.3f}"


def scatter_accuracy(jax: dict, torch: dict, out: str):
    """Reconstruction-accuracy scatter over the all-phantoms (PE) results.

    ``jax``/``torch`` are ``{sample: metrics}``.
    """
    # only samples with a ground-truth constant sound speed
    samples = [s for s in jax if "c_true" in jax[s]]
    c_true = np.array([jax[s]["c_true"] for s in samples])
    order = np.argsort(c_true)
    c_true = c_true[order]
    samples = [samples[i] for i in order]

    fig, (ax_mean, ax_mae) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Recovered mean sound speed vs. ground truth.
    jv = np.array([jax[s]["mean_c"] for s in samples])
    tv = np.array([torch[s]["mean_c"] for s in samples])
    ax_mean.scatter(c_true, jv, marker="o", s=60, label="JAX", color="tab:blue")
    ax_mean.scatter(c_true, tv, marker="x", s=70, label="PyTorch", color="tab:red")
    ax_mean.plot(c_true, c_true, color="k", ls="--", lw=1, label="expected")
    ax_mean.set_title("mean_c")
    ax_mean.set_ylabel("mean_c (m/s)")

    # MAE ± standard error vs. ground truth.
    jmae = np.array([jax[s]["mae"] for s in samples])
    jse = np.array([jax[s]["mae_se"] for s in samples])
    tmae = np.array([torch[s]["mae"] for s in samples])
    tse = np.array([torch[s]["mae_se"] for s in samples])
    ax_mae.errorbar(c_true, jmae, yerr=jse, fmt="o", ms=6, capsize=3,
                    label="JAX", color="tab:blue")
    ax_mae.errorbar(c_true, tmae, yerr=tse, fmt="x", ms=8, capsize=3,
                    label="PyTorch", color="tab:red")
    ax_mae.set_title("MAE ± SE")
    ax_mae.set_ylabel("MAE (m/s)")

    for ax in (ax_mean, ax_mae):
        ax.set_xlabel("c_true (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # --- Tables: all losses on the single homogeneous phantom ----------------
    jax_losses = load_results("losses-jax")
    torch_losses = load_results("losses-torch")
    table = losses_table(jax_losses, torch_losses, ALL_LOSS_SAMPLE)
    print(table)
    with open("comparison/losses_table.txt", "w") as f:
        f.write(table + "\n")

    # --- Plots: phase-error reconstruction across all phantoms ---------------
    for s, c_true in CTRUE.items():
        plot_differences(s, c_true, PLOT_LOSS)

    jax_phantoms = load_results("phantoms-jax")
    torch_phantoms = load_results("phantoms-torch")
    scatter_accuracy(jax_phantoms, torch_phantoms, "comparison/accuracy_scatter.png")
