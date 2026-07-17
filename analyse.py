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
ACCURACY = ["mean_c", "mae", "rmse"]


def plot_differences(sample: str, c_true: float):
    try:
        result_jax = np.load(f"results/{sample}-jax.npy")
        result_torch = np.load(f"results/{sample}-torch.npy")
    except Exception as e:
        print(f"Problems with samples {sample}\n{e}")
        print("=======")
        return

    diff_jax_torch = result_jax - result_torch
    fig_diff, ax_diff = plt.subplots()
    im_diff = ax_diff.imshow(diff_jax_torch, cmap="RdBu_r")
    ax_diff.set_title("JAX - PyTorch")
    fig_diff.colorbar(im_diff, ax=ax_diff)
    fig_diff.savefig(f"comparison/diff_jax_torch_{sample}.png")

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
        fig_true.savefig(f"comparison/diff_ctrue_{sample}.png", dpi=300)


def load_results(backend: str) -> dict:
    with open(f"results/results-{backend}.json") as f:
        return json.load(f)


def losses_table(jax: dict, torch: dict) -> str:
    header = f"{'sample':<16} " + " ".join(
        f"{l}(jax) {l}(torch) {l}(|diff|)" for l in LOSSES
    )
    lines = [header, "-" * len(header)]
    for s in jax:
        j, t = jax.get(s, {}), torch.get(s, {})
        if "error" in j and "error" in t:
            continue
        cells = []
        for l in LOSSES:
            jv, tv = j.get(l), t.get(l)
            cells.append(
                f"{jv:8.4f} {tv:9.4f} {abs(jv - tv):9.4f}"
                if jv is not None and tv is not None
                else f"{'--':>8} {'--':>9} {'--':>9}"
            )
        lines.append(f"{s:<16} " + " ".join(cells))
    return "\n".join(lines)


def scatter_accuracy(jax: dict, torch: dict, out: str):
    # only samples with a ground-truth constant sound speed
    samples = [s for s in jax if "c_true" in jax[s]]
    c_true = np.array([jax[s]["c_true"] for s in samples])
    order = np.argsort(c_true)
    c_true = c_true[order]
    samples = [samples[i] for i in order]

    fig, axes = plt.subplots(1, len(ACCURACY), figsize=(5 * len(ACCURACY), 4.5))
    for ax, metric in zip(axes, ACCURACY):
        jv = np.array([jax[s][metric] for s in samples])
        tv = np.array([torch[s][metric] for s in samples])
        ax.scatter(c_true, jv, marker="o", s=60, label="JAX", color="tab:blue")
        ax.scatter(c_true, tv, marker="x", s=70, label="PyTorch", color="tab:red")
        if metric == "mean_c":
            ax.plot(c_true, c_true, color="k", ls="--", lw=1, label="expected")
        ax.set_title(metric)
        ax.set_xlabel("c_true (m/s)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    for sample, c_true in CTRUE.items():
        plot_differences(sample, c_true)

    jax, torch = load_results("jax"), load_results("torch")
    table = losses_table(jax, torch)
    print(table)
    with open("comparison/losses_table.txt", "w") as f:
        f.write(table + "\n")
    scatter_accuracy(jax, torch, "comparison/accuracy_scatter.png")
