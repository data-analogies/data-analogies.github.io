import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_data(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def plot_figure(data: Dict) -> plt.Figure:
    """
    Plot comparison of architectures:
      - Pi0 (pretrained)
      - Pi0 (no pretraining)
      - Diffusion Policy

    The JSON format is:
    {
      "tasks": [...],
      "architectures": [...],
      "data": [
        {"task": str, "architecture": str, "mean": float, "ci95": float},
        ...
      ]
    }
    """
    tasks: List[str] = data["tasks"]
    architectures: List[str] = data["architectures"]

    # Colors consistent across the figure
    arch_to_color = {
        "Pi0": "#4f8ef7",  # blue
        "Pi0 (no pretraining)": "#9e9e9e",  # gray
        "Diffusion Policy": "#2e7d32",  # green
    }

    n_tasks = len(tasks)
    fig_width = 4.0 * max(n_tasks, 1)
    fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, 3.8), sharey=True)
    if n_tasks == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    bar_width = 0.7
    x = np.arange(len(architectures))

    # Collect max y for consistent limits
    all_means_plus_ci = []

    for ax, task in zip(axes, tasks):
        means = []
        cis = []
        for arch in architectures:
            entry = next(
                item
                for item in data["data"]
                if item["task"] == task and item["architecture"] == arch
            )
            means.append(float(entry["mean"]))
            cis.append(float(entry["ci95"]))

        means_arr = np.array(means, dtype=float)
        cis_arr = np.array(cis, dtype=float)
        all_means_plus_ci.extend((means_arr + cis_arr).tolist())

        for i, arch in enumerate(architectures):
            color = arch_to_color.get(arch, "#555555")
            ax.bar(
                x[i],
                means_arr[i],
                width=bar_width,
                color=color,
                edgecolor="#333333",
                linewidth=0.8,
                yerr=cis_arr[i],
                error_kw=dict(
                    ecolor="#333333",
                    lw=0.8,
                    capsize=3,
                    capthick=0.8,
                ),
            )

        # Annotate values
        for i, (m, c) in enumerate(zip(means_arr, cis_arr)):
            ax.text(
                x[i],
                m + c + 1.0,
                f"{m:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

        ax.set_title(task, fontsize=12, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Pi0", "Pi0\n(no pretrain)", "Diffusion\nPolicy"],
            fontsize=9,
        )
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Success Rate (%)", fontsize=11)

    max_y = max(all_means_plus_ci) if all_means_plus_ci else 0.0
    for ax in axes:
        ax.set_ylim(0, max(85, int(np.ceil(max_y / 5.0) * 5 + 5)))

    # Legend for architectures
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=arch_to_color.get(arch, "#555555"),
            lw=8,
            label=arch,
        )
        for arch in architectures
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(architectures),
        fontsize=10,
        frameon=False,
        labelspacing=0.3,
        borderaxespad=0.2,
        handlelength=1.6,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir, "figure5_pi0_architecture_comparison.json")
    data = load_data(data_path)
    fig = plot_figure(data)

    out_png = os.path.join(this_dir, "figure5_pi0_architecture_comparison.png")
    out_pdf = os.path.join(this_dir, "figure5_pi0_architecture_comparison.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved Figure 5 (Pi0 architecture comparison) to: {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
