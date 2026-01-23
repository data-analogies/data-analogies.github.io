import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_data(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def plot_figure(data: Dict) -> plt.Figure:
    tasks: List[str] = data["tasks"]  # ["Pen in cup", "Book on bookshelf"]
    robots: List[str] = data["robots"]  # e.g., ["PiperX -> WidowX", "WidowX -> Franka", ...]
    conditions: List[str] = data["conditions"]

    # Use Figure 1 colors: gray, blue, green
    fig1_colors = ["#9e9e9e", "#4f8ef7", "#2e7d32"]
    robot_to_color = {r: fig1_colors[i % len(fig1_colors)] for i, r in enumerate(robots)}

    # Layout: 1 row x N task columns (two tasks for this figure)
    n_rows = 1
    n_cols = len(tasks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.2, 3.9), sharey=True)
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    # Clustered bar positions (repeat condition set per robot group)
    n_cond = len(conditions)
    n_robots = len(robots)
    cluster_gap = 0.9
    bar_width = 0.8

    # Abbreviated x tick labels, same style as Figure 2
    short_cond_labels = [
        "Baseline",
        "Narrow\n(Bridge+DROID)",
        "OXE",
        "OXE+Translational",
    ]

    for ax, task in zip(axes, tasks):
        # Collect values per robot per condition
        means_by_robot = {r: [] for r in robots}
        cis_by_robot = {r: [] for r in robots}
        for cond in conditions:
            for r in robots:
                entry = next(
                    item
                    for item in data["data"]
                    if item["task"] == task
                    and item["robot"] == r
                    and item["condition"] == cond
                )
                means_by_robot[r].append(float(entry["mean"]))
                cis_by_robot[r].append(float(entry["ci95"]))

        # Compute x positions for each robot cluster
        x_clusters = [np.arange(n_cond) + i * (n_cond + cluster_gap) for i in range(n_robots)]
        x_all = np.concatenate(x_clusters) if x_clusters else np.array([])

        # Plot each robot cluster
        for i, r in enumerate(robots):
            ax.bar(
                x_clusters[i],
                means_by_robot[r],
                width=bar_width,
                color=robot_to_color[r],
                edgecolor="#333333",
                linewidth=0.8,
                yerr=cis_by_robot[r],
                error_kw=dict(ecolor="#333333", lw=0.8, capsize=3, capthick=0.8),
                label=r if ax is axes[0] else None,
            )
            for xi, m, ci in zip(x_clusters[i], means_by_robot[r], cis_by_robot[r]):
                ax.text(
                    xi,
                    m + ci + 1.0,
                    f"{m:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#222222",
                )

        # Titles, ticks, separators
        ax.set_title(task, fontsize=12, pad=6)
        ax.set_xticks(x_all)
        ax.set_xticklabels(short_cond_labels * n_robots, fontsize=9, rotation=25, ha="right", rotation_mode="anchor")
        for i in range(n_robots - 1):
            sep_x = (x_clusters[i][-1] + x_clusters[i + 1][0]) / 2
            ax.axvline(sep_x, color="#999999", lw=0.8, ls=":")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Success Rate (%)", fontsize=11)

    # Shared y-limit to accommodate annotations
    max_y = max((entry["mean"] + entry["ci95"]) for entry in data["data"]) if data.get("data") else 0
    for ax in axes:
        ax.set_ylim(0, max(80, int(np.ceil(max_y / 5.0) * 5 + 5)))

    # Figure-level legend for robots
    handles = [
        plt.Line2D([0], [0], color=robot_to_color[r], lw=8, label=r)
        for r in robots
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(robots), 4),
        fontsize=10,
        frameon=False,
        labelspacing=0.3,
        borderaxespad=0.2,
        handlelength=1.6,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir, "figure4_real_world.json")
    data = load_data(data_path)
    fig = plot_figure(data)

    out_png = os.path.join(this_dir, "figure4_real_world.png")
    out_pdf = os.path.join(this_dir, "figure4_real_world.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved Figure 4 to: {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()


