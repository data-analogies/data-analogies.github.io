import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_data(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def _make_order(robots: List[str], conditions: List[str]) -> List[Tuple[str, str]]:
    return [(robot, condition) for robot in robots for condition in conditions]


def plot_figure(data: Dict) -> plt.Figure:
    tasks: List[str] = data["tasks"]
    environments: List[str] = data["environments"]
    robots: List[str] = data["robots"]
    conditions: List[str] = data["conditions"]

    order = _make_order(robots, conditions)

    robot_to_color = {
        robots[0]: "#4f8ef7",
        robots[1]: "#2e7d32",
    }
    condition_to_hatch = {
        conditions[0]: "",
        conditions[1]: "//",
    }

    n_tasks = len(tasks)
    n_bars = len(order)
    bar_width = 0.18
    cluster_gap = 0.35
    base = np.arange(n_tasks) * (n_bars * bar_width + cluster_gap)
    x_ticks = base + (n_bars - 1) * bar_width / 2

    fig, axes = plt.subplots(1, len(environments), figsize=(12.8, 3.9), sharey=True)
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    max_y = 0.0
    for ax, env in zip(axes, environments):
        for task_idx, task in enumerate(tasks):
            entries = []
            for robot, condition in order:
                entry = next(
                    item
                    for item in data["data"]
                    if item["environment"] == env
                    and item["task"] == task
                    and item["robot"] == robot
                    and item["condition"] == condition
                )
                entries.append(entry)

            for i, entry in enumerate(entries):
                robot, condition = order[i]
                x = base[task_idx] + i * bar_width
                mean = float(entry["mean"])
                ci = float(entry["ci95"])
                max_y = max(max_y, mean + ci)
                ax.bar(
                    x,
                    mean,
                    width=bar_width,
                    color=robot_to_color[robot],
                    edgecolor="#333333",
                    linewidth=0.8,
                    yerr=ci,
                    error_kw=dict(ecolor="#333333", lw=0.8, capsize=3, capthick=0.8),
                    hatch=condition_to_hatch[condition],
                )
                ax.text(
                    x,
                    mean + ci + 1.0,
                    f"{mean:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#222222",
                )

        ax.set_title(env, fontsize=12, pad=6)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(["Strawberry\nin Pot", "Stack\nRed on Blue"], fontsize=9)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Success Rate (%)", fontsize=11)
    for ax in axes:
        ax.set_ylim(0, max(80, int(np.ceil(max_y / 5.0) * 5 + 5)))

    # Legends: robot colors + condition hatches
    from matplotlib.patches import Patch

    robot_handles = [
        Patch(facecolor=robot_to_color[r], edgecolor="#333333", label=r)
        for r in robots
    ]
    condition_handles = [
        Patch(facecolor="#ffffff", edgecolor="#333333", hatch=condition_to_hatch[c], label=c)
        for c in conditions
    ]
    fig.legend(
        handles=robot_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(robots),
        fontsize=9,
        title="Embodiment",
        title_fontsize=9,
        frameon=False,
    )
    fig.legend(
        handles=condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(conditions),
        fontsize=9,
        title="Training Data",
        title_fontsize=9,
        frameon=False,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir, "figure_bridge_transfer.json")
    data = load_data(data_path)
    fig = plot_figure(data)

    out_png = os.path.join(this_dir, "figure_bridge_transfer.png")
    out_pdf = os.path.join(this_dir, "figure_bridge_transfer.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved BRIDGE transfer figure to: {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()


