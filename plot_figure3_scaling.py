import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_data(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def plot_figure(data: Dict) -> plt.Figure:
    xs_global: List[int] = data.get("x", [])
    methods: List[str] = data["methods"]
    domains: List[str] = data["domains"]
    refs = data.get("refs", {})
    x_by_domain = data.get("x_by_domain", {})
    target_only_global = float(refs.get("target_only", 0.0))
    upper_bound_global = float(refs.get("upper_bound", 0.0))
    target_only_by_domain = refs.get("target_only_by_domain", {})
    upper_bound_by_domain = refs.get("upper_bound_by_domain", {})

    # Map method to color/marker
    method_style = {
        "Naive (Uniform)": {"color": "#9e9e9e", "marker": "o"},
        "Targeted Coverage": {"color": "#4f8ef7", "marker": "s"},
        "Trajectory-Paired": {"color": "#2e7d32", "marker": "^"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8), sharey=True)

    # Collect and plot per domain
    line_handles = []
    for ax, domain in zip(axes, domains):
        xs = x_by_domain.get(domain, xs_global)
        for m in methods:
            x_vals, y_vals, y_errs = [], [], []
            for x in xs:
                entry = next(item for item in data["data"] if item["domain"] == domain and item["method"] == m and item["x"] == x)
                x_vals.append(x)
                y_vals.append(entry["mean"])
                y_errs.append(entry["ci95"])
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals, dtype=float)
            e_arr = np.array(y_errs, dtype=float)
            style = method_style[m]
            ln = ax.errorbar(
                x_arr,
                y_arr,
                yerr=e_arr,
                color=style["color"],
                marker=style["marker"],
                markersize=6,
                linewidth=2.0,
                capsize=3,
                label=m,
            )
            if ax is axes[0]:
                line_handles.append(ln)

        # Reference lines per axis (allow per-domain overrides)
        target_only = float(target_only_by_domain.get(domain, target_only_global))
        upper_bound = float(upper_bound_by_domain.get(domain, upper_bound_global))
        if target_only > 0:
            ax.axhline(target_only, linestyle="--", color="#7f7f7f", linewidth=1.0)
        if upper_bound > 0:
            ax.axhline(upper_bound, linestyle=(0, (4, 2)), color="#b22222", linewidth=1.0)

        ax.set_title(domain, fontsize=12, pad=8)
        ax.set_xticks(xs)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Success Rate (%)", fontsize=11)
    axes[1].set_xlabel("Diversity of training data", fontsize=11)

    # Shared y limit
    max_y = upper_bound_global
    for ax in axes:
        ax.set_ylim(0, max(80, np.ceil(max_y / 5) * 5 + 5))

    # Build figure-level legend including reference lines
    ref_handles = [
        plt.Line2D([0], [0], color="#7f7f7f", lw=1.2, ls="--", label="Target-only (few-shot)"),
        plt.Line2D([0], [0], color="#b22222", lw=1.2, ls=(0, (4, 2)), label="Target upper bound"),
    ]
    method_handles = [
        plt.Line2D([0], [0], color=method_style[m]["color"], marker=method_style[m]["marker"], lw=2.0, label=m)
        for m in methods
    ]
    fig.legend(
        handles=method_handles + ref_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        fontsize=9,
        frameon=False,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir, "figure3_scaling.json")
    data = load_data(data_path)
    fig = plot_figure(data)

    out_png = os.path.join(this_dir, "figure3_scaling.png")
    out_pdf = os.path.join(this_dir, "figure3_scaling.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved Figure 3 to: {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
