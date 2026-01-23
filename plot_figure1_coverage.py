import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_data(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def plot_figure(data: Dict) -> plt.Figure:
    pairings: List[str] = ["Unpaired", "Task-Paired", "Trajectory-Paired"]
    coverages: List[str] = ["Targeted", "Diverse"]
    order = [(cov, pair) for cov in coverages for pair in pairings]

    # Colors by pairing, hatches by coverage for two-level legend
    pairing_to_color = {
        "Unpaired": "#9e9e9e",
        "Task-Paired": "#4f8ef7",
        "Trajectory-Paired": "#2e7d32",
    }
    hatch_by_coverage = {
        "Targeted": "",
        "Diverse": "//",
    }

    domains: List[str] = data["domains"]
    refs = data.get("refs", {})
    target_only_global = float(refs.get("target_only", 0.0))
    upper_bound_global = float(refs.get("upper_bound", 0.0))
    target_only_by_domain = refs.get("target_only_by_domain", {})
    upper_bound_by_domain = refs.get("upper_bound_by_domain", {})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)

    for ax, domain in zip(axes, domains):
        # Extract and order bars
        bars = []
        for cov, pair in order:
            entry = next(item for item in data["data"] if item["domain"] == domain and item["coverage"] == cov and item["pairing"] == pair)
            bars.append(entry)

        x = np.arange(len(bars))
        means = np.array([b["mean"] for b in bars], dtype=float)
        cis = np.array([b["ci95"] for b in bars], dtype=float)

        # Plot bars
        for i, b in enumerate(bars):
            cov = b["coverage"]
            pair = b["pairing"]
            color = pairing_to_color[pair]
            hatch = hatch_by_coverage[cov]
            ax.bar(
                x[i],
                means[i],
                width=0.8,
                color=color,
                edgecolor="#333333",
                linewidth=0.8,
                yerr=cis[i],
                error_kw=dict(ecolor="#333333", lw=0.8, capsize=3, capthick=0.8),
                hatch=hatch,
            )

        # Annotate means
        for i, m in enumerate(means):
            ax.text(
                x[i],
                m + cis[i] + 1.2,
                f"{m:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

        # Reference lines (allow per-domain overrides)
        target_only = float(target_only_by_domain.get(domain, target_only_global))
        upper_bound = float(upper_bound_by_domain.get(domain, upper_bound_global))
        if target_only > 0:
            ax.axhline(target_only, linestyle="--", color="#7f7f7f", linewidth=1.2, label="Target-only (few-shot)")
        if upper_bound > 0:
            ax.axhline(upper_bound, linestyle=(0, (4, 2)), color="#b22222", linewidth=1.2, label="Target upper bound")

        ax.set_title(domain, fontsize=12, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["T-U", "T-TP", "T-TR", "D-U", "D-TP", "D-TR"],
            fontsize=9,
        )
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)

    # Shared y label and limits
    axes[0].set_ylabel("Success Rate (%)", fontsize=11)
    max_y = max(upper_bound_global, max(b["mean"] + b["ci95"] for b in data["data"]))
    for ax in axes:
        ax.set_ylim(0, max(85, np.ceil(max_y / 5) * 5 + 5))

    # Build legends (pairing colors, coverage hatches)
    from matplotlib.patches import Patch

    pair_legend_handles = [
        Patch(facecolor=pairing_to_color[p], edgecolor="#333333", label=p)
        for p in pairings
    ]
    cov_legend_handles = [
        Patch(facecolor="#ffffff", edgecolor="#333333", hatch=hatch_by_coverage[c], label=c)
        for c in coverages
    ]

    # Figure-level legends above the subplots to avoid covering data
    fig.legend(
        handles=pair_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=9,
        title="Pairing",
        title_fontsize=9,
        frameon=False,
    )
    fig.legend(
        handles=cov_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=2,
        fontsize=9,
        title="Coverage",
        title_fontsize=9,
        frameon=False,
    )

    # Add reference line legend on the last axis (to avoid duplicates)
    ref_handles = []
    if target_only > 0:
        ref_handles.append(plt.Line2D([0], [0], color="#7f7f7f", lw=1.2, ls="--", label="Target-only (few-shot)"))
    if upper_bound > 0:
        ref_handles.append(plt.Line2D([0], [0], color="#b22222", lw=1.2, ls=(0, (4, 2)), label="Target upper bound"))
    if ref_handles:
        fig.legend(handles=ref_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=9, frameon=False)

    # Leave top margin for the figure-level legends
    plt.tight_layout(rect=(0, 0, 1, 0.82))
    return fig


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir, "figure1_coverage.json")
    data = load_data(data_path)
    fig = plot_figure(data)

    out_png = os.path.join(this_dir, "figure1_main_coverage.png")
    out_pdf = os.path.join(this_dir, "figure1_main_coverage.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved Figure 1 to: {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
