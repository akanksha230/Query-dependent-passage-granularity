"""Regenerate the main results figure from results/results_test.csv.

Run from the project root:
    python src/plot_results.py

The figure is saved to figures/fig_main.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Support both layouts:
#   project_root/plot_results.py
#   project_root/src/plot_results.py
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_CANDIDATES = (SCRIPT_DIR, SCRIPT_DIR.parent)
ROOT = next(
    (candidate for candidate in ROOT_CANDIDATES
     if (candidate / "results" / "results_test.csv").is_file()),
    SCRIPT_DIR,
)
RESULTS = ROOT / "results" / "results_test.csv"
OUTPUT = ROOT / "figures" / "fig_main.png"


def get_row(data: pd.DataFrame, budget: int, system: str) -> pd.Series:
    rows = data[(data["B"] == budget) & (data["system"] == system)]
    if rows.empty:
        raise ValueError(f"Missing results row: B={budget}, system={system}")
    return rows.iloc[0]


def main() -> None:
    if not RESULTS.is_file():
        raise FileNotFoundError(
            "Could not find results/results_test.csv. Put this script either "
            "in the project root or in its src/ folder, and keep the results/ "
            "folder alongside the project root."
        )
    data = pd.read_csv(RESULTS)
    budget = 512

    main_systems = [
        ("fixed_sent", "Fixed sentence", "fixed"),
        ("fixed_p128", "Fixed 128", "fixed"),
        ("fixed_p256", "Fixed 256", "fixed"),
        ("fixed_p512", "Fixed 512", "fixed"),
        ("fixed_section", "Fixed section", "fixed"),
        ("fixed_para", "Fixed paragraph", "fixed"),
        ("c2f_s1", "Coarse-to-fine\n(section→sentence)", "coarse"),
        ("mog_style_query_only", "Query-only\nrouter (MoG-style)", "router"),
        ("query+ret_p256", "Query+retrieval\n(p256 signal)", "retrieval"),
        ("query+ret_all", "Query+retrieval\n(all signals)", "retrieval"),
        ("oracle", "Oracle", "oracle"),
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 5.7), gridspec_kw={"width_ratios": [1.75, 1]}
    )

    palette = {
        "fixed": "#9aa6b2",
        "coarse": "#e2a452",
        "router": "#3f70b5",
        "retrieval": "#2a9d8f",
        "oracle": "#c9c9c9",
    }
    values, lower, upper, labels, colors = [], [], [], [], []
    for system, label, group in main_systems:
        row = get_row(data, budget, system)
        values.append(row["coverage"])
        lower.append(row["coverage"] - row["cov_lo"])
        upper.append(row["cov_hi"] - row["coverage"])
        labels.append(label)
        colors.append(palette[group])

    x = np.arange(len(values))
    ax1.bar(x, values, color=colors, width=0.8, zorder=2)
    ax1.errorbar(
        x, values, yerr=np.array([lower, upper]), fmt="none",
        ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=3,
    )
    baseline = get_row(data, budget, "fixed_para")["coverage"]
    ax1.axhline(baseline, color="black", linestyle="--", linewidth=1)
    ax1.set_title("QASPER test, matched 512-word budget (95% paper-clustered CI)")
    ax1.set_ylabel("Evidence coverage @ 512 words")
    ax1.set_ylim(0.3, 0.82)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")

    sensitivity_systems = [
        ("fixed_para", "Fixed paragraph", "#9aa6b2"),
        ("mog_style_query_only", "Query-only router", "#3f70b5"),
        ("query+ret_all", "Query+retrieval (all)", "#2a9d8f"),
        ("query+ret_p256", "Query+retrieval (p256)", "#7bc8bd"),
    ]
    budgets = [256, 512, 1024]
    group_x = np.arange(len(budgets))
    width = 0.18
    for idx, (system, label, color) in enumerate(sensitivity_systems):
        vals = [get_row(data, b, system)["coverage"] for b in budgets]
        ax2.bar(group_x + (idx - 1.5) * width, vals, width, label=label, color=color)

    ax2.set_title("Budget sensitivity")
    ax2.set_xlabel("Word budget")
    ax2.set_ylim(0.35, 0.8)
    ax2.set_xticks(group_x)
    ax2.set_xticklabels([str(b) for b in budgets])
    ax2.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT}")


if __name__ == "__main__":
    main()
