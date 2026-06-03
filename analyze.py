"""
Generate publication-style figures from experiment_results.csv.
Black-and-white, print-safe output.

Outputs (PNG @ 300 dpi + PDF vector for LaTeX):
    fig1_workflow   - experimental procedure (Methods section)
    fig2_regression - RTT vs download time + overall regression
    fig3_violin     - download time distribution by region (violin + jitter)
    fig4_cdf        - CDF of download times (log x-axis)
    summary_table.csv - statistics for paper tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Style — Black & White, LaTeX-ready (high DPI + PDF vector export)
# ---------------------------------------------------------------------------

SAVE_DPI = 300

sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": SAVE_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.0,
        "patch.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

REGION_ORDER  = ["France", "Canada", "Singapore"]
REGION_LABELS = {
    "France":    "Europe (France)",
    "Canada":    "North America (Canada)",
    "Singapore": "Asia (Singapore)",
}
REGION_COLORS  = {
    "France":    "black",
    "Canada":    "#444444",
    "Singapore": "#888888",
}
REGION_MARKERS = {"France": "o",   "Canada": "s",  "Singapore": "^"}
REGION_LINES   = {"France": "-",   "Canada": "--", "Singapore": ":"}
REGION_FILLS   = {
    "France":    "white",
    "Canada":    "#cccccc",
    "Singapore": "#888888",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    """Save PNG (300 dpi) and PDF (vector) — use .pdf in LaTeX for best quality."""
    kwargs = dict(bbox_inches="tight", pad_inches=0.12, facecolor="white", edgecolor="none")
    fig.savefig(outdir / f"{stem}.png", dpi=SAVE_DPI, **kwargs)
    fig.savefig(outdir / f"{stem}.pdf", **kwargs)
    plt.close(fig)
    print(f"  saved {stem}.png + {stem}.pdf")


def style_axes(ax: plt.Axes) -> None:
    """Ensure axis ticks, labels, and spines are crisp black for print/LaTeX."""
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        width=1.5,
        length=5,
        color="black",
        labelcolor="black",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")


def style_legend(leg: plt.Legend) -> None:
    frame = leg.get_frame()
    frame.set_linewidth(1.2)
    frame.set_edgecolor("black")
    frame.set_facecolor("white")
    for text in leg.get_texts():
        text.set_color("black")


def remove_outliers_iqr(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)


def load_and_clean(csv_path: Path):
    df = pd.read_csv(csv_path)
    mask_dl  = df.groupby("Region")["Download_s"].transform(remove_outliers_iqr)
    mask_rtt = df.groupby("Region")["RTT_ms"].transform(remove_outliers_iqr)
    df_clean = df[mask_dl & mask_rtt].reset_index(drop=True)
    removed  = len(df) - len(df_clean)
    return df, df_clean, removed


def print_statistics(df_clean: pd.DataFrame) -> None:
    r_all, p_all = stats.pearsonr(df_clean["RTT_ms"], df_clean["Download_s"])
    print("\n--- OVERALL PEARSON (RTT vs Download) ---")
    print(f"r = {r_all:.4f}, p = {p_all:.6f}")

    print("\n--- PER REGION ---")
    for region in REGION_ORDER:
        g = df_clean[df_clean["Region"] == region]
        if len(g) > 2:
            r, p = stats.pearsonr(g["RTT_ms"], g["Download_s"])
            print(f"  {region:12s}  n={len(g):2d}  r={r:+.3f}  p={p:.4f}")

    groups = [df_clean[df_clean["Region"] == r]["Download_s"].values
              for r in REGION_ORDER]
    f_stat, p_anova = stats.f_oneway(*groups)
    print("\n--- ONE-WAY ANOVA ---")
    print(f"F = {f_stat:.4f}, p = {p_anova:.6f}")


def save_summary_table(df_clean: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df_clean.groupby("Region")
        .agg(
            N=("Download_s", "count"),
            RTT_mean_ms=("RTT_ms", "mean"),
            RTT_std_ms=("RTT_ms", "std"),
            Download_mean_s=("Download_s", "mean"),
            Download_median_s=("Download_s", "median"),
            Download_std_s=("Download_s", "std"),
            Throughput_mean_Mbps=("Throughput_Mbps", "mean"),
            Throughput_std_Mbps=("Throughput_Mbps", "std"),
        )
        .reindex(REGION_ORDER)
        .round(3)
    )
    path = outdir / "summary_table.csv"
    summary.to_csv(path)
    print(f"\n--- SUMMARY TABLE -> {path} ---\n{summary}\n")


# ---------------------------------------------------------------------------
# Figure 1 — Workflow
# ---------------------------------------------------------------------------

def fig1_workflow(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # box width=2.2, gap=0.4 → starts: 0.2, 2.8, 5.4, 8.0
    boxes = [
        (0.2,  1.0, "Client\n(Windows + curl)"),
        (2.8,  1.0, "TCP connect\n(RTT measure)"),
        (5.4,  1.0, "HTTPS GET\n10 MB file"),
        (8.0,  1.0, "Log CSV\n30 samples\n× 3 regions"),
    ]
    BOX_W = 2.2
    for x, y, text in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), BOX_W, 1.0,
            boxstyle="round,pad=0.05",
            linewidth=1.5,
            edgecolor="black",
            facecolor="#f0f0f0",
        )
        ax.add_patch(rect)
        ax.text(x + BOX_W / 2, y + 0.5, text,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="black")

    # arrows: from right edge of box N to left edge of box N+1
    arrow_xs = [
        (0.2  + BOX_W, 2.8),   # box1 → box2
        (2.8  + BOX_W, 5.4),   # box2 → box3
        (5.4  + BOX_W, 8.0),   # box3 → box4
    ]
    for x_start, x_end in arrow_xs:
        ax.annotate("",
            xy=(x_end, 1.5), xytext=(x_start, 1.5),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="black"),
        )

    ax.text(6.0, 2.65,
            "OVH proof servers: France | Canada | Singapore",
            ha="center", fontsize=10, style="italic", color="black")
    save_figure(fig, outdir, "fig1_workflow")


# ---------------------------------------------------------------------------
# Figure 2 — Regression
# ---------------------------------------------------------------------------

def fig2_regression(df_clean: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for region in REGION_ORDER:
        g = df_clean[df_clean["Region"] == region]
        jitter = np.random.normal(0, 3, len(g)) if region == "France" \
                 else np.zeros(len(g))
        ax.scatter(
            g["RTT_ms"] + jitter, g["Download_s"],
            c=REGION_COLORS[region],
            marker=REGION_MARKERS[region],
            label=REGION_LABELS[region],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.6,
            s=50,
            zorder=3,
        )

    m, b, r, p, _ = stats.linregress(df_clean["RTT_ms"], df_clean["Download_s"])
    xs = np.linspace(df_clean["RTT_ms"].min(), df_clean["RTT_ms"].max(), 200)
    ax.plot(xs, m * xs + b,
            color="black", linewidth=2.0, linestyle="--",
            label=f"Overall regression (r={r:.3f}, p<0.001)", zorder=2)

    ax.set_xlabel("Round-trip time (RTT) [ms]")
    ax.set_ylabel("Download time [s]")
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.35, color="#666666", linewidth=0.8)
    style_axes(ax)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                    borderaxespad=0, frameon=True)
    style_legend(leg)

    save_figure(fig, outdir, "fig2_regression")


# ---------------------------------------------------------------------------
# Figure 3 — Violin + Jitter  (replaces plain box plot)
# ---------------------------------------------------------------------------

def fig3_violin(df_clean: pd.DataFrame, outdir: Path) -> None:
    """
    Violin plot with individual jitter points overlaid + IQR box + median marker.

    Why violin over plain box plot for n=30:
      - Shows the full shape of the distribution (skewness, modes).
      - Individual points remain visible so readers can judge the raw data.
      - IQR box + median diamond give the familiar summary statistics.
      - Outliers are obvious without needing a separate flier marker.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    np.random.seed(42)  # reproducible jitter

    data_by_region = [
        df_clean[df_clean["Region"] == r]["Download_s"].values
        for r in REGION_ORDER
    ]

    # --- violin bodies ---
    vp = ax.violinplot(
        data_by_region,
        positions=range(1, len(REGION_ORDER) + 1),
        widths=0.55,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, region in zip(vp["bodies"], REGION_ORDER):
        body.set_facecolor(REGION_FILLS[region])
        body.set_edgecolor(REGION_COLORS[region])
        body.set_linewidth(1.4)
        body.set_alpha(0.55)

    # --- IQR box + whiskers drawn manually (thin, inside violin) ---
    for i, (region, data) in enumerate(zip(REGION_ORDER, data_by_region)):
        pos = i + 1
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        lo  = max(data.min(), q1 - 1.5 * iqr)
        hi  = min(data.max(), q3 + 1.5 * iqr)

        # whisker line
        ax.vlines(pos, lo, hi,
                  color=REGION_COLORS[region], linewidth=1.2, zorder=3)
        # IQR box
        box = mpatches.FancyBboxPatch(
            (pos - 0.07, q1), 0.14, iqr,
            boxstyle="square,pad=0",
            linewidth=1.4,
            edgecolor=REGION_COLORS[region],
            facecolor="white",
            zorder=4,
        )
        ax.add_patch(box)
        # median diamond
        ax.scatter([pos], [med],
                   marker="D", s=28,
                   color=REGION_COLORS[region],
                   zorder=5, linewidths=0)

    # --- jitter points ---
    for i, (region, data) in enumerate(zip(REGION_ORDER, data_by_region)):
        pos   = i + 1
        jit_x = np.random.normal(0, 0.07, len(data))
        ax.scatter(
            np.full(len(data), pos) + jit_x, data,
            color=REGION_COLORS[region],
            marker=REGION_MARKERS[region],
            alpha=0.65,
            s=18,
            edgecolors="black",
            linewidths=0.4,
            zorder=6,
        )

    labels = [REGION_LABELS[r].replace(" (", "\n(") for r in REGION_ORDER]
    ax.set_xticks(range(1, len(REGION_ORDER) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Download time [s]")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, color="#666666", linewidth=0.8)
    style_axes(ax)

    # RTT boxes centred under each region, below tick labels (offset in points)
    avg_rtts = df_clean.groupby("Region")["RTT_ms"].mean()
    for i, region in enumerate(REGION_ORDER):
        ax.annotate(
            f"Avg RTT: {avg_rtts[region]:.0f} ms",
            xy=(i + 1, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -52),
            textcoords="offset points",
            ha="center", va="top",
            fontsize=9, fontweight="bold", color="black",
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="black", linewidth=1.2),
        )

    ax.set_xlabel("Geographical region", labelpad=42)
    fig.subplots_adjust(bottom=0.28)

    # --- legend for jitter markers ---
    legend_handles = [
        mpatches.Patch(
            facecolor=REGION_FILLS[r],
            edgecolor=REGION_COLORS[r],
            linewidth=1.2,
            label=REGION_LABELS[r],
        )
        for r in REGION_ORDER
    ]
    leg = ax.legend(handles=legend_handles, loc="upper left",
                    bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
    style_legend(leg)

    save_figure(fig, outdir, "fig3_violin")


# ---------------------------------------------------------------------------
# Figure 4 — CDF
# ---------------------------------------------------------------------------

def fig4_cdf(df_clean: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))

    for region in REGION_ORDER:
        sorted_dl = np.sort(
            df_clean.loc[df_clean["Region"] == region, "Download_s"])
        cdf = np.arange(1, len(sorted_dl) + 1) / len(sorted_dl)

        ax.plot(sorted_dl, cdf,
                color=REGION_COLORS[region],
                linestyle=REGION_LINES[region],
                linewidth=2.5,
                label=REGION_LABELS[region])

        ax.scatter(sorted_dl[::5], cdf[::5],
                   color=REGION_COLORS[region],
                   marker=REGION_MARKERS[region],
                   s=45, edgecolors="black",
                   linewidths=0.8, zorder=5)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([2, 3, 5, 7, 10, 15, 20, 25])
    ax.set_xlim(1.8, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Download time [s] (log scale)")
    ax.set_ylabel("Cumulative probability")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", alpha=0.35, color="#666666", linewidth=0.8)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.20, color="#888888", linewidth=0.6)
    style_axes(ax)
    leg = ax.legend(title="Region", bbox_to_anchor=(1.02, 1),
                    loc="upper left", frameon=True)
    style_legend(leg)
    leg.get_title().set_color("black")

    save_figure(fig, outdir, "fig4_cdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="experiment_results.csv")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--no-outlier-filter", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw, df_clean, removed = load_and_clean(csv_path)
    if args.no_outlier_filter:
        df_clean = df_raw.copy()
        removed  = 0

    print(f"Loaded {len(df_raw)} samples from {csv_path}")
    print(f"Outliers removed: {removed} -> {len(df_clean)} clean samples")

    print_statistics(df_clean)
    save_summary_table(df_clean, outdir)

    print("\nGenerating figures...")
    fig1_workflow(outdir)
    fig2_regression(df_clean, outdir)
    fig3_violin(df_clean, outdir)      # ← violin+jitter بدل box plot
    fig4_cdf(df_clean, outdir)
    print(f"\nDone. Figures saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

