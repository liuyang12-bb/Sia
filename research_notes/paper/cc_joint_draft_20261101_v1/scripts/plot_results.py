"""Rebuild manuscript figures from the immutable, extracted evidence JSON.

No simulator or scheduler imports, no new experiments, no fitted response model.
Run with the existing Sia Python environment; output PNG(300 dpi), SVG and PDF.
"""
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "evidence" / "paper_results.json"
DATA = json.loads(DATA_PATH.read_text(encoding="utf8"))
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "savefig.facecolor": "white", "svg.fonttype": "none",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
RED = "#D55E00"
GRAY = "#777777"
LIGHT = "#B8B8B8"
METRIC_COLORS = {"wait": BLUE, "training": ORANGE, "checkpoint": GREEN}
SYNTHETIC = "Synthetic cross-hardware transfer; not real Saturn performance"


def row(dataset, cohort, method):
    return next(r for r in DATA["cells"] if
                (r["dataset"], r["cohort"], r["method"]) == (dataset, cohort, method))


def improvement(baseline, new, metric="mean_jct"):
    return 100.0 * (baseline[metric] - new[metric]) / baseline[metric]


def style(ax, axis="y"):
    ax.grid(axis=axis, color="#DDDDDD", linewidth=.6)


def save(fig, stem, footer):
    # Long methodological disclaimers belong in the manuscript caption, not
    # inside a figure that will be printed at a single-column width of 170 mm.
    fig.tight_layout(pad=1.2, h_pad=2.2, w_pad=1.7)
    for ext in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def service():
    methods = ["A00", "A01", "A10", "A11", "A_SIA"]
    rs = [row("clean_ab", "A20", m) for m in methods]
    labels = ["A00", "A01", "A10", "A11", "SIA-P"]
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 4.6))
    x = np.arange(len(rs))
    vals = [r["mean_jct"] / 3600 for r in rs]
    bars = axs[0].bar(x, vals, color=[LIGHT, LIGHT, ORANGE, BLUE, GRAY], width=.64,
                       edgecolor="white", linewidth=.6)
    axs[0].bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    axs[0].set_ylim(0, max(vals) * 1.18)
    axs[0].set_xticks(x, labels)
    axs[0].set_ylabel("Mean JCT (hours)")
    axs[0].set_title("(a) Service ablation\n20 jobs, Neutral")
    style(axs[0])
    bottom = np.zeros(len(rs))
    for metric, label in [("wait", "Waiting"), ("training", "Productive training"),
                          ("checkpoint", "Startup / checkpoint")]:
        values = np.array([r["totals"][metric] / r["total"] / 3600 for r in rs])
        axs[1].bar(x, values, bottom=bottom, label=label,
                   color=METRIC_COLORS[metric], width=.64, edgecolor="white", linewidth=.5)
        bottom += values
    axs[1].set_xticks(x, labels)
    axs[1].set_ylabel("Time per job (hours)")
    axs[1].set_title("(b) Mutually exclusive\nJCT components")
    axs[1].set_ylim(0, max(bottom) * 1.50)
    axs[1].legend(loc="upper right", frameon=False, fontsize=9.5)
    style(axs[1])
    save(fig, "fig02_service_ablation",
         "20 jobs, Neutral, common fairness score; Axy: x = FIFO, y = continuity. " + SYNTHETIC)


def resources():
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    cohorts = ["BURST", "COMPACT", "SPREAD"]
    methods = ["CC_P", "CC_CPU", "CC_MEM", "CC_FULL"]
    x = np.arange(3)
    width = .19
    for i, (method, label, color) in enumerate(zip(methods,
            ["Proportional", "CPU only", "RAM only", "CPU + RAM"], [GRAY, GREEN, PURPLE, BLUE])):
        values = [row("arrival18", co, method)["mean_jct"] /
                  row("arrival18", co, "CC_P")["mean_jct"] for co in cohorts]
        bars = axs[0].bar(x + (i - 1.5) * width, values, width, label=label,
                           color=color, edgecolor="white", linewidth=.5)
        axs[0].bar_label(bars, fmt="%.2f", fontsize=9, padding=3, rotation=90)
    axs[0].axhline(1, color="#444444", linestyle="--", linewidth=.8)
    axs[0].set_xticks(x, cohorts)
    axs[0].set_ylim(0, 1.43)
    axs[0].set_ylabel("Mean JCT / proportional")
    axs[0].set_title("(a) CPU / RAM ablation\n12 jobs, Medium")
    axs[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=9,
                  columnspacing=.8, handlelength=1.1)
    style(axs[0])
    cohorts2 = ["COMPACT", "SPREAD"]
    xx = np.arange(2)
    for i, (method, label, color) in enumerate([("B_FO", "Serial FO", ORANGE), ("B_JO", "Joint JO", BLUE)]):
        values = [row("clean_ab", co, method)["mean_jct"] /
                  row("clean_ab", co, "B_FO")["mean_jct"] for co in cohorts2]
        bars = axs[1].bar(xx + (i - .5) * .30, values, .30,
                           color=color, label=label, edgecolor="white", linewidth=.5)
        axs[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    axs[1].set_xticks(xx, cohorts2)
    axs[1].set_ylim(0, 1.43)
    axs[1].set_ylabel("Mean JCT / serial FO")
    axs[1].set_title("(b) Joint versus serial\nClean control")
    axs[1].legend(loc="upper center", ncol=1, frameon=False, fontsize=10)
    axs[1].axhline(1, color="#444444", linestyle="--", linewidth=.8)
    style(axs[1])
    save(fig, "fig03_resource_ablation",
         "Panels use distinct frozen versions; compare within each panel. " + SYNTHETIC)


def core80():
    fig = plt.figure(figsize=(8.0, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    axs = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
           fig.add_subplot(grid[1, :])]
    methods = ["SIA_P", "SIA_AUX", "PROP", "CPU", "MEM", "FO", "FULL"]
    rs = [row("core80", "DUAL80", m) for m in methods]
    vals = [r["mean_jct"] / 3600 for r in rs]
    bars = axs[0].bar(np.arange(7), vals, color=[GRAY, LIGHT, "#555555", GREEN, PURPLE, ORANGE, BLUE],
                       edgecolor="white", linewidth=.5)
    axs[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axs[0].set_xticks(np.arange(7), [m.replace("_", "-") for m in methods], rotation=45, ha="right")
    axs[0].set_ylabel("Mean JCT (hours)")
    axs[0].set_ylim(0, max(vals) * 1.18)
    axs[0].set_title("(a) Mean JCT\n80 completed jobs")
    style(axs[0])
    for m, color, ls in [("SIA_P", GRAY, "--"), ("FO", ORANGE, "-."), ("FULL", BLUE, "-")]:
        r = row("core80", "DUAL80", m)
        values = np.sort([j["jct"] / 3600 for j in r["jobs"].values()])
        xs = np.r_[0, values]
        ys = np.r_[0, np.arange(1, len(values) + 1) / len(values)]
        axs[1].step(xs, ys, where="post", label=m.replace("_", "-"), color=color, linestyle=ls, linewidth=1.7)
    axs[1].set_xlim(left=0)
    axs[1].set_ylim(0, 1.02)
    axs[1].set_xlabel("Per-job JCT (hours)")
    axs[1].set_ylabel("Empirical CDF")
    axs[1].set_title("(b) JCT distribution\nEmpirical CDF")
    axs[1].legend(frameon=False, loc="lower right")
    style(axs[1])
    full = row("core80", "DUAL80", "FULL")
    metrics = ["mean_jct", "p95_jct", "p99_jct", "makespan"]
    y = np.arange(4)
    for i, (baseline, color) in enumerate([("SIA_P", GRAY), ("FO", ORANGE)]):
        vals = [improvement(row("core80", "DUAL80", baseline), full, metric) for metric in metrics]
        bars = axs[2].barh(y + (i - .5) * .32, vals, height=.32,
                            label="vs " + baseline.replace("_", "-"), color=color)
        for b, v in zip(bars, vals):
            axs[2].text(v + (0.8 if v >= 0 else -0.8), b.get_y() + b.get_height()/2,
                        f"{v:+.1f}%", ha="left" if v >= 0 else "right", va="center", fontsize=11)
    axs[2].set_yticks(y, ["Mean", "P95", "P99", "Makespan"])
    axs[2].invert_yaxis()
    axs[2].set_xlim(-31, 40)
    axs[2].axvline(0, color="#333333", linewidth=.9)
    axs[2].set_xlabel("FULL improvement (%)")
    axs[2].set_title("(c) FULL improvement: positive is better")
    axs[2].legend(frameon=False, loc="lower right", fontsize=10)
    style(axs[2], "x")
    save(fig, "fig04_core80", "One controlled 80-job workload; empirical tails, no independent-replication CI. " + SYNTHETIC)


def tradeoffs():
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 7.8))
    old = row("mixed12", "MIXED12", "FO")
    new = row("mixed12", "MIXED12", "FULL")
    names = list(old["jobs"])
    saved = [old["jobs"][n]["jct"] - new["jobs"][n]["jct"] for n in names]
    bars = axs[0].bar(np.arange(len(names)), saved, color=[BLUE if v > 0 else RED for v in saved], width=.68)
    for b, v in zip(bars, saved):
        axs[0].text(b.get_x()+b.get_width()/2, v + (35 if v >= 0 else -35),
                    f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    labelmap = {"cifar10": "C", "deepspeech2": "D", "bert": "B"}
    labels = [labelmap[n.rsplit("-", 1)[0]] + n.rsplit("-", 1)[1] for n in names]
    axs[0].set_xticks(np.arange(len(names)), labels)
    axs[0].axhline(0, color="#333333", linewidth=.9)
    axs[0].set_ylim(min(saved) - 240, max(saved) + 300)
    axs[0].set_ylabel("JCT saved: FO - FULL (seconds)")
    axs[0].set_xlabel("Fixed job order: C = CIFAR10, D = DeepSpeech2, B = BERT")
    axs[0].set_title("(a) Latest 12-job pair: 6 improve, 6 worsen")
    style(axs[0])
    pairs = [("clean_ab", "A20", "A_SIA", "A11"), ("core80", "DUAL80", "SIA_P", "FULL"),
             ("mixed12", "MIXED12", "FO", "FULL")]
    x = np.arange(3)
    for i, (key, label) in enumerate([("wait", "Waiting"), ("training", "Productive training"),
                                    ("checkpoint", "Startup / checkpoint")]):
        values=[]
        for dataset, cohort, baseline, method in pairs:
            b=row(dataset,cohort,baseline);n=row(dataset,cohort,method)
            values.append((b['totals'][key]-n['totals'][key])/n['total'])
        bars = axs[1].bar(x+(i-1)*.24, values, width=.24, color=METRIC_COLORS[key], label=label)
    axs[1].axhline(0, color="#333333", linewidth=.9)
    axs[1].set_xticks(x, ["A11 vs Sia\n20 jobs", "FULL vs Sia\n80 jobs", "FULL vs FO\n12 jobs"])
    axs[1].set_ylabel("Time saved per job (seconds)")
    axs[1].set_title("(b) Queue savings can offset slower training")
    axs[1].set_ylim(-1400, 6000)
    axs[1].legend(frameon=False, loc="upper right", fontsize=10)
    style(axs[1])
    save(fig, "fig05_tradeoffs",
         "Different frozen versions / workloads in (b); no pooled estimate. " + SYNTHETIC)


if __name__ == "__main__":
    service()
    resources()
    core80()
    tradeoffs()
    manifest={"data_file":str(DATA_PATH),"data_sha256":hashlib.sha256(DATA_PATH.read_bytes()).hexdigest().upper(),
              "matplotlib_version":matplotlib.__version__,"numpy_version":np.__version__,
              "note":"Offline plots only; data extraction preserves negative and heterogeneous-version results.",
              "outputs":{p.name:hashlib.sha256(p.read_bytes()).hexdigest().upper()
                         for p in sorted(OUT.glob('fig0[2-5]*.*')) if p.suffix in ['.png','.svg','.pdf']}}
    (OUT/'plot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({"figures":len(manifest['outputs']),"data_sha256":manifest['data_sha256']},indent=2))
