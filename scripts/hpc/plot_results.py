#!/usr/bin/env python3
"""Figures and tables for the report from the CSV files written by run_all.sh and run_gpu.sh.

    python scripts/hpc/plot_results.py results/hpc/<run> [more run folders ...] [--out DIR]

Needs pandas and matplotlib. Several run folders are merged, so experiments may run separately.

- Every point is the MEDIAN over repetitions; whiskers show the fastest and slowest run. The
  median is used because the fastest run can reverse a comparison when the runs are noisy.
- Time per force evaluation = t_total / (steps + 1): a run does one initial force evaluation plus
  `steps` leapfrog steps, each with one force evaluation. Axes say "per force evaluation".
- Strong-scaling speedup and efficiency are measured against the serial Barnes-Hut program at
  the same N (median of all e7b_v1, e7c_v1 and ebase_v1 runs), so efficiency above 100 % is shown as it is.
  Hybrid runs (e7b: MPI ranks x 4 OpenMP threads) and pure-MPI runs (e7bm/e7cm: 1 thread per
  rank) are drawn in the same figure; "cores" = MPI ranks x OpenMP threads.
- Breakdowns use each phase's AVERAGE-rank time (*_mean); the rest of the slowest rank's total
  is shown as "imbalance + other". Slowest-rank times of different phases come from different
  ranks and cannot be stacked.
- E7a (ranks x threads on one node) uses only the newest run folder that has e7a.csv.
- Weak-scaling efficiency is measured against the 1-core MPI run. It is also shown per
  interaction, because the work per particle grows with N.
A figure is skipped when its CSV files are missing. Mixed code versions are reported as a warning.
Adapted from the team's scripts/plot.py.
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e3e2de", "#9b9a95"
# Each entity keeps one colour and marker in every figure.
STYLE = {"direct": (SLOT[3], "D"), "bh": (SLOT[0], "o"), "blocking": (SLOT[0], "o"), "overlap": (SLOT[1], "s"),
         "p2p": (SLOT[3], "D"), "cpu": (SLOT[0], "o"), "gpu": (SLOT[1], "s"), "insertion": (SLOT[0], "o"),
         "morton": (SLOT[1], "s"), "windows off": (SLOT[0], "o"), "windows on": (SLOT[2], "^"),
         "raw": (SLOT[0], "o"), "per interaction": (SLOT[1], "s")}
SIZE_SLOTS = [(SLOT[0], "o"), (SLOT[1], "s"), (SLOT[2], "^"), (SLOT[3], "D")]
PARTS = [("t_decomposition_mean", "decomposition"), ("t_migration_mean", "migration"),
         ("t_tree_build_mean", "tree build"), ("t_let_build_mean", "LET build"),
         ("t_let_exchange_mean", "LET exchange"), ("t_let_wait_mean", "MPI wait"),
         ("t_force_local_mean", "local traversal"), ("t_force_remote_mean", "remote traversal"),
         ("t_integrate_mean", "integrate")]
TEVAL = "time per force evaluation (s)"
CORES = "cores (MPI ranks × OpenMP threads)"
# Strong/weak scaling layouts: (file pattern, legend name, line style).
LAYOUTS = [("e7b_n*.csv", "hybrid, 4 threads/rank", "-"), ("e7bm_n*.csv", "pure MPI, 1 thread/rank", "--")]
WEAK_LAYOUTS = [("e7c_n*.csv", "hybrid, 4 threads/rank", "-"), ("e7cm_n*.csv", "pure MPI, 1 thread/rank", "--")]
V1 = ("e7b_v1.csv", "e7c_v1.csv", "ebase_v1.csv")  # serial Barnes-Hut runs

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 130,
})
FMT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")


def load(dirs, *patterns, newest=False):
    """Concatenate matching CSV files. newest=True: only the last folder (by name) that has a match."""
    if newest:
        have = [d for d in sorted(dirs, key=lambda d: os.path.basename(os.path.normpath(d)))
                if any(glob.glob(os.path.join(d, pat)) for pat in patterns)]
        dirs = have[-1:]
    frames = []
    for d in dirs:
        for pat in patterns:
            for f in sorted(glob.glob(os.path.join(d, pat))):
                frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    total = df["t_total_max"] if "t_total_max" in df else df["t_total"]
    if "t_total_max" in df and "t_total" in df:
        total = df["t_total_max"].fillna(df["t_total"])
    df["t_step"] = total / (df["steps"] + 1)
    if "mpi_ranks" in df:
        df["cores"] = df["mpi_ranks"] * df["omp_threads"]
    return df


def stats(df, keys, value="t_step"):
    """Median, fastest and slowest over repetitions."""
    g = df.groupby(keys, sort=True)[value]
    return g.agg(median="median", low="min", high="max", runs="count").reset_index()


def line(ax, s, x, label, key, whisker=True):
    col, mk = STYLE.get(key, key if isinstance(key, tuple) else (SLOT[6], "o"))
    y, lo, hi = s["median"].values, s["low"].values, s["high"].values
    err = np.array([y - lo, hi - y]) if whisker else None
    ax.errorbar(s[x], y, yerr=err, fmt=mk + "-", color=col, lw=2, ms=8, capsize=3, label=label,
                markeredgecolor="white", markeredgewidth=1)


def title(ax, xlabel, ylabel, text):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(text, loc="left", fontsize=10.5, color=INK)


def log2x(ax, values):
    """Log-2 axis with a tick at every value; a label is left out when it would touch the next
    larger one (e.g. 64 next to 72), so the largest value always keeps its label."""
    ticks = sorted(set(float(v) for v in values))
    keep, last = set(), None
    for v in reversed(ticks):
        if last is None or last / v >= 1.3:
            keep.add(v)
            last = v
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v in keep else ""))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def bars(ax, groups, series, values, styles):
    """Grouped bars: values[s][g] = (median, low, high)."""
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        xs = np.arange(len(groups)) + (i - (len(series) - 1) / 2) * w
        v = np.array([values[s].get(g, (np.nan, np.nan, np.nan)) for g in groups], dtype=float)
        ax.bar(xs, v[:, 0], width=w * 0.9, color=styles[s][0], label=s, edgecolor="white", linewidth=1.2,
               yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], ecolor=INK2, capsize=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.margins(y=0.2)


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def serial(dirs, out):
    d, b = load(dirs, "e1_direct.csv"), load(dirs, "e1_bh.csv")
    if d.empty and b.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for df, key, nm in ((d, "direct", "direct sum"), (b, "bh", "Barnes-Hut")):
        if df.empty:
            continue
        s = stats(df, ["n"])
        line(ax, s, "n", nm, key)
        s.to_csv(os.path.join(out, f"table_serial_{key}.csv"), index=False)
        if len(s) >= 3:
            top = s.tail(3)
            slope = np.polyfit(np.log(top["n"]), np.log(top["median"]), 1)[0]
            print(f"{nm}: growth over the largest 3 sizes = N^{slope:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    title(ax, "particles N", "time per force evaluation (s), 1 core", "Serial runtime against problem size")
    ax.legend(loc="upper left")
    save(fig, out, "fig_serial")


def theta(dirs, out):
    t, a = load(dirs, "e2_time.csv"), pd.DataFrame()
    files = [f for d in dirs for f in glob.glob(os.path.join(d, "e2_accuracy.csv"))]
    if files:
        a = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if t.empty or a.empty:
        return
    s = stats(t, ["theta"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    line(a1, s, "theta", "time", "bh")
    title(a1, "opening angle θ", TEVAL, "Cost against θ")
    a2.plot(a["theta"], a["accel_rms_rel"], "o-", color=SLOT[0], lw=2, ms=8, label="RMS")
    a2.plot(a["theta"], a["accel_p99_rel"], "s-", color=SLOT[1], lw=2, ms=8, label="99th percentile")
    a2.set_yscale("log")
    title(a2, "opening angle θ", "relative force error", "Force error against direct sum")
    a2.legend()
    save(fig, out, "fig_theta")


def treebuild(dirs, out):
    df = load(dirs, "e3.csv")
    if df.empty:
        return
    dists = sorted(df["dist"].unique())
    fig, axes = plt.subplots(1, len(dists), figsize=(4.8 * len(dists), 3.8), sharey=True, squeeze=False)
    for ax, dist in zip(axes[0], dists):
        for b in ("insertion", "morton"):
            s = stats(df[(df["dist"] == dist) & (df["tree_build"] == b)], ["omp_threads"], "t_tree_build")
            line(ax, s, "omp_threads", b, b)
        log2x(ax, df["omp_threads"])
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FMT)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        title(ax, "threads", "tree build time (s)", f"{dist}, N = {int(df['n'].iloc[0]):,}")
        ax.legend()
    save(fig, out, "fig_treebuild")


def schedules(dirs, out):
    df = load(dirs, "e4.csv")
    if df.empty:
        return
    df["schedule"] = df["sched"].astype(str) + ":" + df["chunk"].astype(int).astype(str)
    groups = list(dict.fromkeys(df["schedule"]))
    dists = sorted(df["dist"].unique())
    styles = {d: (SLOT[i], "o") for i, d in enumerate(dists)}
    vals = {d: {r["schedule"]: (r["median"], r["low"], r["high"]) for _, r in
                stats(df[df["dist"] == d], ["schedule"]).iterrows()} for d in dists}
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars(ax, groups, dists, vals, styles)
    title(ax, "OpenMP schedule:chunk", TEVAL, f"Thread schedules, N = {int(df['n'].iloc[0]):,}")
    ax.legend()
    save(fig, out, "fig_schedules")


def handle(container):
    """Legend entry for an errorbar: its data line only (caps would make a solid line look dashed)."""
    ln = container.lines[0]
    ln.set_label(container.get_label())
    return ln


def baseline(dirs):
    """Serial Barnes-Hut time per force evaluation at each N (median over all V1 runs)."""
    v1 = load(dirs, *V1)
    return v1.groupby("n")["t_step"].median() if not v1.empty else pd.Series(dtype=float)


def strong(dirs, out):
    base = baseline(dirs)
    data = [(load(dirs, pat), nm, ls) for pat, nm, ls in LAYOUTS]
    data = [(df, nm, ls) for df, nm, ls in data if not df.empty]
    if not data:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    tables, allcores, handles = [], [], []
    sizes = sorted(set().union(*[set(df["n"]) for df, _, _ in data]))
    for df, nm, ls in data:
        for n in sorted(df["n"].unique()):
            col, mk = SIZE_SLOTS[sizes.index(n) % len(SIZE_SLOTS)]
            s = stats(df[df["n"] == n], ["cores"])
            b = base.get(n, np.nan)
            if np.isnan(b):
                print(f"WARNING: no serial baseline at N={int(n)}; using the fewest cores")
                b = s["median"].iloc[0] * s["cores"].iloc[0]
            s["speedup"], s["speedup_low"], s["speedup_high"] = b / s["median"], b / s["high"], b / s["low"]
            s["efficiency_pct"] = 100 * s["speedup"] / s["cores"]
            s.insert(0, "layout", nm)
            s.insert(0, "n", int(n))
            tables.append(s)
            allcores += list(s["cores"])
            lab = f"N = {n / 1e6:g}M, {nm}"
            hollow = ls != "-"
            kw = dict(fmt=mk + ls, color=col, lw=2, ms=7, capsize=3, label=lab,
                      mfc="white" if hollow else col, markeredgecolor=col if hollow else "white",
                      markeredgewidth=1.6 if hollow else 1)
            a1.errorbar(s["cores"], s["speedup"], yerr=[s["speedup"] - s["speedup_low"],
                                                        s["speedup_high"] - s["speedup"]], **kw)
            handles.append(handle(a2.errorbar(s["cores"], s["efficiency_pct"],
                        yerr=[100 * (s["speedup"] - s["speedup_low"]) / s["cores"],
                              100 * (s["speedup_high"] - s["speedup"]) / s["cores"]], **kw)))
            tag = "mpi" if hollow else "hybrid"
            breakdown(df[df["n"] == n], out, f"fig_breakdown_strong_{tag}_{int(n / 1e6) if n >= 1e6 else int(n)}",
                      f"Strong scaling, N = {n / 1e6:g}M, {nm}: where the time goes")
    cmax = max(allcores)
    a1.plot([1, cmax], [1, cmax], ":", color=REF, lw=1.2, label="ideal")
    log2x(a1, allcores)
    log2x(a2, allcores)
    a1.set_yscale("log", base=2)
    a1.yaxis.set_major_formatter(FMT)
    a2.axhline(100, ls=":", color=REF, lw=1.2)
    a2.set_ylim(0, 115)
    title(a1, CORES, "speedup over serial Barnes-Hut", "Strong scaling")
    title(a2, CORES, "parallel efficiency (%)", "Parallel efficiency")
    a2.legend(handles=handles, handlelength=3.5, loc="lower left", fontsize=8)
    save(fig, out, "fig_strong")
    pd.concat(tables).to_csv(os.path.join(out, "table_strong.csv"), index=False)


def breakdown(df, out, name, text):
    """Average-rank time of each phase, and the rest of the slowest rank's total ("imbalance +
    other"), as shares of the slowest rank's total (median over repeats)."""
    med = df.groupby("cores")[[c for c, _ in PARTS] + ["t_total_max"]].median()
    parts = med[[c for c, _ in PARTS]]
    rest = (med["t_total_max"] - parts.sum(axis=1)).clip(lower=0)
    shares = 100 * parts.div(med["t_total_max"], axis=0)
    shares["rest"] = 100 * rest / med["t_total_max"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    bottom = np.zeros(len(shares))
    labels = [str(int(c)) for c in shares.index]
    colours = SLOT + ["#8c8b86", "#3d3c39"]
    for i, (c, nm) in enumerate(PARTS + [("rest", "imbalance + other")]):
        ax.bar(labels, shares[c], bottom=bottom, color=colours[i % len(colours)], label=nm,
               edgecolor="white", linewidth=1.2, width=0.7)
        bottom += shares[c].values
    title(ax, CORES, "share of slowest rank's time (%)", text)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    ax.set_ylim(0, 100)
    save(fig, out, name)


def weak(dirs, out):
    data = [(load(dirs, pat), nm, ls) for pat, nm, ls in WEAK_LAYOUTS]
    data = [(df, nm, ls) for df, nm, ls in data if not df.empty]
    if not data:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    tables, allcores, h1, h2 = [], [], [], []
    for i, (df, nm, ls) in enumerate(data):
        col, mk = SIZE_SLOTS[i]
        s = stats(df, ["cores"])
        work = df.assign(ipp=(df["interactions_pp"] + df["interactions_pc"]) / df["n"] / (df["steps"] + 1))
        s["n"] = df.groupby("cores")["n"].first().values
        s["interactions_per_particle"] = work.groupby("cores")["ipp"].median().values
        b = s.iloc[0]
        s["efficiency_pct"] = 100 * b["median"] / s["median"]
        s["efficiency_low_pct"] = 100 * b["median"] / s["high"]
        s["efficiency_high_pct"] = 100 * b["median"] / s["low"]
        s["efficiency_per_interaction_pct"] = (s["efficiency_pct"] * s["interactions_per_particle"]
                                               / b["interactions_per_particle"])
        s.insert(0, "layout", nm)
        tables.append(s)
        allcores += list(s["cores"])
        h1.append(handle(a1.errorbar(s["cores"], s["median"], yerr=[s["median"] - s["low"], s["high"] - s["median"]],
                              fmt=mk + ls, color=col, lw=2, ms=7, capsize=3, label=nm,
                              markeredgecolor="white")))
        h2.append(handle(a2.errorbar(s["cores"], s["efficiency_pct"], yerr=[s["efficiency_pct"] - s["efficiency_low_pct"],
                                                         s["efficiency_high_pct"] - s["efficiency_pct"]],
                    fmt=mk + ls, color=col, lw=2, ms=7, capsize=3, label=f"{nm}: time",
                    markeredgecolor="white")))
        h2 += a2.plot(s["cores"], s["efficiency_per_interaction_pct"], mk + ls, color=col, lw=1.2, ms=7,
                mfc="white", markeredgecolor=col, label=f"{nm}: time per interaction")
    per_core = int(data[0][0]["n"].min() / data[0][0]["cores"].min())
    pd.concat(tables).to_csv(os.path.join(out, "table_weak.csv"), index=False)
    for ax in (a1, a2):
        log2x(ax, allcores)
    a1.set_ylim(bottom=0)
    a2.axhline(100, ls=":", color=REF, lw=1.2)
    a2.set_ylim(0, 115)
    title(a1, CORES + ", N grows with cores", TEVAL, f"Weak scaling, {per_core:,} particles per core")
    title(a2, CORES + ", N grows with cores", "efficiency against 1 core (%)", "Weak-scaling efficiency")
    a1.legend(handles=h1, handlelength=3.5, loc="lower right", fontsize=8.5)
    a2.legend(handles=h2, handlelength=3.5, loc="lower left", fontsize=7.5)
    save(fig, out, "fig_weak")
    for df, nm, ls in data:
        breakdown(df, out, "fig_breakdown_weak_" + ("hybrid" if ls == "-" else "mpi"),
                  f"Weak scaling, {nm}: where the time goes")


def comm(dirs, out):
    df = load(dirs, "e5a_*_n*.csv")
    if df.empty:
        return
    modes = [m for m in ("blocking", "p2p", "overlap") if m in set(df["comm"])]
    cores = sorted(df["cores"].unique())
    groups = [str(int(c)) for c in cores]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, value, ylabel in ((a1, "t_step", TEVAL), (a2, "t_let_wait_max", "MPI wait, slowest rank (s)")):
        vals = {m: {str(int(r["cores"])): (r["median"], r["low"], r["high"]) for _, r in
                    stats(df[df["comm"] == m], ["cores"], value).iterrows()} for m in modes}
        bars(ax, groups, modes, vals, STYLE)
        title(ax, "cores", ylabel, f"LET exchange modes, N = {int(df['n'].iloc[0]):,}")
    a1.legend()
    save(fig, out, "fig_comm")


def shm(dirs, out):
    df = load(dirs, "e5b_shm*_n*.csv")
    if df.empty:
        return
    df["mode"] = np.where(df["shm"] == 1, "windows on", "windows off")
    modes = ["windows off", "windows on"]
    groups = [str(int(c)) for c in sorted(df["cores"].unique())]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    df["let_mb"] = df["let_bytes_received_max"] / 1e6
    for ax, value, ylabel in ((a1, "t_step", TEVAL), (a2, "let_mb", "LET received, largest rank (MB)")):
        vals = {m: {str(int(r["cores"])): (r["median"], r["low"], r["high"]) for _, r in
                    stats(df[df["mode"] == m], ["cores"], value).iterrows()} for m in modes}
        bars(ax, groups, modes, vals, STYLE)
        title(ax, "cores, one thread per rank", ylabel, f"Shared-memory windows, N = {int(df['n'].iloc[0]):,}")
    a1.legend()
    save(fig, out, "fig_shm")


def hybrid(dirs, out):
    frames = [("e7a.csv", "fig_hybrid_node"), ("e7d_n*.csv", "fig_hybrid_full")]
    base = baseline(dirs)
    for pat, name in frames:
        df = load(dirs, pat, newest=(pat == "e7a.csv"))
        if df.empty:
            continue
        s = stats(df, ["mpi_ranks", "omp_threads"]).sort_values("omp_threads")
        labels = [f"{int(r)} × {int(t)}" for r, t in zip(s["mpi_ranks"], s["omp_threads"])]
        fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(labels)), 3.6))
        ax.bar(labels, s["median"], color=SLOT[0], edgecolor="white", linewidth=1.2, width=0.7,
               yerr=[s["median"] - s["low"], s["high"] - s["median"]], ecolor=INK2, capsize=3)
        ax.tick_params(axis="x", labelrotation=30 if len(labels) > 6 else 0)
        b = base.get(df["n"].iloc[0], np.nan)
        if not np.isnan(b):  # speedup over serial Barnes-Hut above every bar
            for x, (m, hi) in enumerate(zip(s["median"], s["high"])):
                ax.text(x, hi, f"{b / m:.0f}×", ha="center", va="bottom", fontsize=8, color=INK2)
            ax.margins(y=0.12)
        cores = int(df["cores"].iloc[0])
        title(ax, "MPI ranks × OpenMP threads", TEVAL,
              f"Ranks × threads on {cores} cores, N = {int(df['n'].iloc[0]):,} (labels: speedup over serial)")
        s.to_csv(os.path.join(out, f"table_{name[4:]}.csv"), index=False)
        save(fig, out, name)


def gpu(dirs, out):
    df = load(dirs, "gpu.csv")
    if df.empty:
        return
    # A row counts as GPU only if the program reports that it ran there.
    if ((df["device"] == "gpu") & (df["device_executed"] != "gpu")).any():
        print("WARNING: some GPU rows ran on the host (device_executed != gpu); not GPU results")
    df["config"] = df["mpi_ranks"].astype(int).astype(str) + " rank(s) × " + df["omp_threads"].astype(int).astype(str) + " threads"
    configs = sorted(df["config"].unique())
    fig, axes = plt.subplots(1, len(configs), figsize=(4.9 * len(configs), 3.8), sharey=True, squeeze=False)
    tables = []
    for ax, c in zip(axes[0], configs):
        for dev, nm in (("cpu", "CPU"), ("gpu", "GPU")):
            s = stats(df[(df["config"] == c) & (df["device"] == dev)], ["n"])
            line(ax, s, "n", nm, dev)
            tables.append(s.assign(config=c, device=dev))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sorted(df["n"].unique()))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(FMT)
        ax.yaxis.set_minor_formatter(FMT)
        title(ax, "particles N", TEVAL, c)
        ax.legend()
    pd.concat(tables).to_csv(os.path.join(out, "table_gpu.csv"), index=False)
    save(fig, out, "fig_gpu")


def long_run(dirs, out):
    files = [f for d in dirs for f in sorted(glob.glob(os.path.join(d, "e9_diag_n*.csv")))]
    if not files:
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for i, (dt, x) in enumerate(sorted(d.groupby("dt"), key=lambda kv: -kv[0])):
        col, mk = SIZE_SLOTS[i % len(SIZE_SLOTS)]
        x = x[x["step"] > 0].sort_values("time")
        lab = f"dt = {dt:g} ({int(x['step'].max())} steps)"
        a1.plot(x["time"], x["rel_energy_drift"].abs(), "-", color=col, lw=1.8, label=lab)
        a2.plot(x["time"], x["rel_momentum_drift"].abs(), "-", color=col, lw=1.8, label=lab)
    r = d.iloc[0]
    for ax, ylab, txt in ((a1, "|E(t) - E(0)| / |E(0)|", "Energy error"),
                          (a2, "|P(t) - P(0)| / sum m|v|", "Momentum error")):
        ax.set_yscale("log")
        title(ax, "simulated time", ylab, f"{txt}, N = {int(r['n']):,}, {int(r['ranks'])} ranks × {int(r['threads'])} threads")
        ax.legend(fontsize=8.5)
    save(fig, out, "fig_long_run")


def load_balance(dirs, out):
    df = load(dirs, "e10_n*.csv")
    if df.empty:
        return
    df["bal"] = np.where(df["rebalance_every"] == 1, "work-weighted", "equal count")
    df["group"] = df["dist"] + "\n" + df["mpi_ranks"].astype(int).astype(str) + " × " + df["omp_threads"].astype(int).astype(str)
    groups = sorted(df["group"].unique())
    series = ["equal count", "work-weighted"]
    styles = {"equal count": (SLOT[3], "D"), "work-weighted": (SLOT[0], "o")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, value, ylab in ((a1, "t_step", TEVAL), (a2, "work_imbalance_last_step", "work imbalance (max / mean rank)")):
        vals = {b: {r["group"]: (r["median"], r["low"], r["high"]) for _, r in
                    stats(df[df["bal"] == b], ["group"], value).iterrows()} for b in series}
        bars(ax, groups, series, vals, styles)
        title(ax, "distribution, MPI ranks × OpenMP threads", ylab, f"Load balancing, N = {int(df['n'].iloc[0]):,}")
    a2.set_ylim(bottom=0.9)
    a2.axhline(1.0, ls=":", color=REF, lw=1.2)  # 1 = perfectly balanced
    a1.legend(fontsize=8.5)
    save(fig, out, "fig_load_balance")


def letboxes(dirs, out):
    df = load(dirs, "e11_n*.csv")
    if df.empty:
        return
    df["let_mb"] = df["let_bytes_received_max"] / 1e6
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for i, (c, x) in enumerate(df.groupby("cores")):
        col, mk = SIZE_SLOTS[i % len(SIZE_SLOTS)]
        lab = f"{int(c)} ranks × 1 thread"
        line(a1, stats(x, ["let_boxes"], "let_mb"), "let_boxes", lab, (col, mk))
        line(a2, stats(x, ["let_boxes"]), "let_boxes", lab, (col, mk))
    for ax in (a1, a2):
        log2x(ax, df["let_boxes"])
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    a1.yaxis.set_major_formatter(FMT)
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_ylim(bottom=0)
    n = f"N = {int(df['n'].iloc[0]):,}"
    title(a1, "LET receiver boxes per rank K", "LET received, largest rank (MB)", f"LET size against K, {n}")
    title(a2, "LET receiver boxes per rank K", TEVAL, f"Run time against K, {n}")
    a1.legend(fontsize=8.5)
    save(fig, out, "fig_letboxes")


def check_versions(dirs):
    versions = set()
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.csv")):
            df = pd.read_csv(f)
            if "git_hash" in df:
                versions |= set(df["git_hash"].astype(str))
    print("code versions in these results:", sorted(versions))
    if len(versions) > 1 or "unknown" in versions:
        print("WARNING: figures should come from one known commit")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders under results/hpc")
    ap.add_argument("--out", help="output folder (default: <first run>/figures)")
    a = ap.parse_args()
    for d in a.runs:
        if not os.path.isdir(d):
            sys.exit(f"no such folder: {d}")
    out = a.out or os.path.join(a.runs[0], "figures")
    os.makedirs(out, exist_ok=True)
    check_versions(a.runs)
    for f in (serial, theta, treebuild, schedules, strong, weak, comm, shm, hybrid, gpu, long_run, load_balance,
              letboxes):
        f(a.runs, out)


if __name__ == "__main__":
    main()
