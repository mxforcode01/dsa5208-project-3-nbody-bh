#!/usr/bin/env python3
"""Comparison tables (Excel + CSV) from the CSV files written by run_all.sh and run_gpu.sh.

    python scripts/hpc/make_tables.py <run folder> [more run folders ...] --out DIR

Uses the same rules as plot_results.py: every value is the MEDIAN over repeats; "s/step" in the
CSV files and "s per force eval" in the Excel sheets are the same thing, time per force evaluation =
total time / (steps + 1) (one initial evaluation plus one per step); per-phase times are the
average rank's (see PHASES). E7a uses only the newest run folder that has e7a.csv. Writes
results_tables.xlsx (one sheet per table) and one CSV per table into DIR.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import LAYOUTS, V1, WEAK_LAYOUTS, load  # noqa: E402

# Phase times are the AVERAGE rank's time (the *_mean columns), so the phases add up to about the
# total; the slowest-rank (*_max) times of different phases come from different ranks and do not.
PHASES = [("t_decomposition_mean", "decomposition"), ("t_migration_mean", "migration"),
          ("t_tree_build_mean", "tree build"), ("t_let_build_mean", "LET build"),
          ("t_let_exchange_mean", "LET exchange"), ("t_let_wait_mean", "MPI wait"),
          ("t_force_local_mean", "local force walk"), ("t_force_remote_mean", "remote force walk"),
          ("t_integrate_mean", "integrate")]
# Communication = decomposition + LET exchange + MPI wait (average rank) as a share of the total time.
COMM = ["t_decomposition_mean", "t_let_exchange_mean", "t_let_wait_mean"]


def r(x, d=3):
    return float(np.round(x, d)) if pd.notna(x) else np.nan


def med(df, keys, col):
    return df.groupby(keys)[col].median()


def per_step(df, col):
    return df[col] / (df["steps"] + 1)


def basic(df, keys):
    g = df.groupby(keys)["t_step"]
    return g.agg(median="median", fastest="min", slowest="max", repeats="count")


def add_mpi_cols(df):
    df = df.copy()
    df["ipp"] = (df["interactions_pp"] + df["interactions_pc"]) / df["n"] / (df["steps"] + 1)
    df["comm_s"] = df[COMM].sum(axis=1) / (df["steps"] + 1)
    df["comm_pct"] = 100 * df[COMM].sum(axis=1) / df["t_total_max"]
    df["wait_s"] = per_step(df, "t_let_wait_mean")
    df["let_mb_max"] = df["let_bytes_received_max"] / 1e6
    df["let_mb_mean"] = df["let_bytes_received_mean"] / 1e6
    df["mem_mb_max"] = df["maxrss_kb_max"] / 1024
    df["particles_spread_pct"] = 100 * (df["particles_max"] / (df["n"] / df["mpi_ranks"]) - 1)
    df["split"] = df["mpi_ranks"].astype(int).astype(str) + " x " + df["omp_threads"].astype(int).astype(str)
    df["nodes"] = np.ceil(df["cores"] / 72).astype(int)
    return df


def t_serial(dirs):
    d, b = load(dirs, "e1_direct.csv"), load(dirs, "e1_bh.csv")
    for x in (d, b):
        x["ipp"] = (x["interactions_pp"] + x["interactions_pc"]) / x["n"] / (x["steps"] + 1)
    sd, sb = basic(d, "n"), basic(b, "n")
    t = pd.DataFrame({"N": sorted(set(sd.index) | set(sb.index))}).set_index("N")
    t["direct sum: s/step"] = sd["median"]
    t["Barnes-Hut: s/step"] = sb["median"]
    t["Barnes-Hut faster by (x)"] = t["direct sum: s/step"] / t["Barnes-Hut: s/step"]
    t["direct: interactions/particle"] = med(d, "n", "ipp")
    t["BH: interactions/particle"] = med(b, "n", "ipp")
    t["repeats"] = sb["repeats"]
    return t.reset_index(), "E1 serial: direct sum vs Barnes-Hut, 1 core, Plummer"


def t_theta(dirs):
    t, a = load(dirs, "e2_time.csv"), pd.concat([pd.read_csv(f) for d in dirs
                                                  for f in glob.glob(os.path.join(d, "e2_accuracy.csv"))])
    t["ipp"] = (t["interactions_pp"] + t["interactions_pc"]) / t["n"] / (t["steps"] + 1)
    t["theta"], a["theta"] = t["theta"].round(3), a["theta"].round(3)
    a = a.groupby("theta").median(numeric_only=True)
    s = basic(t, "theta")
    out = pd.DataFrame({"theta": s.index, "s/step (1 core)": s["median"].values,
                        "interactions/particle": med(t, "theta", "ipp").values}).set_index("theta")
    out["force error RMS (%)"] = 100 * a["accel_rms_rel"]
    out["force error median (%)"] = 100 * a["accel_median_rel"]
    out["force error 99th pct (%)"] = 100 * a["accel_p99_rel"]
    out["force error max (%)"] = 100 * a["accel_max_rel"]
    out["repeats"] = s["repeats"]
    n = int(t["n"].iloc[0])
    return out.reset_index(), f"E2 opening angle theta: cost and force error vs direct sum, N = {n:,} Plummer"


def t_treebuild(dirs):
    df = load(dirs, "e3.csv")
    rows = []
    for dist in sorted(df["dist"].unique()):
        x = df[df["dist"] == dist]
        ins = med(x[x["tree_build"] == "insertion"], "omp_threads", "t_tree_build")
        mor = med(x[x["tree_build"] == "morton"], "omp_threads", "t_tree_build")
        for th in sorted(x["omp_threads"].unique()):
            rows.append({"distribution": dist, "threads": int(th), "insertion build (s)": ins.get(th),
                         "Morton build (s)": mor.get(th),
                         "Morton speedup vs its 1 thread": mor.iloc[0] / mor.get(th),
                         "Morton faster than insertion (x)": ins.get(th) / mor.get(th)})
    n = int(df["n"].iloc[0])
    return pd.DataFrame(rows), f"E3 tree build time, N = {n:,} (one build, 1 rank)"


def t_sched(dirs):
    df = load(dirs, "e4.csv")
    df["schedule"] = df["sched"] + ":" + df["chunk"].astype(int).astype(str)
    rows = []
    for dist in sorted(df["dist"].unique()):
        x = df[df["dist"] == dist]
        s = basic(x, "schedule")
        imb = med(x, "schedule", "thread_imbalance")
        best = s["median"].min()
        for sch in dict.fromkeys(x["schedule"]):
            rows.append({"distribution": dist, "schedule:chunk": sch, "s/step": s.loc[sch, "median"],
                         "vs best (%)": 100 * (s.loc[sch, "median"] / best - 1),
                         "thread imbalance (max/mean)": imb[sch], "repeats": s.loc[sch, "repeats"]})
    n, th = int(df["n"].iloc[0]), int(df["omp_threads"].iloc[0])
    return pd.DataFrame(rows), f"E4 OpenMP schedules, N = {n:,}, 1 rank x {th} threads"


def t_comm(dirs):
    df = add_mpi_cols(load(dirs, "e5a_*_n*.csv"))
    rows = []
    for c in sorted(df["cores"].unique()):
        x = df[df["cores"] == c]
        best = med(x, "comm", "t_step").min()
        for m in ("blocking", "p2p", "overlap"):
            y = x[x["comm"] == m]
            if y.empty:
                continue
            rows.append({"cores": int(c), "ranks x threads": y["split"].iloc[0], "mode": m,
                         "s/step": y["t_step"].median(), "vs best mode (%)": 100 * (y["t_step"].median() / best - 1),
                         "LET exchange (s/step)": per_step(y, "t_let_exchange_mean").median(),
                         "MPI wait, avg rank (s/step)": y["wait_s"].median(),
                         "communication (%)": y["comm_pct"].median(), "repeats": len(y)})
    return pd.DataFrame(rows), f"E5a LET exchange modes, N = {int(df['n'].iloc[0]):,}"


def t_shm(dirs):
    df = add_mpi_cols(load(dirs, "e5b_shm*_n*.csv"))
    rows = []
    for c in sorted(df["cores"].unique()):
        x = df[df["cores"] == c]
        off, on = x[x["shm"] == 0], x[x["shm"] == 1]
        rows.append({"cores": int(c), "ranks x threads": x["split"].iloc[0],
                     "windows off (s/step)": off["t_step"].median(), "windows on (s/step)": on["t_step"].median(),
                     "on is slower by (x)": on["t_step"].median() / off["t_step"].median(),
                     "shm publish, on (s/step)": per_step(on, "t_shm_publish_max").median(),
                     "LET received max rank, off (MB)": off["let_mb_max"].median(),
                     "LET received max rank, on (MB)": on["let_mb_max"].median(),
                     "repeats": min(len(off), len(on))})
    return pd.DataFrame(rows), f"E5b MPI shared-memory windows, N = {int(df['n'].iloc[0]):,}, 1 thread per rank"


def v1_base(dirs):
    v1 = load(dirs, *V1)
    return v1, v1.groupby("n")["t_step"].median()


def t_baseline(dirs):
    v1, _ = v1_base(dirs)
    rows = []
    for n in sorted(v1["n"].unique()):
        x = v1[v1["n"] == n]
        rows.append({"N": int(n), "s/step": x["t_step"].median(), "fastest": x["t_step"].min(),
                     "slowest": x["t_step"].max(), "spread (%)": 100 * (x["t_step"].max() / x["t_step"].min() - 1),
                     "repeats": len(x), "hosts": ", ".join(sorted(x["hostname"].astype(str).unique()))})
    return pd.DataFrame(rows), "Serial Barnes-Hut baselines (1 core, 1 thread): the reference for every speedup"


def t_strong(dirs):
    v1, base = v1_base(dirs)
    rows = []
    for n in sorted(v1["n"].unique()):
        x = v1[v1["n"] == n]
        rows.append({"N": int(n), "layout": "serial", "cores": 1, "nodes": 1, "ranks x threads": "serial BH",
                     "s/step": x["t_step"].median(), "fastest": x["t_step"].min(), "slowest": x["t_step"].max(),
                     "speedup": 1.0, "efficiency (%)": 100.0, "repeats": len(x)})
    for pat, layout, _ in LAYOUTS:
        df = load(dirs, pat)
        if df.empty:
            continue
        df = add_mpi_cols(df)
        for n in sorted(df["n"].unique()):
            x = df[df["n"] == n]
            b = base.get(n, np.nan)
            s = basic(x, "cores")
            for c in s.index:
                y = x[x["cores"] == c]
                rows.append({"N": int(n), "layout": layout, "cores": int(c), "nodes": int(y["nodes"].iloc[0]),
                             "ranks x threads": y["split"].iloc[0], "s/step": s.loc[c, "median"],
                             "fastest": s.loc[c, "fastest"], "slowest": s.loc[c, "slowest"],
                             "speedup": b / s.loc[c, "median"],
                             "efficiency (%)": 100 * b / s.loc[c, "median"] / c,
                             "communication (%)": y["comm_pct"].median(),
                             "MPI wait, avg rank (s/step)": y["wait_s"].median(),
                             "work imbalance (max/mean)": y["work_imbalance_last_step"].median(),
                             "LET received, max rank (MB)": y["let_mb_max"].median(),
                             "memory, max rank (MB)": y["mem_mb_max"].median(),
                             "repeats": int(s.loc[c, "repeats"])})
    out = pd.DataFrame(rows)
    out = out[out["N"].isin(out.loc[out["layout"] != "serial", "N"])]
    order = {"serial": 0, **{nm: i + 1 for i, (_, nm, _) in enumerate(LAYOUTS)}}
    out = out.sort_values(["N", "cores"], kind="stable").sort_values(
        ["N", "cores", "layout"], key=lambda c: c.map(order) if c.name == "layout" else c, kind="stable")
    return out, "E7b/E7b-MPI strong scaling (fixed N): hybrid (ranks x 4 threads) and pure MPI, speedup vs serial Barnes-Hut"


def t_breakdown(dirs, layouts, title):
    rows = []
    for pat, layout, _ in layouts:
        df = load(dirs, pat)
        if df.empty:
            continue
        df = add_mpi_cols(df)
        strong = pat.startswith("e7b")
        for n in sorted(df["n"].unique()) if strong else [None]:
            x = df if n is None else df[df["n"] == n]
            for c in sorted(x["cores"].unique()):
                y = x[x["cores"] == c]
                row = {"layout": layout, "N": int(y["n"].iloc[0]), "cores": int(c),
                       "ranks x threads": y["split"].iloc[0]}
                for col, nm in PHASES:
                    row[nm + " (s/step)"] = per_step(y, col).median()
                row["total, slowest rank (s/step)"] = y["t_step"].median()
                row["imbalance + other (s/step)"] = (row["total, slowest rank (s/step)"]
                                                     - sum(row[nm + " (s/step)"] for _, nm in PHASES))
                row["force walks share (%)"] = 100 * (row["local force walk (s/step)"] +
                                                      row["remote force walk (s/step)"]) / row["total, slowest rank (s/step)"]
                row["communication share (%)"] = y["comm_pct"].median()
                rows.append(row)
    return pd.DataFrame(rows), title


def t_weak(dirs):
    rows, per = [], None
    for pat, layout, _ in WEAK_LAYOUTS:
        df = load(dirs, pat)
        if df.empty:
            continue
        df = add_mpi_cols(df)
        s = basic(df, "cores")
        ipp = med(df, "cores", "ipp")
        b, bi = s["median"].iloc[0], ipp.iloc[0]
        per = int(df["n"].min() / df["cores"].min())
        for c in s.index:
            y = df[df["cores"] == c]
            eff = 100 * b / s.loc[c, "median"]
            rows.append({"layout": layout, "cores": int(c), "nodes": int(y["nodes"].iloc[0]), "N": int(y["n"].iloc[0]),
                         "ranks x threads": y["split"].iloc[0], "s/step": s.loc[c, "median"],
                         "fastest": s.loc[c, "fastest"], "slowest": s.loc[c, "slowest"],
                         "efficiency, raw (%)": eff, "interactions/particle": ipp[c],
                         "efficiency per interaction (%)": eff * ipp[c] / bi,
                         "communication (%)": y["comm_pct"].median(),
                         "MPI wait, avg rank (s/step)": y["wait_s"].median(),
                         "repeats": int(s.loc[c, "repeats"])})
    out = pd.DataFrame(rows).sort_values(["cores", "layout"], kind="stable")
    return out, f"E7c/E7c-MPI weak scaling, {per:,} particles per core, Plummer: hybrid and pure MPI"


def t_hybrid(dirs, pattern, title, newest=False):
    df = add_mpi_cols(load(dirs, pattern, newest=newest))
    _, base = v1_base(dirs)
    b = base.get(df["n"].iloc[0], np.nan)
    s = basic(df, ["mpi_ranks", "omp_threads"]).reset_index().sort_values("omp_threads")
    best = s["median"].min()
    rows = []
    for _, z in s.iterrows():
        y = df[(df["mpi_ranks"] == z["mpi_ranks"]) & (df["omp_threads"] == z["omp_threads"])]
        rows.append({"ranks x threads": f"{int(z['mpi_ranks'])} x {int(z['omp_threads'])}",
                     "s/step": z["median"], "fastest": z["fastest"], "slowest": z["slowest"],
                     "speedup vs serial": b / z["median"], "vs best (%)": 100 * (z["median"] / best - 1),
                     "communication (%)": y["comm_pct"].median(),
                     "thread imbalance (max/mean)": y["thread_imbalance_max"].median(),
                     "memory, max rank (MB)": y["mem_mb_max"].median(), "repeats": int(z["repeats"])})
    folder = os.path.basename(os.path.normpath(
        [d for d in sorted(dirs, key=lambda d: os.path.basename(os.path.normpath(d)))
         if glob.glob(os.path.join(d, pattern))][-1])) if newest else ""
    out = pd.DataFrame(rows)
    t = title.format(n=int(df["n"].iloc[0]), c=int(df["cores"].iloc[0]))
    return out, t + (f" (run {folder})" if folder else "")


def t_gpu(dirs):
    df = add_mpi_cols(load(dirs, "gpu.csv"))
    rows = []
    for ranks in sorted(df["mpi_ranks"].unique()):
        for n in sorted(df["n"].unique()):
            x = df[(df["mpi_ranks"] == ranks) & (df["n"] == n)]
            c, g = x[x["device"] == "cpu"], x[x["device"] == "gpu"]
            gt = g["t_step"].median()
            rows.append({"GPUs (= ranks)": int(ranks), "N": int(n),
                         "CPU 36 threads/rank (s/step)": c["t_step"].median(), "GPU (s/step)": gt,
                         "GPU slower by (%)": 100 * (gt / c["t_step"].median() - 1),
                         "GPU kernel share (%)": 100 * g["t_gpu_kernel_max"].median() / g["t_total_max"].median(),
                         "host-to-GPU copy share (%)": 100 * g["t_gpu_h2d_max"].median() / g["t_total_max"].median(),
                         "GPU-to-host copy share (%)": 100 * g["t_gpu_d2h_max"].median() / g["t_total_max"].median(),
                         "ran on GPU": bool((g["device_executed"] == "gpu").all()),
                         "repeats": min(len(c), len(g))})
    out = pd.DataFrame(rows)
    one = out[out["GPUs (= ranks)"] == 1].set_index("N")["GPU (s/step)"]
    two = out[out["GPUs (= ranks)"] == 2].set_index("N")["GPU (s/step)"]
    out["2 GPUs speedup over 1"] = out["N"].map(one / two).where(out["GPUs (= ranks)"] == 2)
    return out, "GPU offload (OpenMP target, nvc++) vs CPU on an A40 node"


def t_gpu_acc(dirs):
    a = pd.concat([pd.read_csv(f) for d in dirs for f in glob.glob(os.path.join(d, "gpu_accuracy.csv"))])
    out = pd.DataFrame({"N": a["n"].astype(int), "GPU vs CPU force: RMS rel. diff": a["accel_rms_rel"],
                        "max rel. diff": a["accel_max_rel"], "potential RMS rel. diff": a["potential_rms_rel"]})
    return out, "GPU result check: GPU forces vs CPU forces"


def t_conservation(dirs):
    rows = []
    for pat, nm in (("e7a.csv", "E7a ranks x threads, 1 node"), ("e7b_n*.csv", "E7b strong, hybrid"),
                    ("e7bm_n*.csv", "E7b strong, pure MPI"), ("e7c_n*.csv", "E7c weak, hybrid"),
                    ("e7cm_n*.csv", "E7c weak, pure MPI"), ("e7d_n*.csv", "E7d ranks x threads, 256 cores"),
                    ("e5a_*_n*.csv", "E5a exchange modes"), ("e5b_shm*_n*.csv", "E5b shared-memory windows"),
                    ("gpu.csv", "GPU")):
        df = load(dirs, pat, newest=(pat == "e7a.csv"))
        if df.empty:
            continue
        rows.append({"experiment": nm, "runs": len(df),
                     "rel. energy drift, max": df["rel_energy_drift"].abs().max(),
                     "rel. momentum drift, max": df["rel_momentum_drift"].abs().max(),
                     "mass error, max": (df["mass_final"] - 1).abs().max(),
                     "LET pruning violations, total": int(df["pruned_violations"].sum()) if "pruned_violations" in df else np.nan})
    return pd.DataFrame(rows), "Correctness checks across all MPI runs (2-step runs, so drifts are tiny)"


def t_inventory(dirs):
    rows = []
    e7a_dirs = sorted([d for d in dirs if os.path.exists(os.path.join(d, "e7a.csv"))],
                      key=lambda d: os.path.basename(os.path.normpath(d)))
    old_e7a = {os.path.normpath(d) for d in e7a_dirs[:-1]}
    new_e7a = os.path.basename(os.path.normpath(e7a_dirs[-1])) if e7a_dirs else ""

    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            df = pd.read_csv(f)
            if "git_hash" not in df:
                continue
            rows.append({"run folder": os.path.basename(os.path.normpath(d)), "file": os.path.basename(f),
                         "rows": len(df), "N values": ", ".join(f"{int(v):,}" for v in sorted(df["n"].unique())),
                         "commit": ", ".join(sorted(df["git_hash"].astype(str).unique())),
                         "hosts": len(df["hostname"].unique()),
                         "note": f"not used: replaced by the full rerun in {new_e7a}"
                         if os.path.basename(f) == "e7a.csv" and os.path.normpath(d) in old_e7a else ""})
    for d in dirs:
        f = os.path.join(d, "failures.txt")
        if os.path.exists(f):
            runs = [line.split()[2] for line in open(f) if line.strip()]
            note = "failed runs: " + ", ".join(runs)
            if runs and all(r.startswith("e7a_") for r in runs) and os.path.normpath(d) in old_e7a:
                note += f" (could not start: Open MPI would not bind a rank across both sockets); " \
                        f"all rerun successfully in {new_e7a}"
            rows.append({"run folder": os.path.basename(os.path.normpath(d)), "file": "failures.txt",
                         "rows": len(runs), "N values": "", "commit": "", "hosts": np.nan, "note": note})
    return pd.DataFrame(rows), "Inventory: which result files went into these tables"


def read_plain(dirs, pattern):
    files = [f for d in dirs for f in sorted(glob.glob(os.path.join(d, pattern)))]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def t_accuracy_parallel(dirs):
    a = read_plain(dirs, "e8_accuracy_n*.csv")
    if a.empty:
        return pd.DataFrame(), "E8 (not run)"
    rows = []
    for _, x in a.iterrows():
        rows.append({"N": int(x["n_total"]), "reference": x["reference"], "code": x["code"],
                     "ranks x threads": f"{int(x['ranks'])} x {int(x['threads'])}",
                     "cores": int(x["ranks"]) * int(x["threads"]),
                     "force error RMS (%)": 100 * x["accel_rms_rel"],
                     "force error median (%)": 100 * x["accel_median_rel"],
                     "force error 99th pct (%)": 100 * x["accel_p99_rel"],
                     "force error max (%)": 100 * x["accel_max_rel"],
                     "potential error RMS (%)": 100 * x["potential_rms_rel"]})
    out = pd.DataFrame(rows).drop_duplicates()
    out["_ref"] = out["reference"].map({"direct sum": 0, "serial BH": 1})
    out["_code"] = out["code"].map(lambda c: 0 if c.startswith("serial") else 1)
    out = out.sort_values(["N", "_ref", "_code", "cores"]).drop(columns=["_ref", "_code"])
    th = load(dirs, "e8_time_n*.csv")
    theta = f"theta = {th['theta'].iloc[0]:g}, " if not th.empty else ""
    return out, (f"E8 force accuracy of the parallel code at t = 0 ({theta}Plummer): against the direct sum, "
                 "and against the serial Barnes-Hut forces")


def t_long_run(dirs):
    s, d = load(dirs, "e9_n*.csv"), read_plain(dirs, "e9_diag_n*.csv")
    if s.empty:
        return pd.DataFrame(), "E9 (not run)"
    rows = []
    for _, x in s.sort_values("dt", ascending=False).iterrows():
        y = d[(d["dt"] == x["dt"]) & (d["n"] == x["n"])] if not d.empty else pd.DataFrame()
        rows.append({"N": int(x["n"]), "ranks x threads": f"{int(x['mpi_ranks'])} x {int(x['omp_threads'])}",
                     "dt": x["dt"], "steps": int(x["steps"]), "simulated time": x["steps"] * x["dt"],
                     "s/step": x["t_step"], "wall time (s)": x["t_total_max"],
                     "rel. energy drift, end": x["rel_energy_drift"],
                     "rel. energy drift, max over run": y["rel_energy_drift"].abs().max() if not y.empty else np.nan,
                     "rel. momentum drift, max over run": y["rel_momentum_drift"].abs().max() if not y.empty else np.nan,
                     "diagnostic points": len(y)})
    out = pd.DataFrame(rows)
    if len(out) >= 2:  # leapfrog is 2nd order: halving dt should cut the energy error about 4x
        out["energy error ratio vs largest dt"] = out["rel. energy drift, max over run"].iloc[0] / out[
            "rel. energy drift, max over run"]
    return out, "E9 long run: energy and momentum conservation over many steps (Plummer, KDK leapfrog)"


def t_load_balance(dirs):
    df = load(dirs, "e10_n*.csv")
    if df.empty:
        return pd.DataFrame(), "E10 (not run)"
    df = add_mpi_cols(df)
    df["balance"] = np.where(df["rebalance_every"] == 1, "work-weighted, every evaluation",
                             "equal count, fixed at start")
    rows = []
    for (dist, split), x in df.groupby(["dist", "split"], sort=True):
        base = x[x["rebalance_every"] == 1]["t_step"].median()
        for bal, y in x.groupby("balance", sort=False):
            rows.append({"distribution": dist, "ranks x threads": split, "split": bal,
                         "s/step": y["t_step"].median(), "fastest": y["t_step"].min(), "slowest": y["t_step"].max(),
                         "vs work-weighted (%)": 100 * (y["t_step"].median() / base - 1),
                         "work imbalance, last step (max/mean)": y["work_imbalance_last_step"].median(),
                         "particles per rank, max vs equal (%)": y["particles_spread_pct"].median(),
                         "decomposition (s/step)": per_step(y, "t_decomposition_mean").median(),
                         "migration (s/step)": per_step(y, "t_migration_mean").median(),
                         "repeats": len(y)})
    out = pd.DataFrame(rows).sort_values(["distribution", "ranks x threads", "split"], ascending=[True, True, False])
    return out, f"E10 load balancing, N = {int(df['n'].iloc[0]):,}: equal-count vs work-weighted Morton split"


def t_letboxes(dirs):
    df = load(dirs, "e11_n*.csv")
    if df.empty:
        return pd.DataFrame(), "E11 (not run)"
    df = add_mpi_cols(df)
    rows = []
    for c, x in df.groupby("cores"):
        ref = x[x["let_boxes"] == x["let_boxes"].max()]["t_step"].median()
        for k, y in x.groupby("let_boxes"):
            rows.append({"ranks x threads": y["split"].iloc[0], "boxes per rank K": int(k),
                         "s/step": y["t_step"].median(), "fastest": y["t_step"].min(), "slowest": y["t_step"].max(),
                         f"vs K = {int(x['let_boxes'].max())} (%)": 100 * (y["t_step"].median() / ref - 1),
                         "LET received, max rank (MB)": y["let_mb_max"].median(),
                         "LET received, avg rank (MB)": y["let_mb_mean"].median(),
                         "LET sent, all ranks (MB)": (y["let_bytes_sent_total"] / 1e6).median(),
                         "LET build, avg rank (s/step)": per_step(y, "t_let_build_mean").median(),
                         "LET exchange, avg rank (s/step)": per_step(y, "t_let_exchange_mean").median(),
                         "remote force walk, avg rank (s/step)": per_step(y, "t_force_remote_mean").median(),
                         "memory, max rank (MB)": y["mem_mb_max"].median(), "repeats": len(y)})
    return pd.DataFrame(rows), (f"E11 LET receiver boxes per rank K, N = {int(df['n'].iloc[0]):,}, "
                                "1 thread per rank (all other experiments use K = 64)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = a.runs
    tables = [
        ("00_baselines", t_baseline(d)),
        ("01_strong", t_strong(d)),
        ("02_strong_breakdown", t_breakdown(d, LAYOUTS, "Strong scaling: time per phase (average rank) "
                                                        "and the slowest rank's total, s per force eval")),
        ("03_weak", t_weak(d)),
        ("04_weak_breakdown", t_breakdown(d, WEAK_LAYOUTS, "Weak scaling: time per phase (average rank) "
                                                           "and the slowest rank's total, s per force eval")),
        ("05_comm_modes", t_comm(d)),
        ("06_shm_windows", t_shm(d)),
        ("07_hybrid_node", t_hybrid(d, "e7a.csv", "E7a ranks x threads on one node ({c} cores), N = {n:,}, "
                                    "every rank pinned to its own physical cores", newest=True)),
        ("08_hybrid_256", t_hybrid(d, "e7d_n*.csv", "E7d ranks x threads on {c} cores (4 nodes), N = {n:,}")),
        ("09_gpu", t_gpu(d)),
        ("10_gpu_accuracy", t_gpu_acc(d)),
        ("11_serial", t_serial(d)),
        ("12_theta", t_theta(d)),
        ("13_tree_build", t_treebuild(d)),
        ("14_schedules", t_sched(d)),
        ("15_correctness", t_conservation(d)),
        ("16_inventory", t_inventory(d)),
        ("17_accuracy_parallel", t_accuracy_parallel(d)),
        ("18_long_run", t_long_run(d)),
        ("19_load_balance", t_load_balance(d)),
        ("20_let_boxes", t_letboxes(d)),
    ]
    tables = [(name, (df, title)) for name, (df, title) in tables if not df.empty]
    xlsx = os.path.join(a.out, "results_tables.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for name, (df, title) in tables:
            df = df.rename(columns=lambda c: c.replace("s/step", "s per force eval"))
            df.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
            pd.DataFrame({title: []}).to_excel(w, sheet_name=name, index=False)
            df.to_excel(w, sheet_name=name, index=False, startrow=2)
            ws = w.sheets[name]
            for i, col in enumerate(df.columns):
                ws.column_dimensions[ws.cell(row=3, column=i + 1).column_letter].width = max(10, min(34, len(str(col)) + 2))
            for row in ws.iter_rows(min_row=4):
                for cell in row:
                    if isinstance(cell.value, float):
                        v = abs(cell.value)
                        cell.number_format = "0.00E+00" if (v and (v < 1e-3 or v >= 1e7)) else "0.000" if v < 100 else "0.0"
    pd.to_pickle(tables, os.path.join(a.out, "tables.pkl"))
    print("wrote", xlsx, "and", len(tables), "CSV files")


if __name__ == "__main__":
    main()
