"""
Lab 1 – Parallel Computing
Analysis script: reads results.csv, computes Speedup & Efficiency,
prints a summary table and saves two PNG charts.
"""

import csv
import statistics
import os
from collections import defaultdict

# ── Try matplotlib ────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("matplotlib not found – skipping charts")

# ── Load CSV ──────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "results.csv")

rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["threads"]    = int(row["threads"])
        row["elapsed_ms"] = float(row["elapsed_ms"])
        row["run_i"]      = int(row["run_i"])
        rows.append(row)

# ── Aggregate: mean + p95 per (workload, impl, threads) ──────────────────────
groups = defaultdict(list)
for r in rows:
    key = (r["workload"], r["impl"], r["threads"])
    groups[key].append(r["elapsed_ms"])

stats = {}
for key, times in groups.items():
    s = sorted(times)
    idx = int(0.95 * (len(s) - 1))
    stats[key] = {
        "mean": statistics.mean(times),
        "p95":  s[idx],
    }

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'Workload':<12} {'Impl':<8} {'Threads':>7}  {'Mean ms':>9}  {'P95 ms':>9}  {'Speedup':>8}  {'Efficiency':>10}")
print("─" * 75)

for workload in ("sum", "transform"):
    serial_key = (workload, "serial", 1)
    T_serial   = stats[serial_key]["mean"]

    for impl in ("serial", "openmp", "threads"):
        for nt in sorted(set(k[2] for k in stats if k[0] == workload and k[1] == impl)):
            key = (workload, impl, nt)
            if key not in stats:
                continue
            m   = stats[key]["mean"]
            p   = stats[key]["p95"]
            sp  = T_serial / m
            eff = sp / nt * 100
            print(f"{workload:<12} {impl:<8} {nt:>7}  {m:>9.2f}  {p:>9.2f}  {sp:>8.3f}  {eff:>9.1f}%")
    print()

# ── Charts ────────────────────────────────────────────────────────────────────
if HAS_PLT:
    IMPL_STYLES = {
        "openmp":  {"color": "#2196F3", "marker": "o", "label": "OpenMP"},
        "threads": {"color": "#FF5722", "marker": "s", "label": "std::thread"},
        "serial":  {"color": "#9E9E9E", "marker": "x", "label": "Serial"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Lab 1 – Parallel Computing Benchmark\n(80M doubles, 10 runs, warm-up 3)",
                 fontsize=14, fontweight="bold")

    workload_titles = {"sum": "Task A: Sum", "transform": "Task B: Element-wise Transform"}

    for col, workload in enumerate(("sum", "transform")):
        serial_key = (workload, "serial", 1)
        T_serial   = stats[serial_key]["mean"]

        ax_sp  = axes[0][col]   # Speedup
        ax_eff = axes[1][col]   # Efficiency

        thread_counts = sorted(set(k[2] for k in stats if k[0] == workload and k[2] > 0))

        for impl in ("openmp", "threads"):
            xs, speedups, effs = [], [], []
            for nt in thread_counts:
                key = (workload, impl, nt)
                if key not in stats:
                    continue
                sp  = T_serial / stats[key]["mean"]
                eff = sp / nt * 100
                xs.append(nt)
                speedups.append(sp)
                effs.append(eff)
            st = IMPL_STYLES[impl]
            ax_sp.plot(xs, speedups, color=st["color"], marker=st["marker"],
                       linewidth=2, markersize=7, label=st["label"])
            ax_eff.plot(xs, effs,    color=st["color"], marker=st["marker"],
                        linewidth=2, markersize=7, label=st["label"])

        # Ideal speedup line
        ax_sp.plot(thread_counts, thread_counts, "k--", linewidth=1.2, label="Ideal")
        ax_sp.plot([1], [1.0], "gv", markersize=10, label=f"Serial ({T_serial:.1f} ms)")

        for ax in (ax_sp, ax_eff):
            ax.set_xlabel("Threads", fontsize=11)
            ax.set_xticks(thread_counts)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)

        ax_sp.set_title(f"{workload_titles[workload]} – Speedup", fontsize=12)
        ax_sp.set_ylabel("Speedup (T_serial / T_parallel)", fontsize=10)

        ax_eff.set_title(f"{workload_titles[workload]} – Efficiency", fontsize=12)
        ax_eff.set_ylabel("Efficiency (%)", fontsize=10)
        ax_eff.axhline(100, color="k", linestyle="--", linewidth=1.2, label="Ideal 100%")
        ax_eff.set_ylim(0, 115)

    plt.tight_layout()
    out_png = os.path.join(script_dir, "speedup_charts.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Charts saved to {out_png}")
else:
    # ASCII fallback
    print("\n=== ASCII Speedup Chart (Task A: Sum) ===")
    serial_key = ("sum", "serial", 1)
    T = stats[serial_key]["mean"]
    for impl in ("openmp", "threads"):
        print(f"  {impl}:")
        for nt in sorted(set(k[2] for k in stats if k[0]=="sum" and k[1]==impl)):
            sp = T / stats[("sum", impl, nt)]["mean"]
            bar = "█" * int(sp * 10)
            print(f"    t={nt:2d}  {bar} {sp:.2f}x")
