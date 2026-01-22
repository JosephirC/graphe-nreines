# benchmark_time_to_first.py
from __future__ import annotations

import time
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt

# Solveurs
from solvers.complete.cp_sat_fixed_search_first_fail import CPSatFirstFailSolver
from solvers.complete.cp_sat_fixed_search_center_out import CPSatCenterOutSolver
from solvers.incomplete.cp_sat_lns import CPSatLNSSolver
from solvers.incomplete.cp_sat_min_conflicts import CPSatMinConflictsSolver


@dataclass
class MethodSpec:
    name: str
    # (n, symmetry_breaking, seed) -> solver instance
    build: Callable[[int, bool, int], Any]


def _time_to_first_solution(result: Any, fallback_wall: float, timeout: float) -> Optional[float]:
    """
    Tries to read time-to-first from SolverResult.
    Fallbacks:
      - if success and has solutions, use execution_time (approx)
      - if not solved, return None
    """
    t_first = getattr(result, "time_to_first_solution", None)
    if isinstance(t_first, (int, float)) and t_first >= 0:
        return float(t_first)

    # Some results may only expose execution_time
    success = bool(getattr(result, "success", False))
    nsol = None
    if hasattr(result, "num_unique_solutions"):
        try:
            nsol = result.num_unique_solutions()
        except Exception:
            nsol = None

    if success and (nsol is None or nsol > 0):
        exec_t = getattr(result, "execution_time", None)
        if isinstance(exec_t, (int, float)) and exec_t > 0:
            return float(exec_t)
        return float(fallback_wall)

    return None


def run_benchmark_time_to_first(
    ns: List[int],
    time_limit: float,
    symmetry_breaking: bool,
    seeds: List[int],
) -> pd.DataFrame:
    """
    Benchmark #2:
    Pour chaque N et chaque méthode, mesurer le temps pour obtenir la première solution valide.
    On répète sur plusieurs seeds, puis on analysera médiane/IQR.
    """

    methods: List[MethodSpec] = [
        MethodSpec(
            name="COMPLETE_first_fail",
            build=lambda n, sym, s: CPSatFirstFailSolver(
                n=n, symmetry_breaking=sym, seed=s),
        ),
        MethodSpec(
            name="COMPLETE_center_out",
            build=lambda n, sym, s: CPSatCenterOutSolver(
                n=n, symmetry_breaking=sym, seed=s),
        ),
        MethodSpec(
            name="INCOMPLETE_lns",
            build=lambda n, sym, s: CPSatLNSSolver(
                n=n,
                symmetry_breaking=sym,
                seed=s,
                neighborhood_size=0.30,
                time_limit_per_iteration=min(1.0, time_limit),
            ),
        ),
        MethodSpec(
            name="INCOMPLETE_min_conflicts",
            build=lambda n, sym, s: CPSatMinConflictsSolver(
                n=n, symmetry_breaking=sym, seed=s),
        ),
    ]

    rows: List[Dict[str, Any]] = []

    for n in ns:
        for m in methods:
            for seed in seeds:
                solver = m.build(n, symmetry_breaking, seed)

                t0 = time.time()
                result = solver.solve(time_limit=time_limit)
                wall = time.time() - t0

                t_first = _time_to_first_solution(
                    result, fallback_wall=wall, timeout=time_limit)
                solved = t_first is not None

                rows.append({
                    "N": n,
                    "Method": m.name,
                    "Seed": seed,
                    "TimeLimit(s)": time_limit,
                    "Solved": solved,
                    # cap for analysis/plot
                    "TimeToFirst(s)": t_first if solved else time_limit,
                    "CappedToTimeout": (not solved),
                    "ExecutionTime(s)": getattr(result, "execution_time", wall),
                    "UniqueSolutions": result.num_unique_solutions() if hasattr(result, "num_unique_solutions") else None,
                })

                print(
                    f"N={n:2d} | {m.name:24s} | seed={seed:4d} | "
                    f"t_first={'TIMEOUT' if not solved else f'{t_first:.4f}s'}"
                )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groupby (N, Method) -> median + IQR, and success rate.
    """
    g = df.groupby(["N", "Method"], as_index=False)

    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)

    out = g.agg(
        median_time=("TimeToFirst(s)", "median"),
        q1=("TimeToFirst(s)", q1),
        q3=("TimeToFirst(s)", q3),
        success_rate=("Solved", "mean"),
        runs=("Solved", "count"),
        timeouts=("CappedToTimeout", "sum"),
    )
    out["iqr"] = out["q3"] - out["q1"]
    return out


def plot_time_to_first(summary_df: pd.DataFrame, out_png: str, title: str) -> None:
    """
    Plot median time to first solution vs N with IQR error bars.
    """
    plt.figure(figsize=(10, 5))

    methods = sorted(summary_df["Method"].unique())
    for method in methods:
        sub = summary_df[summary_df["Method"] == method].sort_values("N")
        x = sub["N"].tolist()
        y = sub["median_time"].tolist()
        yerr_low = (sub["median_time"] - sub["q1"]).tolist()
        yerr_high = (sub["q3"] - sub["median_time"]).tolist()
        yerr = [yerr_low, yerr_high]
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=method)

    plt.xlabel("Taille du problème (N)")
    plt.ylabel("Temps pour la première solution (s) — médiane + IQR")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_success_rate(summary_df: pd.DataFrame, out_png: str, title: str) -> None:
    """
    Optional plot: success rate vs N.
    """
    plt.figure(figsize=(10, 5))

    methods = sorted(summary_df["Method"].unique())
    for method in methods:
        sub = summary_df[summary_df["Method"] == method].sort_values("N")
        plt.plot(sub["N"], sub["success_rate"], marker="o", label=method)

    plt.xlabel("Taille du problème (N)")
    plt.ylabel("Taux de succès (proportion de runs résolus)")
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark #2: temps pour la première solution (complet vs incomplet).")
    parser.add_argument("--n-min", type=int, default=8, help="N minimal")
    parser.add_argument("--n-max", type=int, default=20,
                        help="N maximal (inclus)")
    parser.add_argument("--n-step", type=int, default=1,
                        help="Pas d'incrément sur N (ex: 20 => 20,40,60...)")
    parser.add_argument("--time-limit", type=float,
                        default=30.0, help="Timeout (secondes) par run")
    parser.add_argument("--symmetry-breaking",
                        action="store_true", help="Active symmetry breaking")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Liste seeds séparées par virgules")
    parser.add_argument("--out-csv", type=str,
                        default="benchmark_time_to_first_raw.csv", help="CSV raw")
    parser.add_argument("--out-summary-csv", type=str,
                        default="benchmark_time_to_first_summary.csv", help="CSV résumé")
    parser.add_argument("--out-png", type=str,
                        default="benchmark_time_to_first.png", help="Graphe temps")
    parser.add_argument("--out-success-png", type=str,
                        default="benchmark_time_to_first_success.png", help="Graphe taux succès")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]

    if args.n_step <= 0:
        raise ValueError("--n-step doit être > 0")

    ns = list(range(args.n_min, args.n_max + 1, args.n_step))

    df = run_benchmark_time_to_first(
        ns=ns,
        time_limit=args.time_limit,
        symmetry_breaking=args.symmetry_breaking,
        seeds=seeds,
    )
    df.to_csv(args.out_csv, index=False)
    print(f"\nCSV raw écrit: {args.out_csv}")

    sum_df = summarize(df)
    sum_df.to_csv(args.out_summary_csv, index=False)
    print(f"CSV résumé écrit: {args.out_summary_csv}")

    title = f"N-reines — Temps première solution (timeout={args.time_limit:.0f}s, symmetry_breaking={args.symmetry_breaking})"
    plot_time_to_first(sum_df, args.out_png, title)
    print(f"PNG écrit: {args.out_png}")

    title2 = f"N-reines — Taux de succès (timeout={args.time_limit:.0f}s)"
    plot_success_rate(sum_df, args.out_success_png, title2)
    print(f"PNG écrit: {args.out_success_png}")


if __name__ == "__main__":
    main()
