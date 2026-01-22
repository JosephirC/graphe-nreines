from __future__ import annotations

import time
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt

# Solveurs (architecture actuelle)
from solvers.complete.cp_sat_fixed_search_first_fail import CPSatFirstFailSolver
from solvers.complete.cp_sat_fixed_search_center_out import CPSatCenterOutSolver
from solvers.incomplete.cp_sat_lns import CPSatLNSSolver
from solvers.incomplete.cp_sat_min_conflicts import CPSatMinConflictsSolver


@dataclass
class MethodSpec:
    name: str
    group: str  # "COMPLETE" | "INCOMPLETE"
    build: Callable[[int, bool, int], Any]


def run_benchmark_solutions(
    ns: List[int],
    time_limit: float,
    symmetry_breaking: bool,
    seed: int,
    repeats: int,
    # LNS params
    lns_neighborhood_size: float,
    lns_policy: str,
    lns_time_per_iter: float,
    lns_initial_time: float,
    lns_multistart: bool,
    # Min-conflicts params
    mc_noise: float,
    mc_max_steps: int,
    mc_pick_policy: str,
) -> pd.DataFrame:
    """
    Benchmark #1:
    Pour chaque N et chaque méthode, compter le nombre de solutions valides distinctes trouvées
    dans la limite de temps.

    Important:
    - COMPLETE = énumération (SearchForAllSolutions) => "nb solutions énumérées"
    - INCOMPLETE = sampling anytime => "nb solutions distinctes rencontrées"
    """

    methods: List[MethodSpec] = [
        MethodSpec(
            name="COMPLETE_first_fail",
            group="COMPLETE",
            build=lambda n, sym, s: CPSatFirstFailSolver(
                n=n, symmetry_breaking=sym, seed=s),
        ),
        MethodSpec(
            name="COMPLETE_center_out",
            group="COMPLETE",
            build=lambda n, sym, s: CPSatCenterOutSolver(
                n=n, symmetry_breaking=sym, seed=s),
        ),
        MethodSpec(
            name="INCOMPLETE_lns",
            group="INCOMPLETE",
            build=lambda n, sym, s: CPSatLNSSolver(
                n=n,
                symmetry_breaking=sym,
                seed=s,
                neighborhood_size=lns_neighborhood_size,
                neighborhood_policy=lns_policy,
                time_limit_per_iteration=lns_time_per_iter,
            ),
        ),
        MethodSpec(
            name="INCOMPLETE_min_conflicts",
            group="INCOMPLETE",
            build=lambda n, sym, s: CPSatMinConflictsSolver(
                n=n,
                symmetry_breaking=sym,
                seed=s,
                noise=mc_noise,
                max_steps=mc_max_steps,
                pick_policy=mc_pick_policy,
            ),
        ),
    ]

    rows: List[Dict[str, Any]] = []

    for n in ns:
        for m in methods:
            for rep in range(repeats):
                run_seed = seed + rep  # simple & reproductible

                solver = m.build(n, symmetry_breaking, run_seed)

                t0 = time.time()

                # LNS: activer le mode itératif si demandé
                if m.name == "INCOMPLETE_lns":
                    result = solver.solve(
                        time_limit=time_limit,
                        initial_time_limit=lns_initial_time,
                        max_iterations=None,
                        multistart=lns_multistart,
                    )
                else:
                    result = solver.solve(time_limit=time_limit)

                wall = time.time() - t0
                exec_time = result.execution_time if result.execution_time else wall

                rows.append({
                    "N": n,
                    "Method": m.name,
                    "Group": m.group,
                    "Repeat": rep,
                    "Seed": run_seed,
                    "TimeLimit(s)": time_limit,
                    "SymmetryBreaking": symmetry_breaking,
                    "UniqueSolutionsInTime": result.num_unique_solutions(),
                    "ExecutionTime(s)": exec_time,
                    "Success": bool(result.success),

                    # utiles pour debug/analyses
                    "TimeToFirst(s)": result.time_to_first_solution,
                    "Iterations": result.iterations,
                    "NodesOrSteps": result.nodes_explored,

                    # garder un peu de metadata clé si dispo
                    "Meta_restarts": result.metadata.get("restarts"),
                    "Meta_policy": result.metadata.get("neighborhood_policy") or result.metadata.get("pick_policy"),
                    "Meta_noise": result.metadata.get("noise"),
                    "Meta_neighborhood_size": result.metadata.get("neighborhood_size"),
                    "Meta_multistart": result.metadata.get("multistart"),
                })

                print(
                    f"N={n:2d} | {m.name:24s} | rep={rep:02d} | "
                    f"unique={rows[-1]['UniqueSolutionsInTime']:6d} | "
                    f"time={rows[-1]['ExecutionTime(s)']:.3f}s | "
                    f"success={rows[-1]['Success']}"
                )

    return pd.DataFrame(rows)


def summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé par (N, Method): moyenne + std sur les repeats.
    """
    g = df_raw.groupby(["N", "Method", "Group"], as_index=False)[
        "UniqueSolutionsInTime"].agg(["mean", "std", "min", "max"])
    g.columns = ["N", "Method", "Group", "MeanUnique",
                 "StdUnique", "MinUnique", "MaxUnique"]
    return g


def plot_solutions(df_summary: pd.DataFrame, out_png: str, title: str) -> None:
    """
    Plot mean +/- std. Courbe lisible même si les incomplete varient.
    """
    plt.figure(figsize=(11, 6))

    methods = sorted(df_summary["Method"].unique())
    for method in methods:
        sub = df_summary[df_summary["Method"] == method].sort_values("N")
        x = sub["N"]
        y = sub["MeanUnique"]
        yerr = sub["StdUnique"].fillna(0.0)

        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=method)

    plt.xlabel("Taille du problème (N)")
    plt.ylabel(
        "Solutions distinctes trouvées dans la limite de temps (moyenne ± écart-type)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark #1: #solutions trouvées en temps limité (complet vs incomplet)."
    )
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=14)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--symmetry-breaking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3,
                        help="Nb de runs par (N, méthode) pour moyenne/std")

    # LNS params
    parser.add_argument("--lns-neighborhood-size", type=float, default=0.30)
    parser.add_argument("--lns-policy", type=str, default="random",
                        choices=["random", "center_fix", "edge_fix", "unstable_relax"])
    parser.add_argument("--lns-time-per-iter", type=float,
                        default=0.15, help="Petit => plus d'itérations")
    parser.add_argument("--lns-initial-time", type=float, default=1.0)
    parser.add_argument("--lns-multistart", action="store_true",
                        help="Active LNS itératif (restarts sur stagnation)")

    # Min-conflicts params
    parser.add_argument("--mc-noise", type=float, default=0.15)
    parser.add_argument("--mc-max-steps", type=int, default=30000)
    parser.add_argument("--mc-pick-policy", type=str, default="max_conflict",
                        choices=["random", "max_conflict"])

    parser.add_argument("--out-csv-raw", type=str,
                        default="benchmark_solutions_time_limit_raw.csv")
    parser.add_argument("--out-csv-summary", type=str,
                        default="benchmark_solutions_time_limit_summary.csv")
    parser.add_argument("--out-png", type=str,
                        default="benchmark_solutions_time_limit.png")
    args = parser.parse_args()

    ns = list(range(args.n_min, args.n_max + 1))

    df_raw = run_benchmark_solutions(
        ns=ns,
        time_limit=args.time_limit,
        symmetry_breaking=args.symmetry_breaking,
        seed=args.seed,
        repeats=args.repeats,

        lns_neighborhood_size=args.lns_neighborhood_size,
        lns_policy=args.lns_policy,
        lns_time_per_iter=args.lns_time_per_iter,
        lns_initial_time=args.lns_initial_time,
        lns_multistart=args.lns_multistart,

        mc_noise=args.mc_noise,
        mc_max_steps=args.mc_max_steps,
        mc_pick_policy=args.mc_pick_policy,
    )

    df_raw.to_csv(args.out_csv_raw, index=False)
    print(f"\nCSV raw écrit: {args.out_csv_raw}")

    df_summary = summarize(df_raw)
    df_summary.to_csv(args.out_csv_summary, index=False)
    print(f"CSV summary écrit: {args.out_csv_summary}")

    title = (
        f"N-reines — Solutions trouvées en {args.time_limit:.0f}s "
        f"(symmetry_breaking={args.symmetry_breaking}, repeats={args.repeats}, LNS_multistart={args.lns_multistart})"
    )
    plot_solutions(df_summary, args.out_png, title)
    print(f"PNG écrit: {args.out_png}")


if __name__ == "__main__":
    main()
