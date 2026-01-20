from __future__ import annotations

import time
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Any

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
    # (n, symmetry_breaking, seed) -> solver instance
    build: Callable[[int, bool, int], Any]


def run_benchmark_solutions(
    ns: List[int],
    time_limit: float,
    symmetry_breaking: bool,
    seed: int,
) -> pd.DataFrame:
    """
    Benchmark #1:
    Pour chaque N et chaque méthode, compter le nombre de solutions valides distinctes trouvées
    dans la limite de temps.
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
                n=n, symmetry_breaking=sym, seed=s,
                neighborhood_size=0.30,
                time_limit_per_iteration=1.0,
            ),
        ),
        MethodSpec(
            name="INCOMPLETE_min_conflicts",
            build=lambda n, sym, s: CPSatMinConflictsSolver(
                n=n, symmetry_breaking=sym, seed=s,
                conflict_types=("columns", "diagonals"),
            ),
        ),
    ]

    rows: List[Dict[str, Any]] = []

    for n in ns:
        for m in methods:
            solver = m.build(n, symmetry_breaking, seed)

            t0 = time.time()
            # Tous tes solveurs ont solve(time_limit=...)
            result = solver.solve(time_limit=time_limit)
            wall = time.time() - t0

            rows.append({
                "N": n,
                "Method": m.name,
                # si tu l'ajoutes, sinon vide
                "SolverType": result.metadata.get("solver_type", ""),
                "UniqueSolutionsInTime": result.num_unique_solutions(),
                "ExecutionTime(s)": result.execution_time if result.execution_time else wall,
                "Success": result.success,
            })

            print(
                f"N={n:2d} | {m.name:24s} | "
                f"unique={rows[-1]['UniqueSolutionsInTime']:6d} | "
                f"time={rows[-1]['ExecutionTime(s)']:.3f}s | "
                f"success={rows[-1]['Success']}"
            )

    return pd.DataFrame(rows)


def plot_solutions(df: pd.DataFrame, out_png: str, title: str) -> None:
    plt.figure(figsize=(10, 5))

    for method in sorted(df["Method"].unique()):
        sub = df[df["Method"] == method].sort_values("N")
        plt.plot(sub["N"], sub["UniqueSolutionsInTime"],
                 marker="o", label=method)

    plt.xlabel("Taille du problème (N)")
    plt.ylabel("Nombre de solutions valides distinctes en temps limité")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark #1: #solutions trouvées en temps limité (complet vs incomplet).")
    parser.add_argument("--n-min", type=int, default=8, help="N minimal")
    parser.add_argument("--n-max", type=int, default=14,
                        help="N maximal (inclus)")
    parser.add_argument("--time-limit", type=float, default=45.0,
                        help="Limite de temps (secondes) par run")
    parser.add_argument("--symmetry-breaking", action="store_true",
                        help="Active symmetry breaking (attention au comptage)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine RNG (reproductibilité)")
    parser.add_argument("--out-csv", type=str,
                        default="benchmark_solutions_45s.csv", help="Fichier CSV de sortie")
    parser.add_argument("--out-png", type=str,
                        default="benchmark_solutions_45s.png", help="Image du graphe")
    args = parser.parse_args()

    ns = list(range(args.n_min, args.n_max + 1))

    df = run_benchmark_solutions(
        ns=ns,
        time_limit=args.time_limit,
        symmetry_breaking=args.symmetry_breaking,
        seed=args.seed,
    )

    df.to_csv(args.out_csv, index=False)
    print(f"\nCSV écrit: {args.out_csv}")

    title = f"N-reines — Solutions trouvées en {args.time_limit:.0f}s (symmetry_breaking={args.symmetry_breaking})"
    plot_solutions(df, args.out_png, title)
    print(f"PNG écrit: {args.out_png}")


if __name__ == "__main__":
    main()
