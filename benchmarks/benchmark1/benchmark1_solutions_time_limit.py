from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

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


def _safe(s: str) -> str:
    # pour noms de dossiers/fichiers
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def _make_run_id(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    core = (
        f"n{args.n_min}-{args.n_max}"
        f"_t{int(args.time_limit)}"
        f"_rep{args.repeats}"
        f"_sym{int(args.symmetry_breaking)}"
        f"_lns{args.lns_policy}_ms{int(args.lns_multistart)}"
        f"_mc{args.mc_pick_policy}"
    )
    return f"{ts}_{_safe(core)}"


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
    log_path: Path,
) -> pd.DataFrame:
    """
    Benchmark #1:
    Pour chaque N et chaque méthode, compter le nombre de solutions valides distinctes trouvées
    dans la limite de temps.

    NOTE:
      - COMPLETE = énumération / search all solutions => "solutions énumérées"
      - INCOMPLETE = anytime sampling => "solutions distinctes rencontrées"
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

    def log(line: str) -> None:
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    for n in ns:
        for m in methods:
            for rep in range(repeats):
                run_seed = seed + rep  # reproductible

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

                row = {
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

                    # utiles pour debug / analyses
                    "TimeToFirst(s)": result.time_to_first_solution,
                    "Iterations": result.iterations,
                    "NodesOrSteps": result.nodes_explored,

                    # metadata clé
                    "Meta_restarts": result.metadata.get("restarts"),
                    "Meta_policy": result.metadata.get("neighborhood_policy") or result.metadata.get("pick_policy"),
                    "Meta_noise": result.metadata.get("noise"),
                    "Meta_neighborhood_size": result.metadata.get("neighborhood_size"),
                    "Meta_multistart": result.metadata.get("multistart"),
                }
                rows.append(row)

                log(
                    f"N={n:2d} | {m.name:24s} | rep={rep:02d} | "
                    f"unique={row['UniqueSolutionsInTime']:6d} | "
                    f"time={row['ExecutionTime(s)']:.3f}s | success={row['Success']}"
                )

    return pd.DataFrame(rows)


def summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    g = df_raw.groupby(["N", "Method", "Group"], as_index=False)[
        "UniqueSolutionsInTime"].agg(["mean", "std", "min", "max"])
    g.columns = ["N", "Method", "Group", "MeanUnique",
                 "StdUnique", "MinUnique", "MaxUnique"]
    return g


def plot_solutions(df_summary: pd.DataFrame, out_png: Path, title: str) -> None:
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
    parser.add_argument("--time-limit", type=float, default=45.0)
    parser.add_argument("--symmetry-breaking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)

    # LNS params
    parser.add_argument("--lns-neighborhood-size", type=float, default=0.30)
    parser.add_argument("--lns-policy", type=str, default="random",
                        choices=["random", "center_fix", "edge_fix", "unstable_relax"])
    parser.add_argument("--lns-time-per-iter", type=float, default=0.10)
    parser.add_argument("--lns-initial-time", type=float, default=1.0)
    parser.add_argument("--lns-multistart", action="store_true")

    # Min-conflicts params
    parser.add_argument("--mc-noise", type=float, default=0.15)
    parser.add_argument("--mc-max-steps", type=int, default=3000)
    parser.add_argument("--mc-pick-policy", type=str, default="max_conflict",
                        choices=["random", "max_conflict"])

    # output root
    parser.add_argument("--out-root", type=str, default="benchmarks/benchmark1/runs",
                        help="Dossier racine où créer le run (un sous-dossier par exécution).")
    args = parser.parse_args()

    ns = list(range(args.n_min, args.n_max + 1))

    run_id = _make_run_id(args)
    run_dir = Path(args.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # fichiers de sortie fixes par run
    raw_csv = run_dir / "raw.csv"
    summary_csv = run_dir / "summary.csv"
    plot_png = run_dir / "plot.png"
    config_json = run_dir / "config.json"
    log_txt = run_dir / "log.txt"

    # dump config
    with config_json.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"RUN_DIR: {run_dir}")

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

        log_path=log_txt,
    )

    df_raw.to_csv(raw_csv, index=False)
    print(f"CSV raw: {raw_csv}")

    df_summary = summarize(df_raw)
    df_summary.to_csv(summary_csv, index=False)
    print(f"CSV summary: {summary_csv}")

    title = (
        f"N-reines — solutions en {int(args.time_limit)}s | "
        f"sym={args.symmetry_breaking} | rep={args.repeats} | "
        f"LNS={args.lns_policy}, ms={args.lns_multistart} | "
        f"MC={args.mc_pick_policy}"
    )
    plot_solutions(df_summary, plot_png, title)
    print(f"PNG: {plot_png}")


if __name__ == "__main__":
    main()
