from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt

from solvers.complete.cp_sat_fixed_search_first_fail import CPSatFirstFailSolver
from solvers.complete.cp_sat_fixed_search_center_out import CPSatCenterOutSolver
from solvers.incomplete.cp_sat_lns import CPSatLNSSolver
from solvers.incomplete.cp_sat_min_conflicts import CPSatMinConflictsSolver


@dataclass
class MethodSpec:
    name: str
    group: str  # "COMPLETE" | "INCOMPLETE"
    build: Callable[[int, bool, int], Any]


def _time_to_first_solution(result: Any, fallback_wall: float) -> Optional[float]:
    """
    Lit time_to_first_solution si disponible.
    Fallback: si success=True, approx = execution_time ou wall.
    Retourne None si pas de solution.
    """
    t_first = getattr(result, "time_to_first_solution", None)
    if isinstance(t_first, (int, float)) and t_first >= 0:
        return float(t_first)

    success = bool(getattr(result, "success", False))
    if not success:
        return None

    exec_t = getattr(result, "execution_time", None)
    if isinstance(exec_t, (int, float)) and exec_t > 0:
        return float(exec_t)

    return float(fallback_wall)


def run_benchmark_time_to_first(
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
    Benchmark #2:
    Pour chaque N et chaque méthode, mesurer le temps pour obtenir la première solution valide.
    Repeats = nb de runs par (N, méthode), avec seed + rep.
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
                run_seed = seed + rep
                solver = m.build(n, symmetry_breaking, run_seed)

                t0 = time.time()

                # LNS
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

                t_first = _time_to_first_solution(result, fallback_wall=wall)
                solved = t_first is not None
                t_first_capped = float(
                    t_first) if solved else float(time_limit)

                meta = getattr(result, "metadata", {}) or {}

                rows.append({
                    "N": n,
                    "Method": m.name,
                    "Group": m.group,
                    "Repeat": rep,
                    "Seed": run_seed,
                    "TimeLimit(s)": time_limit,
                    "SymmetryBreaking": symmetry_breaking,

                    # métriques benchmark2
                    "Solved": solved,
                    "TimeToFirst(s)": t_first_capped,
                    "CappedToTimeout": (not solved),

                    # debug / comparabilité
                    "ExecutionTime(s)": getattr(result, "execution_time", wall),
                    "UniqueSolutionsInRun": (result.num_unique_solutions() if hasattr(result, "num_unique_solutions") else None),
                    "Iterations": getattr(result, "iterations", None),
                    "NodesOrSteps": getattr(result, "nodes_explored", None),

                    "Meta_policy": meta.get("neighborhood_policy") or meta.get("pick_policy"),
                    "Meta_noise": meta.get("noise"),
                    "Meta_neighborhood_size": meta.get("neighborhood_size"),
                    "Meta_multistart": meta.get("multistart"),
                })

                label = "TIMEOUT" if not solved else f"{t_first:.4f}s"
                print(
                    f"N={n:3d} | {m.name:24s} | rep={rep:02d} | t_first={label:>10s} | wall={wall:.3f}s"
                )

    return pd.DataFrame(rows)


def summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé par (N, Method): médiane + IQR + taux de succès.
    """
    g = df_raw.groupby(["N", "Method", "Group"], as_index=False)

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
    plt.figure(figsize=(11, 6))

    methods = sorted(summary_df["Method"].unique())
    for method in methods:
        sub = summary_df[summary_df["Method"] == method].sort_values("N")
        x = sub["N"].tolist()
        y = sub["median_time"].tolist()
        yerr_low = (sub["median_time"] - sub["q1"]).tolist()
        yerr_high = (sub["q3"] - sub["median_time"]).tolist()
        plt.errorbar(x, y, yerr=[yerr_low, yerr_high],
                     marker="o", capsize=4, label=method)

    plt.xlabel("Taille du problème (N)")
    plt.ylabel("Temps 1ère solution (s) — médiane + IQR")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_success_rate(summary_df: pd.DataFrame, out_png: str, title: str) -> None:
    plt.figure(figsize=(11, 6))

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
    plt.savefig(out_png, dpi=160)
    plt.close()


def _make_run_dir(runs_dir: Path, args: argparse.Namespace) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{stamp}"
        f"__N{args.n_min}-{args.n_max}"
        f"__step{args.n_step}"
        f"__tl{int(args.time_limit)}"
        f"__rep{args.repeats}"
        f"__sb{int(args.symmetry_breaking)}"
        f"__lns{args.lns_policy}"
        f"__ms{int(args.lns_multistart)}"
        f"__mc{args.mc_pick_policy}"
    )
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark #2: temps pour la première solution (complet vs incomplet), avec runs."
    )

    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--n-step", type=int, default=20,
                        help="Pas sur N (ex: 20 => 20,40,60...)")

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

    # runs directory
    parser.add_argument("--runs-dir", type=str,
                        default="benchmarks/benchmark2/runs")

    args = parser.parse_args()

    if args.n_step <= 0:
        raise ValueError("--n-step doit être > 0")

    ns = list(range(args.n_min, args.n_max + 1, args.n_step))
    run_dir = _make_run_dir(Path(args.runs_dir), args)

    # save config utilisée
    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    df_raw = run_benchmark_time_to_first(
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

    raw_csv = run_dir / "benchmark2_time_to_first_raw.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"\nCSV raw écrit: {raw_csv}")

    df_sum = summarize(df_raw)
    sum_csv = run_dir / "benchmark2_time_to_first_summary.csv"
    df_sum.to_csv(sum_csv, index=False)
    print(f"CSV summary écrit: {sum_csv}")

    title = (
        f"N-reines — Temps 1ère solution (timeout={args.time_limit:.0f}s, repeats={args.repeats}, "
        f"symmetry_breaking={args.symmetry_breaking}, step={args.n_step})"
    )

    out_png_time = run_dir / "benchmark2_time_to_first.png"
    plot_time_to_first(df_sum, str(out_png_time), title)
    print(f"PNG écrit: {out_png_time}")

    out_png_success = run_dir / "benchmark2_success_rate.png"
    plot_success_rate(df_sum, str(out_png_success),
                      f"N-reines — Taux de succès (timeout={args.time_limit:.0f}s)")
    print(f"PNG écrit: {out_png_success}")

    print(f"\nRun terminé: {run_dir}")


if __name__ == "__main__":
    main()
