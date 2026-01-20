from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from solvers.complete.cp_sat_fixed_search_center_out import CPSatCenterOutSolver
from solvers.complete.cp_sat_fixed_search_first_fail import CPSatFirstFailSolver
from solvers.incomplete.cp_sat_lns import CPSatLNSSolver
from solvers.incomplete.cp_sat_min_conflicts import CPSatMinConflictsSolver


logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(message)s")


@dataclass(frozen=True)
class SolverCase:
    name: str
    kind: str  # "complete" | "incomplete"
    factory: Callable[[], Any]
    time_limit: float


def _format_result(case: SolverCase, result: Any) -> str:
    """
    Result expected to be SolverResult-like:
      - execution_time (float)
      - num_unique_solutions() (int)
    Some incomplete solvers may only return 1 solution; that's normal.
    """
    nsol = result.num_unique_solutions() if hasattr(
        result, "num_unique_solutions") else None
    t = getattr(result, "execution_time", None)

    if case.kind == "complete":
        # For complete solvers, 'nsol' is meaningful (enumeration)
        return f"{nsol} solutions uniques, {t:.4f}s"
    else:
        # For incomplete solvers, we mainly care about "found something" + time
        # nsol will usually be 1 (or small), that's fine.
        return f"{nsol} solution(s) unique(s), {t:.4f}s"


def run_case(case: SolverCase) -> None:
    print(f"\n--- {case.name} ({case.kind}) ---")
    solver = case.factory()
    result = solver.solve(time_limit=case.time_limit)
    print(f"✓ OK: {_format_result(case, result)}")


def main() -> None:
    n = 8
    seed = 42

    cases = [
        SolverCase(
            name="CP-SAT Complete - Fixed Search (First-Fail)",
            kind="complete",
            factory=lambda: CPSatFirstFailSolver(
                n=n, symmetry_breaking=False, seed=seed),
            time_limit=5.0,
        ),
        SolverCase(
            name="CP-SAT Complete - Fixed Search (Center-Out)",
            kind="complete",
            factory=lambda: CPSatCenterOutSolver(
                n=n, symmetry_breaking=False, seed=seed),
            time_limit=5.0,
        ),
        SolverCase(
            name="CP-SAT Incomplete - LNS",
            kind="incomplete",
            factory=lambda: CPSatLNSSolver(
                n=n, neighborhood_size=0.3, seed=seed),
            time_limit=2.0,  # inutile de cramer 5s pour un test d'archi
        ),
        SolverCase(
            name="CP-SAT Incomplete - Min-Conflicts",
            kind="incomplete",
            factory=lambda: CPSatMinConflictsSolver(n=n, seed=seed),
            time_limit=2.0,
        ),
    ]

    print("=" * 60)
    print("ARCHITECTURE SMOKE TEST")
    print("=" * 60)

    for case in cases:
        run_case(case)

    print("\n✅ Tous les solveurs se lancent et retournent un résultat.")


if __name__ == "__main__":
    main()
