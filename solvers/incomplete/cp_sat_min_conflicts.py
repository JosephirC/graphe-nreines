from __future__ import annotations

from ortools.sat.python import cp_model
from typing import Optional, List, Set, Tuple, Dict, Iterable
import time

from ..base_solver import IncompleteSolver, SolverResult

Solution = List[int]
SolutionKey = Tuple[int, ...]


_STATUS_MAP = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


class CPSatMinConflictsSolver(IncompleteSolver):
    """
    CP-SAT Min-Conflicts via contraintes molles (optimisation).

    Idée:
    - On autorise des violations (soft constraints) sur certains types de conflits.
    - On minimise une somme pondérée de booléens de conflits.
    - Une solution "valide N-reines" = 0 conflit (colonnes + diagonales).

    Note importante:
    - Si un type n'est PAS dans conflict_types, il est imposé en contrainte dure.
      Exemple: conflict_types=("diagonals",) => colonnes en dur (AllDifferent).
    """

    def __init__(
        self,
        n: int,
        conflict_types: Optional[Iterable[str]] = ("columns", "diagonals"),
        conflict_weights: Optional[Dict[str, int]] = None,
        symmetry_breaking: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(n, symmetry_breaking, seed)

        if conflict_types is None:
            conflict_types = ("columns", "diagonals")

        self.conflict_types = tuple(conflict_types)

        allowed = {"columns", "diagonals"}
        unknown = set(self.conflict_types) - allowed
        if unknown:
            raise ValueError(
                f"conflict_types inconnus: {sorted(unknown)} (attendus: {sorted(allowed)})")

        # Poids par défaut + merge user
        base_weights = {
            "columns": 1,
            "diag_desc": 1,  # ↘ : (row + col) égal
            "diag_asc": 1,   # ↙ : (row - col) égal
        }
        if conflict_weights:
            base_weights.update(conflict_weights)

        # Validation simple
        for k, v in base_weights.items():
            if not isinstance(v, int) or v <= 0:
                raise ValueError(
                    f"conflict_weights[{k!r}] doit être un int > 0 (reçu: {v!r})")

        self.conflict_weights = base_weights

    @property
    def method_id(self) -> str:
        return "min_conflicts"

    @property
    def algorithm_name(self) -> str:
        return "CP-SAT + Min-Conflicts (soft constraints)"

    def solve(self, time_limit: float = 45.0) -> SolverResult:
        tag = self.method_id
        self.logger.info(
            f"[{tag}] Démarrage Min-Conflicts pour N={self.n}, "
            f"conflict_types={list(self.conflict_types)}, time_limit={time_limit}s"
        )

        start_time = time.time()

        # Tracking solutions valides distinctes
        unique_keys: Set[SolutionKey] = set()
        all_solutions: List[Solution] = []

        # -----------------------
        # Modèle
        # -----------------------
        model = cp_model.CpModel()
        queens = [model.NewIntVar(
            0, self.n - 1, f"q{i}") for i in range(self.n)]

        # Symmetry breaking (optionnel)
        if self.symmetry_breaking:
            model.Add(queens[0] <= (self.n - 1) // 2)

        conflict_terms: List[cp_model.LinearExpr] = []

        # Colonnes: soit soft (bools), soit hard (AllDifferent)
        if "columns" in self.conflict_types:
            w_col = self.conflict_weights.get("columns", 1)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    b = model.NewBoolVar(f"col_conf_{i}_{j}")
                    model.Add(queens[i] == queens[j]).OnlyEnforceIf(b)
                    model.Add(queens[i] != queens[j]).OnlyEnforceIf(b.Not())
                    conflict_terms.append(w_col * b)
        else:
            model.AddAllDifferent(queens)

        # Diagonales: soit soft (bools), soit hard (AllDifferent sur expr)
        if "diagonals" in self.conflict_types:
            w_desc = self.conflict_weights.get("diag_desc", 1)
            w_asc = self.conflict_weights.get("diag_asc", 1)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    b_desc = model.NewBoolVar(f"diag_desc_conf_{i}_{j}")
                    model.Add(queens[i] + i == queens[j] +
                              j).OnlyEnforceIf(b_desc)
                    model.Add(queens[i] + i != queens[j] +
                              j).OnlyEnforceIf(b_desc.Not())
                    conflict_terms.append(w_desc * b_desc)

                    b_asc = model.NewBoolVar(f"diag_asc_conf_{i}_{j}")
                    model.Add(queens[i] - i == queens[j] -
                              j).OnlyEnforceIf(b_asc)
                    model.Add(queens[i] - i != queens[j] -
                              j).OnlyEnforceIf(b_asc.Not())
                    conflict_terms.append(w_asc * b_asc)
        else:
            model.AddAllDifferent([queens[i] + i for i in range(self.n)])
            model.AddAllDifferent([queens[i] - i for i in range(self.n)])

        # Objectif si on a des conflits "soft"
        if conflict_terms:
            model.Minimize(sum(conflict_terms))

        # -----------------------
        # Solveur + callback
        # -----------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 1
        if self.seed is not None:
            solver.parameters.random_seed = self.seed

        cb = _ValidSolutionCollector(
            queens=queens,
            solver_instance=self,
            start_time=start_time,
            unique_keys=unique_keys,
            solutions=all_solutions,
        )

        self.logger.info(f"[{tag}] Lancement de l'optimisation...")
        status = solver.Solve(model, cb)

        execution_time = time.time() - start_time
        num_branches = solver.NumBranches()
        num_conflicts_solver = solver.NumConflicts()
        status_str = _STATUS_MAP.get(status, "UNKNOWN")

        objective_value = None
        # Si pas d'objectif, ObjectiveValue() vaut 0.0 mais ce n'est pas très informatif.
        if conflict_terms and status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            objective_value = solver.ObjectiveValue()

        success = len(unique_keys) > 0

        self.logger.info(
            f"[{tag}] Terminé: {len(unique_keys)} solutions valides uniques, "
            f"objectif_final={objective_value}, {execution_time:.4f}s"
        )

        return SolverResult(
            solutions=all_solutions,
            unique_solutions=unique_keys,
            execution_time=execution_time,
            time_to_first_solution=cb.time_to_first,
            iterations=len(all_solutions),
            nodes_explored=num_branches,
            success=success,
            metadata={
                "method_id": self.method_id,
                "status": status_str,
                "num_branches": num_branches,
                "num_conflicts_solver": num_conflicts_solver,
                "objective_value": objective_value,
                "conflict_types": list(self.conflict_types),
                "conflict_weights": self.conflict_weights,
                "symmetry_breaking": self.symmetry_breaking,
                "solutions_collected": len(all_solutions),
                "unique_solutions_collected": len(unique_keys),
            },
        )


class _ValidSolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    Collecte uniquement les solutions N-reines VALIDE (0 conflit total).
    """

    def __init__(
        self,
        queens: List[cp_model.IntVar],
        solver_instance: CPSatMinConflictsSolver,
        start_time: float,
        unique_keys: Set[SolutionKey],
        solutions: List[Solution],
    ):
        super().__init__()
        self._queens = queens
        self._solver = solver_instance
        self._start_time = start_time
        self._unique_keys = unique_keys
        self._solutions = solutions
        self.time_to_first: Optional[float] = None

    def OnSolutionCallback(self) -> None:
        sol = [self.Value(v) for v in self._queens]

        # On ne garde que les solutions valides N-reines
        if not self._solver.verify_solution(sol):
            return

        key = self._solver.solution_key(sol)
        if key in self._unique_keys:
            return

        self._unique_keys.add(key)
        self._solutions.append(sol)

        if self.time_to_first is None:
            self.time_to_first = time.time() - self._start_time
            tag = self._solver.method_id
            self._solver.logger.info(
                f"[{tag}] Première solution VALIDE trouvée en {self.time_to_first:.4f}s"
            )


if __name__ == "__main__":
    print("=" * 70)
    print("Test Min-Conflicts - CP-SAT (soft constraints)")
    print("=" * 70)

    # Exemple 1: diagonales en soft, colonnes en dur
    solver = CPSatMinConflictsSolver(
        n=8,
        conflict_types=("diagonals",),
        conflict_weights={"diag_desc": 2, "diag_asc": 2},
        symmetry_breaking=False,
        seed=42,
    )
    result = solver.solve(time_limit=45.0)
    print(result)
    print("\nMétriques benchmark:")
    print(
        f"  - Solutions valides uniques en 45s: {result.num_unique_solutions()}")
    if result.time_to_first_solution is not None:
        print(
            f"  - Temps première solution valide: {result.time_to_first_solution:.4f}s")
    else:
        print("  - Aucune solution valide trouvée")

    if result.solutions:
        solver.print_board(result.solutions[0])
