"""
Wrapper pour votre code CP-SAT existant (main.py)
"""
from ortools.sat.python import cp_model
import time
from ..base_solver import CompleteSolver, SolverResult


class CPSATOriginalSolver(CompleteSolver):
    """Votre implémentation actuelle de main.py"""

    @property
    def method_id(self) -> str:
        return "ORIGINAL"

    @property
    def algorithm_name(self) -> str:
        return "CP-SAT Original (votre code)"

    def solve(self, time_limit: float = 30.0) -> SolverResult:
        """Votre fonction solve_n_queens adaptée"""
        from main import solve_n_queens  # Import de votre code actuel

        start_time = time.time()

        # Appeler votre fonction existante
        solutions, wall_time = solve_n_queens(self.n, max_solutions=None)

        # Convertir au format SolverResult
        unique_hashes = set()
        for sol in solutions:
            unique_hashes.add(self.hash_solution(sol))

        return SolverResult(
            solutions=solutions,
            unique_solutions=unique_hashes,
            execution_time=wall_time,
            time_to_first_solution=wall_time / len(solutions) if solutions else None,
            iterations=len(solutions),
            nodes_explored=0,
            success=len(solutions) > 0,
            metadata={"method_id": "ORIGINAL"}
        )