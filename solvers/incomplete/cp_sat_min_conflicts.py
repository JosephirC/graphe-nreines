from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Set, Tuple

from ..base_solver import IncompleteSolver, SolverResult

Solution = List[int]
SolutionKey = Tuple[int, ...]


class MinConflictsSolver(IncompleteSolver):
    """
    VRAI Min-Conflicts (recherche locale) pour N-Reines.

    Représentation:
      - 1 reine par ligne (toujours)
      - variable = ligne i
      - valeur = colonne sol[i] dans [0..n-1]

    Algorithme:
      - Initialisation aléatoire
      - Tant qu'il reste des conflits:
          - choisir une ligne en conflit
          - déplacer la reine sur une colonne minimisant les conflits
          - (optionnel) bruit: avec prob noise, move aléatoire
      - Multi-start: si pas de solution après max_steps, on redémarre
      - Anytime: continue à chercher d'autres solutions jusqu'au timeout
    """

    def __init__(
        self,
        n: int,
        symmetry_breaking: bool = False,
        seed: Optional[int] = None,
        noise: float = 0.10,
        max_steps: int = 20000,
        pick_policy: str = "random",  # "random" | "max_conflict"
    ):
        super().__init__(n, symmetry_breaking, seed)

        if not (0.0 <= noise <= 1.0):
            raise ValueError(f"noise doit être dans [0,1] (reçu {noise})")
        if max_steps <= 0:
            raise ValueError(f"max_steps doit être > 0 (reçu {max_steps})")

        allowed = {"random", "max_conflict"}
        if pick_policy not in allowed:
            raise ValueError(
                f"pick_policy doit être dans {allowed} (reçu {pick_policy})")

        self.noise = float(noise)
        self.max_steps = int(max_steps)
        self.pick_policy = pick_policy
        self._rng = random.Random(seed)

    @property
    def method_id(self) -> str:
        return "min_conflicts"

    @property
    def algorithm_name(self) -> str:
        return "Local Search - Min-Conflicts (multi-start)"

    # ---------------------------
    # Helpers: conflicts tracking
    # ---------------------------
    def _diag1(self, row: int, col: int) -> int:
        # row - col in [-(n-1) .. (n-1)]
        return row - col

    def _diag2(self, row: int, col: int) -> int:
        # row + col in [0 .. 2n-2]
        return row + col

    def _row_conflicts(
        self,
        row: int,
        col: int,
        col_count: List[int],
        d1_count: Dict[int, int],
        d2_count: List[int],
    ) -> int:
        # conflicts contributed by placing queen at (row,col), excluding itself handled by counts-1
        c = col_count[col] - 1
        c += d1_count[self._diag1(row, col)] - 1
        c += d2_count[self._diag2(row, col)] - 1
        return c

    def _build_counts(self, sol: Solution):
        col_count = [0] * self.n
        d1_count: Dict[int, int] = {}
        d2_count = [0] * (2 * self.n - 1)

        for r, c in enumerate(sol):
            col_count[c] += 1
            k1 = self._diag1(r, c)
            d1_count[k1] = d1_count.get(k1, 0) + 1
            d2_count[self._diag2(r, c)] += 1

        return col_count, d1_count, d2_count

    def _is_conflicting_row(
        self,
        r: int,
        sol: Solution,
        col_count: List[int],
        d1_count: Dict[int, int],
        d2_count: List[int],
    ) -> bool:
        c = sol[r]
        return (
            col_count[c] > 1
            or d1_count[self._diag1(r, c)] > 1
            or d2_count[self._diag2(r, c)] > 1
        )

    # symmetry breaking (partiel) helper
    def _row0_max_col_if_symmetry(self) -> int:
        # Symmetry breaking partiel: impose seulement q0 dans la moitié gauche.
        # (Ça réduit la symétrie miroir gauche-droite, mais ne supprime pas toutes les symétries.)
        return (self.n - 1) // 2

    def _random_initial_solution(self) -> Solution:
        sol = [0] * self.n
        if self.symmetry_breaking:
            # simple symmetry breaking: row 0 in left half
            max_c = (self.n - 1) // 2
            sol[0] = self._rng.randint(0, self._row0_max_col_if_symmetry())
            start = 1
        else:
            start = 0

        for r in range(start, self.n):
            sol[r] = self._rng.randrange(self.n)

        return sol

    def _best_columns_for_row(
        self,
        r: int,
        sol: Solution,
        col_count: List[int],
        d1_count: Dict[int, int],
        d2_count: List[int],
    ) -> List[int]:
        """
        Returns all columns achieving minimal conflicts for row r,
        given current counts (which include current placement of row r).
        """
        # Temporarily remove current queen from counts
        old_c = sol[r]
        col_count[old_c] -= 1
        k1_old = self._diag1(r, old_c)
        d1_count[k1_old] -= 1
        if d1_count[k1_old] == 0:
            del d1_count[k1_old]
        d2_count[self._diag2(r, old_c)] -= 1

        # Evaluate
        best_cols: List[int] = []
        best_score = None

        # symmetry breaking constraint applies only to row 0
        if self.symmetry_breaking and r == 0:
            col_iter = range(0, self._row0_max_col_if_symmetry() + 1)
        else:
            col_iter = range(self.n)

        for c in col_iter:
            # compute conflicts if we place at (r,c) with counts excluding row r
            score = col_count[c] + d1_count.get(
                self._diag1(r, c), 0) + d2_count[self._diag2(r, c)]
            # score is number of other queens attacking this position
            if best_score is None or score < best_score:
                best_score = score
                best_cols = [c]
            elif score == best_score:
                best_cols.append(c)

        # Add queen back at old position to restore counts
        col_count[old_c] += 1
        d1_count[k1_old] = d1_count.get(k1_old, 0) + 1
        d2_count[self._diag2(r, old_c)] += 1

        return best_cols

    def _apply_move(
        self,
        r: int,
        new_c: int,
        sol: Solution,
        col_count: List[int],
        d1_count: Dict[int, int],
        d2_count: List[int],
    ) -> None:
        old_c = sol[r]
        if new_c == old_c:
            return

        # remove old
        col_count[old_c] -= 1
        k1_old = self._diag1(r, old_c)
        d1_count[k1_old] -= 1
        if d1_count[k1_old] == 0:
            del d1_count[k1_old]
        d2_count[self._diag2(r, old_c)] -= 1

        # set new
        sol[r] = new_c
        col_count[new_c] += 1
        k1_new = self._diag1(r, new_c)
        d1_count[k1_new] = d1_count.get(k1_new, 0) + 1
        d2_count[self._diag2(r, new_c)] += 1

    def solve(self, time_limit: float = 45.0) -> SolverResult:
        tag = self.method_id

        symmetry_note = "none"
        if self.symmetry_breaking:
            symmetry_note = "partial(row0_left_half)"

        self.logger.info(
            f"[{tag}] Démarrage Min-Conflicts pour N={self.n}, "
            f"noise={self.noise}, max_steps={self.max_steps}, "
            f"symmetry_breaking={self.symmetry_breaking}, time_limit={time_limit}s"
        )

        start_time = time.time()
        deadline = start_time + time_limit

        unique_keys: Set[SolutionKey] = set()
        solutions: List[Solution] = []

        time_to_first: Optional[float] = None

        total_steps = 0
        restarts = 0

        # Anytime loop: keep restarting until timeout, collect distinct valid solutions
        while time.time() < deadline:
            restarts += 1
            sol = self._random_initial_solution()
            col_count, d1_count, d2_count = self._build_counts(sol)

            # One restart search
            for _ in range(self.max_steps):
                now = time.time()
                if now >= deadline:
                    break

                total_steps += 1

                # find conflicting rows
                conflicted_rows = [r for r in range(self.n) if self._is_conflicting_row(
                    r, sol, col_count, d1_count, d2_count)]

                if not conflicted_rows:
                    # found a valid solution
                    if self.verify_solution(sol):
                        key = self.solution_key(sol)
                        if key not in unique_keys:
                            unique_keys.add(key)
                            solutions.append(sol.copy())
                            if time_to_first is None:
                                time_to_first = now - start_time
                                self.logger.info(
                                    f"[{tag}] Première solution trouvée en {time_to_first:.4f}s"
                                )
                    # restart immediately to sample more solutions
                    break

                # choose a conflicting variable (row)
                if self.pick_policy == "random":
                    r = self._rng.choice(conflicted_rows)
                else:
                    # max-conflict
                    best_rows = []
                    best_c = -1
                    for rr in conflicted_rows:
                        cc = self._row_conflicts(
                            rr, sol[rr], col_count, d1_count, d2_count)
                        if cc > best_c:
                            best_c = cc
                            best_rows = [rr]
                        elif cc == best_c:
                            best_rows.append(rr)
                    r = self._rng.choice(best_rows)

                # choose move
                if self._rng.random() < self.noise:
                    # random move (diversification)
                    if self.symmetry_breaking and r == 0:
                        new_c = self._rng.randint(
                            0, self._row0_max_col_if_symmetry())
                    else:
                        new_c = self._rng.randrange(self.n)
                else:
                    best_cols = self._best_columns_for_row(
                        r, sol, col_count, d1_count, d2_count)
                    new_c = self._rng.choice(best_cols)

                self._apply_move(r, new_c, sol, col_count, d1_count, d2_count)

        execution_time = time.time() - start_time
        success = len(unique_keys) > 0

        self.logger.info(
            f"[{tag}] Terminé: {len(unique_keys)} solutions valides uniques, "
            f"{execution_time:.4f}s, restarts={restarts}, steps={total_steps}"
        )

        return SolverResult(
            solutions=solutions,
            unique_solutions=unique_keys,
            execution_time=execution_time,
            time_to_first_solution=time_to_first,
            iterations=total_steps,          # steps de recherche locale
            nodes_explored=total_steps,      # analogue (pas un arbre)
            success=success,
            metadata={
                "method_id": self.method_id,
                "algorithm_name": self.algorithm_name,
                "n": self.n,
                "symmetry_breaking": self.symmetry_breaking,
                "symmetry_breaking_note": "partial: row0 in left half" if self.symmetry_breaking else "none",
                "pick_policy": self.pick_policy,
                "seed": self.seed,
                "noise": self.noise,
                "max_steps": self.max_steps,
                "restarts": restarts,
                "steps": total_steps,
                "iterations_definition": "1 step = 1 local move (not a search tree node)",
                "nodes_explored_note": "compat_field_only; alias_of_steps",
                "unique_solutions_collected": len(unique_keys),
                "solutions_collected": len(solutions),
            },
        )


CPSatMinConflictsSolver = MinConflictsSolver


if __name__ == "__main__":
    print("=" * 70)
    print("Test Min-Conflicts (Local Search) - N-Queens")
    print("=" * 70)

    solver = CPSatMinConflictsSolver(
        n=14, symmetry_breaking=False, seed=42, noise=0.15, max_steps=30000, pick_policy="max_conflict")
    result = solver.solve(time_limit=45.0)
    print(result)
    print(
        f"\nSolutions valides uniques en 5s: {result.num_unique_solutions()}")
    if result.time_to_first_solution is not None:
        print(f"Temps première solution: {result.time_to_first_solution:.4f}s")
    else:
        print("Aucune solution trouvée")
