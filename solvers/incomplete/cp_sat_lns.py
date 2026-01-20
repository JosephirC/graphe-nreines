from __future__ import annotations
from ortools.sat.python import cp_model
from typing import Optional, List, Set, Tuple
import time
import random


from ..base_solver import IncompleteSolver, SolverResult

Solution = List[int]
SolutionKey = Tuple[int, ...]


class CPSatLNSSolver(IncompleteSolver):
    """
    Solveur Large Neighborhood Search avec CP-SAT

    Principe:
    1. Trouver une solution initiale (ou partir d'une assignation)
    2. Itérer:
       a. Fixer une grande partie des variables
       b. Relâcher un sous-ensemble (le "voisinage")
       c. Relancer CP-SAT sur ce sous-problème sous limite de temps
    3. Garder les solutions valides trouvées et compter les distinctes

    Paramètres exposés :
    - neighborhood_size: % de variables relâchées (ou nb de lignes relâchées)
    - time_limit_per_iteration: temps max par réparation
    - max_iterations (ou time_limit global 45s)
    - seed: pour randomisation du voisinage

    Sorties benchmark :
    - solutions valides distinctes trouvées en 45s
    - temps jusqu'à la première solution valide
    """

    def __init__(
        self,
        n: int,
        neighborhood_size: float = 0.3,
        # "random" | "center_fix" | "edge_fix" | "unstable_relax"
        neighborhood_policy: str = "random",
        time_limit_per_iteration: float = 1.0,
        symmetry_breaking: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialise le solveur LNS

        Args:
            n: Taille de l'échiquier
            neighborhood_size: % de variables à relâcher (0.1-0.5)
            time_limit_per_iteration: Temps max par réparation
            symmetry_breaking: Activer symmetry breaking (optionnel)
            seed: Graine pour randomisation du voisinage
        """
        super().__init__(n, symmetry_breaking, seed)

        if not 0.0 < neighborhood_size < 1.0:
            raise ValueError(
                f"neighborhood_size doit être entre 0 et 1, reçu: {neighborhood_size}"
            )

        if time_limit_per_iteration <= 0.0:
            raise ValueError(
                f"time_limit_per_iteration doit être > 0, reçu: {time_limit_per_iteration}"
            )

        allowed = {"random", "center_fix", "edge_fix", "unstable_relax"}
        if neighborhood_policy not in allowed:
            raise ValueError(
                f"neighborhood_policy doit être dans {allowed} (reçu {neighborhood_policy})")

        self.neighborhood_size = neighborhood_size
        self.time_limit_per_iteration = time_limit_per_iteration
        self.neighborhood_policy = neighborhood_policy
        self.change_count = [0] * n

        self.rng = random.Random(seed)

    @property
    def method_id(self) -> str:
        return "lns"

    @property
    def algorithm_name(self) -> str:
        return "CP-SAT + LNS (Large Neighborhood Search)"

    def solve(
        self,
        time_limit: float = 45.0,
        max_iterations: Optional[int] = None,
        initial_time_limit: float = 5.0
    ) -> SolverResult:
        """
        Résout avec LNS

        Args:
            time_limit: Limite de temps globale (45s benchmark)
            max_iterations: Nombre max d'itérations LNS (None = jusqu'à timeout)
            initial_time_limit: Temps pour solution initiale

        Returns:
            SolverResult avec métriques benchmark
        """
        tag = self.method_id
        self.logger.info(
            f"[{tag}] Démarrage LNS pour N={self.n}, "
            f"neighborhood={self.neighborhood_size*100:.0f}%, "
            f"policy={self.neighborhood_policy}, "
            f"symmetry_breaking={self.symmetry_breaking}, "
            f"time_limit={time_limit}s, per_iter={self.time_limit_per_iteration}s"
        )

        start_time = time.time()
        if self.neighborhood_policy == "unstable_relax":
            self.change_count = [0] * self.n

        # Tracking des solutions uniques
        unique_keys: Set[SolutionKey] = set()
        all_solutions: List[Solution] = []
        time_to_first: Optional[float] = None

        iterations_done = 0
        total_nodes = 0
        initial_nodes = 0
        repair_nodes = 0
        repairs_success = 0
        repairs_fail = 0

        # ========== ÉTAPE 1: SOLUTION INITIALE ==========
        current_solution, nodes = self._find_initial_solution(
            time_limit=initial_time_limit)

        initial_nodes += nodes
        total_nodes += nodes

        if current_solution is None:
            self.logger.warning(
                f"[{tag}] Échec de la recherche de solution initiale")
            return SolverResult(
                solutions=[],
                unique_solutions=set(),
                execution_time=time.time() - start_time,
                time_to_first_solution=None,
                iterations=0,
                nodes_explored=total_nodes,
                success=False,
                metadata={
                    "method_id": self.method_id,
                    "algorithm_name": self.algorithm_name,
                    "n": self.n,
                    "symmetry_breaking": self.symmetry_breaking,
                    "reason": "no_initial_solution",
                    "neighborhood_size": self.neighborhood_size,
                    "neighborhood_policy": self.neighborhood_policy,
                    "time_limit_per_iteration": self.time_limit_per_iteration,
                    "initial_time_limit": initial_time_limit,
                    "max_iterations": max_iterations,
                    "seed": self.seed,
                },
            )

        # Avec les contraintes CP-SAT complètes, toute solution trouvée est valide.
        key0 = self.solution_key(current_solution)
        unique_keys.add(key0)
        all_solutions.append(current_solution)
        time_to_first = time.time() - start_time
        self.logger.info(
            f"[{tag}] Solution initiale trouvée en {time_to_first:.4f}s")

        # ========== ÉTAPE 2: ITÉRATIONS LNS ==========
        iteration = 0
        while True:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                self.logger.info(
                    f"[{tag}] Timeout global atteint après {iterations_done} itérations")
                break
            if max_iterations is not None and iteration >= max_iterations:
                self.logger.info(
                    f"[{tag}] Max iterations atteint: {max_iterations}")
                break

            iterations_done += 1

            remaining_time = time_limit - elapsed
            iter_time_limit = min(
                self.time_limit_per_iteration, remaining_time)
            if iter_time_limit <= 0:
                break

            rows_to_fix = self._select_rows_to_fix(
                n=self.n, neighborhood_size=self.neighborhood_size)

            new_solution, nodes = self._repair_solution(
                current_solution=current_solution,
                rows_to_fix=rows_to_fix,
                time_limit=iter_time_limit,
                iteration=iteration,
            )
            repair_nodes += nodes
            total_nodes += nodes

            if new_solution is not None:
                repairs_success += 1
                # Comptage des solutions distinctes
                key = self.solution_key(new_solution)
                if key not in unique_keys:
                    unique_keys.add(key)
                    all_solutions.append(new_solution)

                # Stratégie humaine: apprendre ce qui bouge
                if self.neighborhood_policy == "unstable_relax":
                    for i in range(self.n):
                        if new_solution[i] != current_solution[i]:
                            self.change_count[i] += 1

                # Diversification simple: on prend la dernière solution trouvée
                current_solution = new_solution
            else:
                repairs_fail += 1

            if (iteration + 1) % 10 == 0:
                self.logger.debug(
                    f"[{tag}] it={iteration+1}, uniques={len(unique_keys)}, nodes={total_nodes}"
                )

            iteration += 1

        execution_time = time.time() - start_time
        success = len(unique_keys) > 0

        self.logger.info(
            f"[{tag}] Terminé: {len(unique_keys)} solutions uniques, "
            f"{iterations_done} itérations, {execution_time:.4f}s"
        )

        return SolverResult(
            solutions=all_solutions,
            unique_solutions=unique_keys,
            execution_time=execution_time,
            time_to_first_solution=time_to_first,
            iterations=iterations_done,
            nodes_explored=total_nodes,
            success=success,
            metadata={
                "method_id": self.method_id,
                "algorithm_name": self.algorithm_name,
                "n": self.n,
                "symmetry_breaking": self.symmetry_breaking,
                "neighborhood_size": self.neighborhood_size,
                "neighborhood_policy": self.neighborhood_policy,
                "time_limit_per_iteration": self.time_limit_per_iteration,
                "initial_time_limit": initial_time_limit,
                "max_iterations": max_iterations,
                "seed": self.seed,
                "solutions_collected": len(all_solutions),
                "unique_solutions_collected": len(unique_keys),
                "nodes_initial": initial_nodes,
                "nodes_repair": repair_nodes,
                "iterations": iterations_done,
                "repairs_success": repairs_success,
                "repairs_fail": repairs_fail,
            },
        )

    # ---------------------------
    # Internals
    # ---------------------------

    def _build_base_model(self) -> tuple[cp_model.CpModel, List[cp_model.IntVar]]:
        model = cp_model.CpModel()
        queens = [model.NewIntVar(
            0, self.n - 1, f"q{i}") for i in range(self.n)]

        model.AddAllDifferent(queens)
        model.AddAllDifferent([queens[i] + i for i in range(self.n)])
        model.AddAllDifferent([queens[i] - i for i in range(self.n)])

        if self.symmetry_breaking:
            model.Add(queens[0] <= (self.n - 1) // 2)

        return model, queens

    def _configure_solver(self, time_limit: float, seed: Optional[int]) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 1
        if seed is not None:
            solver.parameters.random_seed = seed
        return solver

    def _find_initial_solution(self, time_limit: float) -> tuple[Optional[Solution], int]:
        model, queens = self._build_base_model()
        solver = self._configure_solver(time_limit=time_limit, seed=self.seed)

        status = solver.Solve(model)
        nodes = solver.NumBranches()

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sol = [solver.Value(queens[i]) for i in range(self.n)]
            return sol, nodes
        return None, nodes

    def _select_rows_to_fix(self, n: int, neighborhood_size: float) -> List[int]:
        """
         Choisit les lignes à FIXER pour l'étape de repair.

        Important:
          - relax_fraction = fraction de lignes RELÂCHÉES (voisinage)
          - la fonction renvoie la liste des lignes FIXÉES
        """
        num_to_relax = int(n * neighborhood_size)
        num_to_relax = max(1, num_to_relax)
        num_to_relax = min(n - 1, num_to_relax)

        num_to_fix = n - num_to_relax

        if self.neighborhood_policy == "random":
            fixed_rows = self.rng.sample(range(n), num_to_fix)
            return fixed_rows

        center = (n - 1) / 2.0
        idx = list(range(n))

        if self.neighborhood_policy == "center_fix":
            # fixer d'abord les variables centrales
            idx.sort(key=lambda i: abs(i - center))
            fixed_rows = idx[:num_to_fix]
            return fixed_rows

        if self.neighborhood_policy == "edge_fix":
            # fixer d'abord les variables de bord
            idx.sort(key=lambda i: -abs(i - center))
            fixed_rows = idx[:num_to_fix]
            return fixed_rows

        if self.neighborhood_policy == "unstable_relax":
            # relâcher les plus instables => fixer le reste
            order = sorted(
                range(n), key=lambda i: self.change_count[i], reverse=True)
            to_relax = set(order[:num_to_relax])
            fixed_rows = [i for i in range(n) if i not in to_relax]
            return fixed_rows

        raise ValueError(
            f"Unknown neighborhood_policy={self.neighborhood_policy}")

    def _repair_solution(
        self,
        current_solution: Solution,
        rows_to_fix: List[int],
        time_limit: float,
        iteration: int
    ) -> tuple[Optional[Solution], int]:
        """
        REPAIR: Résout le sous-problème avec lignes fixées

        Args:
            current_solution: Solution actuelle
            rows_to_fix: Indices des lignes à fixer
            time_limit: Temps max

        Returns:
            (nouvelle solution ou None, nœuds explorés)
        """
        model, queens = self._build_base_model()

        for row in rows_to_fix:
            model.Add(queens[row] == current_solution[row])

        # Seed CP-SAT: dérivée de la seed + itération (reproductible, varie vraiment)
        solver_seed = None
        if self.seed is not None:
            solver_seed = self.seed + iteration + 1

        solver = self._configure_solver(
            time_limit=time_limit, seed=solver_seed)
        status = solver.Solve(model)
        nodes = solver.NumBranches()

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sol = [solver.Value(queens[i]) for i in range(self.n)]
            return sol, nodes
        return None, nodes


# ========== EXEMPLE D'UTILISATION ==========
if __name__ == "__main__":
    print("=" * 70)
    print("Test LNS - CP-SAT + LNS")
    print("=" * 70)

    solver = CPSatLNSSolver(
        n=20,
        neighborhood_size=0.3,
        time_limit_per_iteration=1.0,
        symmetry_breaking=False,
        seed=42,
    )

    result = solver.solve(
        time_limit=45.0,
        max_iterations=None,
        initial_time_limit=5.0,
    )

    print(f"\n{result}")
    print("\nMétriques benchmark:")
    print(f"  - Solutions uniques en 45s: {result.num_unique_solutions()}")
    if result.time_to_first_solution is not None:
        print(
            f"  - Temps première solution: {result.time_to_first_solution:.4f}s")
    else:
        print("  - Aucune solution valide")
    print(f"  - Itérations LNS: {result.iterations}")

    if result.solutions:
        print(f"\nPremière solution: {result.solutions[0]}")
        solver.print_board(result.solutions[0])
