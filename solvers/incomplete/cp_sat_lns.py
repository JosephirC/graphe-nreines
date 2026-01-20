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
    1. Trouver une solution initiale
    2. Itérer:
       a. Fixer une grande partie des variables
       b. Relâcher un sous-ensemble (le "voisinage")
       c. Relancer CP-SAT sur ce sous-problème sous limite de temps
    3. Garder les solutions valides trouvées et compter les distinctes

    Paramètres exposés :
    - neighborhood_size: fraction de variables relâchées (0 < size < 1)
    - time_limit_per_iteration: temps max par réparation
    - max_iterations : nombre max d'itérations LNS par run
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
        initial_time_limit: float = 5.0,
        multistart: bool = False,
    ) -> SolverResult:
        """
        Résout avec LNS

        Args:
            time_limit: Limite de temps globale (45s benchmark)
            max_iterations: Nombre max d'itérations LNS (None = jusqu'à timeout)
            initial_time_limit: Temps pour solution initiale
            multistart: Activer multistart (recherche de plusieurs solutions itérativement)
        Returns:
            SolverResult avec métriques benchmark
        """
        tag = self.method_id
        self.logger.info(
            f"[{tag}] Démarrage LNS pour N={self.n}, "
            f"neighborhood={self.neighborhood_size*100:.0f}%, "
            f"policy={self.neighborhood_policy}, "
            f"symmetry_breaking={self.symmetry_breaking}, "
            f"multistart={multistart}, "
            f"time_limit={time_limit}s, per_iter={self.time_limit_per_iteration}s"
        )

        start_time = time.time()
        deadline = start_time + time_limit

        # reset apprentissage si besoin
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

        restarts = 0
        global_iteration = 0

        STAGNATION_THRESHOLD = 200

        # Boucle multi-start: si multistart=False -> un seul run (break à la fin)
        while time.time() < deadline:
            restarts += 1

            # --- Trouver une solution initiale (budget = min(initial_time_limit, temps restant)) ---
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            init_tl = min(initial_time_limit, remaining)
            current_solution, nodes = self._find_initial_solution(
                time_limit=init_tl)

            initial_nodes += nodes
            total_nodes += nodes

            if current_solution is None:
                # si multistart, on réessaye tant que le temps n'est pas fini
                if multistart:
                    continue

                # sinon, échec global immédiat (comportement précédent)
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
                        "multistart": multistart,
                        "seed": self.seed,
                    },
                )

            # enregistrer solution initiale
            key0 = self.solution_key(current_solution)
            if key0 not in unique_keys:
                unique_keys.add(key0)
                all_solutions.append(current_solution)

            if time_to_first is None:
                time_to_first = time.time() - start_time
                self.logger.info(
                    f"[{tag}] Première solution trouvée en {time_to_first:.4f}s")

            # --- LNS iterations pour ce run ---
            local_iteration = 0
            no_progress = 0  # nb d'itérations sans nouvelle solution unique

            while time.time() < deadline:
                if max_iterations is not None and local_iteration >= max_iterations:
                    break

                remaining = deadline - time.time()
                iter_time_limit = min(self.time_limit_per_iteration, remaining)
                if iter_time_limit <= 0:
                    break

                iterations_done += 1
                global_iteration += 1
                local_iteration += 1

                rows_to_fix = self._select_rows_to_fix(
                    n=self.n, neighborhood_size=self.neighborhood_size
                )

                new_solution, nodes = self._repair_solution(
                    current_solution=current_solution,
                    rows_to_fix=rows_to_fix,
                    time_limit=iter_time_limit,
                    iteration=global_iteration,  # seed varie globalement
                )

                repair_nodes += nodes
                total_nodes += nodes

                if new_solution is not None:
                    repairs_success += 1

                    key = self.solution_key(new_solution)
                    if key not in unique_keys:
                        unique_keys.add(key)
                        all_solutions.append(new_solution)
                        no_progress = 0
                    else:
                        no_progress += 1

                    # Stratégie humaine: apprendre ce qui bouge
                    if self.neighborhood_policy == "unstable_relax":
                        for i in range(self.n):
                            if new_solution[i] != current_solution[i]:
                                self.change_count[i] += 1

                    current_solution = new_solution
                else:
                    repairs_fail += 1
                    no_progress += 1

                if (global_iteration % 10) == 0:
                    self.logger.debug(
                        f"[{tag}] it={global_iteration}, uniques={len(unique_keys)}, nodes={total_nodes}, restarts={restarts}"
                    )

                # Si multistart: on restart quand on stagne trop
                if multistart and no_progress >= STAGNATION_THRESHOLD:
                    break

            # Si pas multistart: on ne fait qu'un run, puis sortie
            if not multistart:
                break

        execution_time = time.time() - start_time
        success = len(unique_keys) > 0

        # si multistart=True et aucune init n'a jamais réussi
        if not success:
            self.logger.warning(
                f"[{tag}] Aucune solution trouvée avant timeout")

        self.logger.info(
            f"[{tag}] Terminé: {len(unique_keys)} solutions uniques, "
            f"{iterations_done} itérations, restarts={restarts}, temps={execution_time:.4f}s"
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
                "multistart": multistart,
                "seed": self.seed,
                "solutions_collected": len(all_solutions),
                "unique_solutions_collected": len(unique_keys),
                "nodes_initial": initial_nodes,
                "nodes_repair": repair_nodes,
                "iterations": iterations_done,
                "repairs_success": repairs_success,
                "repairs_fail": repairs_fail,
                "restarts": restarts,
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
          - neighborhood_size = fraction de lignes RELÂCHÉES (voisinage)
          - la fonction renvoie la liste des lignes FIXÉES.
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

    n = 30
    neighborhood_size = 0.3
    neighborhood_policy = "center_fix"
    time_limit_per_iteration = 0.1
    multistart = True
    seed = 42

    solver = CPSatLNSSolver(
        n=30,
        neighborhood_size=0.3,
        neighborhood_policy="center_fix",
        time_limit_per_iteration=0.1,
        symmetry_breaking=False,
        seed=42,
    )

    result = solver.solve(
        time_limit=45.0,
        max_iterations=None,
        initial_time_limit=1.0,
        multistart=True,
    )

    print(f"Parameters: n={n}, neighborhood_size={neighborhood_size}, "
          f"policy={neighborhood_policy}, time_limit_per_iteration={time_limit_per_iteration}, "
          f"multistart={multistart}, seed={seed}")

    print(f"\n{result}")
    print("\nMétriques benchmark:")
    print(f"  - Solutions uniques en 45s: {result.num_unique_solutions()}")
    if result.time_to_first_solution is not None:
        print(
            f"  - Temps première solution: {result.time_to_first_solution:.4f}s")
    else:
        print("  - Aucune solution valide")
    print(f"  - Itérations LNS: {result.iterations}")
