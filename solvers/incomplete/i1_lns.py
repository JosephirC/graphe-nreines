"""
I1 - CP-SAT + LNS (Large Neighborhood Search)
Méthode INCOMPLÈTE par grands voisinages

Conforme au TODO:
- Type: recherche locale par grands voisinages (incomplète)
- Solveur: OR-Tools CP-SAT
- Incomplétude: OUI (exploration partielle, pas de preuve)
- Principe: solution initiale → itérer (fixer variables, relâcher voisinage, CP-SAT sous limite)
- Paramètres: neighborhood_size, time_limit_per_iteration, max_iterations, seed
- Sorties: solutions valides distinctes en 30s, temps première solution
"""

from ortools.sat.python import cp_model
from typing import Optional, List, Set
import time
import random

from ..base_solver import IncompleteSolver, SolverResult


class I1LNSSolver(IncompleteSolver):
    """
    I1: Solveur Large Neighborhood Search avec CP-SAT
    
    Principe:
    1. Trouver une solution initiale (ou partir d'une assignation)
    2. Itérer:
       a. Fixer une grande partie des variables
       b. Relâcher un sous-ensemble (le "voisinage")
       c. Relancer CP-SAT sur ce sous-problème sous limite de temps
    3. Garder les solutions valides trouvées et compter les distinctes
    
    Paramètres exposés (TODO):
    - neighborhood_size: % de variables relâchées (ou nb de lignes relâchées)
    - time_limit_per_iteration: temps max par réparation
    - max_iterations (ou time_limit global 30s)
    - seed: pour randomisation du voisinage
    
    Sorties benchmark (TODO):
    - solutions valides distinctes trouvées en 30s
    - temps jusqu'à la première solution valide
    """
    
    def __init__(
        self,
        n: int,
        neighborhood_size: float = 0.3,
        time_limit_per_iteration: float = 1.0,
        symmetry_breaking: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialise le solveur I1 LNS
        
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
        
        self.neighborhood_size = neighborhood_size
        self.time_limit_per_iteration = time_limit_per_iteration
        
        if seed is not None:
            random.seed(seed)
    
    @property
    def method_id(self) -> str:
        return "I1"
    
    @property
    def algorithm_name(self) -> str:
        return "CP-SAT + LNS (Large Neighborhood Search)"
    
    def solve(
        self,
        time_limit: float = 30.0,
        max_iterations: Optional[int] = None,
        initial_time_limit: float = 5.0
    ) -> SolverResult:
        """
        Résout avec LNS
        
        Args:
            time_limit: Limite de temps globale (30s benchmark)
            max_iterations: Nombre max d'itérations LNS (None = jusqu'à timeout)
            initial_time_limit: Temps pour solution initiale
            
        Returns:
            SolverResult avec métriques benchmark
        """
        self.logger.info(
            f"[I1] Démarrage LNS pour N={self.n}, "
            f"neighborhood={self.neighborhood_size*100:.0f}%, "
            f"time_limit={time_limit}s"
        )
        
        start_time = time.time()
        self.reset_seen_solutions()
        
        # Tracking des solutions uniques
        unique_solutions: Set[str] = set()
        all_solutions: List[List[int]] = []
        time_to_first: Optional[float] = None
        iterations_done = 0
        total_nodes = 0
        
        # ========== ÉTAPE 1: SOLUTION INITIALE ==========
        current_solution = self._find_initial_solution(initial_time_limit)
        
        if current_solution is None:
            self.logger.warning("[I1] Échec de la recherche de solution initiale")
            return SolverResult(
                solutions=[],
                unique_solutions=set(),
                execution_time=time.time() - start_time,
                time_to_first_solution=None,
                iterations=0,
                nodes_explored=0,
                success=False,
                metadata={
                    "method_id": "I1",
                    "reason": "no_initial_solution",
                    "neighborhood_size": self.neighborhood_size
                }
            )
        
        # Vérifier si solution initiale est valide
        if self.count_conflicts(current_solution) == 0:
            solution_hash = self.hash_solution(current_solution)
            unique_solutions.add(solution_hash)
            all_solutions.append(current_solution)
            time_to_first = time.time() - start_time
            self.logger.info(f"[I1] Solution initiale valide trouvée en {time_to_first:.4f}s")
        else:
            self.logger.info(
                f"[I1] Solution initiale avec {self.count_conflicts(current_solution)} conflits"
            )
        
        best_solution = current_solution
        best_conflicts = self.count_conflicts(best_solution)
        
        # ========== ÉTAPE 2: ITÉRATIONS LNS ==========
        iteration = 0
        while True:
            # Vérifier conditions d'arrêt
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                self.logger.info(f"[I1] Timeout global atteint après {iteration} itérations")
                break
            
            if max_iterations and iteration >= max_iterations:
                self.logger.info(f"[I1] Max iterations atteint: {max_iterations}")
                break
            
            iterations_done = iteration + 1
            
            # Temps restant
            remaining_time = time_limit - elapsed
            iter_time_limit = min(self.time_limit_per_iteration, remaining_time)
            
            # DESTROY: Sélectionner lignes à FIXER (ne pas relâcher)
            rows_to_fix = self._select_rows_to_fix(self.n, self.neighborhood_size)
            
            # REPAIR: Résoudre le sous-problème
            new_solution, nodes = self._repair_solution(
                current_solution,
                rows_to_fix,
                iter_time_limit
            )
            
            total_nodes += nodes
            
            if new_solution is None:
                iteration += 1
                continue
            
            # Évaluer la nouvelle solution
            new_conflicts = self.count_conflicts(new_solution)
            
            # Si solution valide, l'ajouter aux uniques
            if new_conflicts == 0:
                solution_hash = self.hash_solution(new_solution)
                if solution_hash not in unique_solutions:
                    unique_solutions.add(solution_hash)
                    all_solutions.append(new_solution)
                    
                    # Première solution valide?
                    if time_to_first is None:
                        time_to_first = time.time() - start_time
                        self.logger.info(
                            f"[I1] Première solution valide trouvée en {time_to_first:.4f}s"
                        )
                    
                    self.logger.debug(
                        f"[I1] Itération {iteration}: Nouvelle solution valide unique! "
                        f"Total: {len(unique_solutions)}"
                    )
            
            # Accepter si meilleur ou égal (diversification)
            if new_conflicts <= self.count_conflicts(current_solution):
                current_solution = new_solution
                
                if new_conflicts < best_conflicts:
                    best_solution = new_solution
                    best_conflicts = new_conflicts
            
            # Log périodique
            if (iteration + 1) % 10 == 0:
                self.logger.debug(
                    f"[I1] Itération {iteration + 1}, "
                    f"solutions uniques: {len(unique_solutions)}, "
                    f"meilleur: {best_conflicts} conflits"
                )
            
            iteration += 1
        
        execution_time = time.time() - start_time
        success = len(unique_solutions) > 0
        
        self.logger.info(
            f"[I1] Terminé: {len(unique_solutions)} solutions uniques valides, "
            f"{iterations_done} itérations, "
            f"{execution_time:.4f}s"
        )
        
        return SolverResult(
            solutions=all_solutions,
            unique_solutions=unique_solutions,
            execution_time=execution_time,
            time_to_first_solution=time_to_first,
            iterations=iterations_done,
            nodes_explored=total_nodes,
            success=success,
            metadata={
                "method_id": "I1",
                "best_conflicts": best_conflicts,
                "neighborhood_size": self.neighborhood_size,
                "time_limit_per_iteration": self.time_limit_per_iteration,
                "seed": self.seed
            }
        )
    
    def _find_initial_solution(self, time_limit: float) -> Optional[List[int]]:
        """
        Trouve une solution initiale avec CP-SAT
        
        Args:
            time_limit: Temps maximum
            
        Returns:
            Solution initiale ou None
        """
        model = cp_model.CpModel()
        queens = [model.NewIntVar(0, self.n - 1, f'q{i}') for i in range(self.n)]
        
        # Contraintes
        model.AddAllDifferent(queens)
        model.AddAllDifferent([queens[i] + i for i in range(self.n)])
        model.AddAllDifferent([queens[i] - i for i in range(self.n)])
        
        if self.symmetry_breaking:
            model.Add(queens[0] <= (self.n - 1) // 2)
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 1
        
        if self.seed is not None:
            solver.parameters.random_seed = self.seed
        
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return [solver.Value(queens[i]) for i in range(self.n)]
        
        return None
    
    def _select_rows_to_fix(self, n: int, neighborhood_size: float) -> List[int]:
        """
        Sélectionne aléatoirement les lignes à FIXER (ne pas relâcher)
        
        Args:
            n: Taille échiquier
            neighborhood_size: % à relâcher
            
        Returns:
            Liste des indices de lignes à fixer
        """
        num_to_relax = int(n * neighborhood_size)
        num_to_relax = max(1, num_to_relax)
        num_to_relax = min(n - 1, num_to_relax)
        
        num_to_fix = n - num_to_relax
        
        return random.sample(range(n), num_to_fix)
    
    def _repair_solution(
        self,
        current_solution: List[int],
        rows_to_fix: List[int],
        time_limit: float
    ) -> tuple[Optional[List[int]], int]:
        """
        REPAIR: Résout le sous-problème avec lignes fixées
        
        Args:
            current_solution: Solution actuelle
            rows_to_fix: Indices des lignes à fixer
            time_limit: Temps max
            
        Returns:
            (nouvelle solution ou None, nœuds explorés)
        """
        model = cp_model.CpModel()
        queens = [model.NewIntVar(0, self.n - 1, f'q{i}') for i in range(self.n)]
        
        # Contraintes de base
        model.AddAllDifferent(queens)
        model.AddAllDifferent([queens[i] + i for i in range(self.n)])
        model.AddAllDifferent([queens[i] - i for i in range(self.n)])
        
        if self.symmetry_breaking:
            model.Add(queens[0] <= (self.n - 1) // 2)
        
        # Fixer les lignes sélectionnées
        for row in rows_to_fix:
            model.Add(queens[row] == current_solution[row])
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 1
        
        if self.seed is not None:
            solver.parameters.random_seed = self.seed + len(rows_to_fix)
        
        status = solver.Solve(model)
        nodes = solver.NumBranches()
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return [solver.Value(queens[i]) for i in range(self.n)], nodes
        
        return None, nodes


# ========== EXEMPLE D'UTILISATION ==========
if __name__ == "__main__":
    print("=" * 70)
    print("Test I1 - CP-SAT + LNS")
    print("=" * 70)
    
    solver = I1LNSSolver(
        n=20,
        neighborhood_size=0.3,
        time_limit_per_iteration=1.0,
        symmetry_breaking=False,
        seed=42
    )
    
    result = solver.solve(
        time_limit=30.0,
        max_iterations=None,  # Jusqu'à timeout
        initial_time_limit=5.0
    )
    
    print(f"\n{result}")
    print(f"\nMétriques benchmark:")
    print(f"  - Solutions uniques en 30s: {result.num_unique_solutions()}")
    print(f"  - Temps première solution: {result.time_to_first_solution:.4f}s" if result.time_to_first_solution else "  - Aucune solution valide")
    print(f"  - Itérations LNS: {result.iterations}")
    
    if result.solutions:
        print(f"\nPremière solution valide: {result.solutions[0]}")
        solver.print_board(result.solutions[0])
