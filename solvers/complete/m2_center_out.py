"""
M2 - Ordre fixe structurel (center-out)
Méthode COMPLÈTE avec stratégie center-out

Conforme au TODO:
- Type: stratégie de recherche (branching à ordre imposé)
- Solveur: OR-Tools CP-SAT
- Complétude: OUI (backtracking exhaustif)
- search_branching = FIXED_SEARCH
- Variables ordonnées selon schéma center-out: mid, mid-1, mid+1, mid-2, mid+2, ...
- AddDecisionStrategy(x_center_out, CHOOSE_FIRST, SELECT_MIN_VALUE)
- Principe: Exploiter la structure géométrique, contraindre diagonales dès le début
"""

from ortools.sat.python import cp_model
from typing import List
import time

from ..base_solver import CompleteSolver, SolverResult


class M2CenterOutSolver(CompleteSolver):
    """
    M2: Solveur avec stratégie Center-Out
    
    Principe:
    - Exploiter la structure géométrique du problème N-Reines
    - Placer d'abord les reines au centre, puis vers les bords
    - Contraindre fortement les diagonales dès le début
    
    Ordre center-out pour N=8:
    mid=3 → ordre: 3, 4, 2, 5, 1, 6, 0, 7
    (ou: 4, 3, 5, 2, 6, 1, 7, 0 si mid=4)
    
    Avantages:
    - Contraintes diagonales plus tôt
    - Peut être plus efficace pour certaines instances
    
    Paramètres:
    - symmetry_breaking: x[0] <= (n-1)//2
    """
    
    @property
    def method_id(self) -> str:
        return "M2"
    
    @property
    def algorithm_name(self) -> str:
        return "Center-Out (fixed structural order)"
    
    def _compute_center_out_order(self) -> List[int]:
        """
        Calcule l'ordre center-out pour N reines
        
        Exemple pour N=8:
        mid = 3 → [3, 4, 2, 5, 1, 6, 0, 7]
        
        Exemple pour N=9:
        mid = 4 → [4, 5, 3, 6, 2, 7, 1, 8, 0]
        
        Returns:
            Liste des indices dans l'ordre center-out
        """
        mid = self.n // 2
        order = []
        
        # Alterner entre mid, mid+1, mid-1, mid+2, mid-2, ...
        for offset in range(self.n):
            if offset == 0:
                order.append(mid)
            elif offset % 2 == 1:  # Impair: à droite
                idx = mid + (offset + 1) // 2
                if idx < self.n:
                    order.append(idx)
            else:  # Pair: à gauche
                idx = mid - offset // 2
                if idx >= 0:
                    order.append(idx)
        
        self.logger.debug(f"Ordre center-out pour N={self.n}: {order}")
        return order
    
    def solve(self, time_limit: float = 30.0) -> SolverResult:
        """
        Résout avec stratégie Center-Out
        
        Args:
            time_limit: Limite de temps en secondes (30s benchmark)
            
        Returns:
            SolverResult avec métriques benchmark
        """
        self.logger.info(
            f"[M2] Démarrage Center-Out pour N={self.n}, "
            f"symmetry_breaking={self.symmetry_breaking}, "
            f"time_limit={time_limit}s"
        )
        
        start_time = time.time()
        self.reset_seen_solutions()
        
        # ========== CRÉATION DU MODÈLE CP-SAT ==========
        model = cp_model.CpModel()
        
        # Variables: queens[i] = colonne de la reine dans la ligne i
        queens = [model.NewIntVar(0, self.n - 1, f'q{i}') for i in range(self.n)]
        
        # ========== CONTRAINTES ==========
        # 1. Colonnes différentes
        model.AddAllDifferent(queens)
        
        # 2. Diagonales ↘ différentes
        model.AddAllDifferent([queens[i] + i for i in range(self.n)])
        
        # 3. Diagonales ↙ différentes
        model.AddAllDifferent([queens[i] - i for i in range(self.n)])
        
        # 4. Symmetry breaking (optionnel)
        if self.symmetry_breaking:
            model.Add(queens[0] <= (self.n - 1) // 2)
            self.logger.debug(f"Symmetry breaking activé: x[0] <= {(self.n - 1) // 2}")
        
        # ========== STRATÉGIE DE RECHERCHE M2 ==========
        # Center-Out: ordre fixe structurel
        center_out_order = self._compute_center_out_order()
        queens_center_out = [queens[i] for i in center_out_order]
        
        model.AddDecisionStrategy(
            queens_center_out,
            cp_model.CHOOSE_FIRST,       # Ordre imposé (center-out)
            cp_model.SELECT_MIN_VALUE    # Valeur minimale
        )
        
        # ========== CONFIGURATION SOLVEUR ==========
        solver = cp_model.CpSolver()
        
        # FIXED_SEARCH: utiliser strictement la stratégie définie
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        
        # Limite de temps
        solver.parameters.max_time_in_seconds = time_limit
        
        # num_workers = 1 pour reproductibilité (TODO)
        solver.parameters.num_search_workers = 1
        
        # Seed pour reproductibilité
        if self.seed is not None:
            solver.parameters.random_seed = self.seed
        
        # ========== COLLECTEUR DE SOLUTIONS ==========
        collector = _M2SolutionCollector(
            queens,
            self,
            start_time,
            time_limit
        )
        
        # ========== RECHERCHE ==========
        self.logger.info("[M2] Lancement de la recherche exhaustive...")
        status = solver.SearchForAllSolutions(model, collector)
        
        execution_time = time.time() - start_time
        
        # ========== STATISTIQUES ==========
        num_branches = solver.NumBranches()
        num_conflicts = solver.NumConflicts()
        
        self.logger.info(
            f"[M2] Terminé: {collector.num_unique} solutions uniques, "
            f"{execution_time:.4f}s, {num_branches} branches, {num_conflicts} conflits"
        )
        
        return SolverResult(
            solutions=collector.solutions,
            unique_solutions=collector.unique_hashes,
            execution_time=execution_time,
            time_to_first_solution=collector.time_to_first,
            iterations=len(collector.solutions),
            nodes_explored=num_branches,
            success=len(collector.solutions) > 0,
            metadata={
                "method_id": "M2",
                "status": self._status_to_string(status),
                "num_branches": num_branches,
                "num_conflicts": num_conflicts,
                "symmetry_breaking": self.symmetry_breaking,
                "search_branching": "FIXED_SEARCH",
                "strategy": "CHOOSE_FIRST (center-out order) + SELECT_MIN_VALUE",
                "center_out_order": center_out_order
            }
        )
    
    def _status_to_string(self, status) -> str:
        """Convertit le statut CP-SAT en string"""
        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN"
        }
        return status_map.get(status, "UNKNOWN")


class _M2SolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    Collecteur de solutions pour M2
    Gère les solutions uniques et le timing
    """
    
    def __init__(self, variables, solver_instance, start_time, time_limit):
        super().__init__()
        self._variables = variables
        self._solver = solver_instance
        self._start_time = start_time
        self._time_limit = time_limit
        
        self.solutions = []
        self.unique_hashes = set()
        self.num_unique = 0
        self.time_to_first = None
        
    def OnSolutionCallback(self):
        """Appelé à chaque solution trouvée"""
        solution = [self.Value(v) for v in self._variables]
        
        # Hash pour anti-doublons
        solution_hash = self._solver.hash_solution(solution)
        
        if solution_hash not in self.unique_hashes:
            self.unique_hashes.add(solution_hash)
            self.solutions.append(solution)
            self.num_unique += 1
            
            # Temps pour première solution
            if self.time_to_first is None:
                self.time_to_first = time.time() - self._start_time
                self._solver.logger.info(
                    f"[M2] Première solution trouvée en {self.time_to_first:.4f}s"
                )


# ========== EXEMPLE D'UTILISATION ==========
if __name__ == "__main__":
    # Test M2
    print("=" * 70)
    print("Test M2 - Center-Out")
    print("=" * 70)
    
    solver = M2CenterOutSolver(n=8, symmetry_breaking=False, seed=42)
    
    # Montrer l'ordre center-out
    order = solver._compute_center_out_order()
    print(f"Ordre center-out pour N=8: {order}")
    
    result = solver.solve(time_limit=30.0)
    
    print(f"\n{result}")
    print(f"\nMétriques benchmark:")
    print(f"  - Solutions uniques en 30s: {result.num_unique_solutions()}")
    print(f"  - Temps première solution: {result.time_to_first_solution:.4f}s")
    
    if result.solutions:
        print(f"\nPremière solution: {result.solutions[0]}")
        solver.print_board(result.solutions[0])
    
    # Comparaison avec symmetry breaking
    print("\n" + "=" * 70)
    print("Test M2 - Center-Out (avec symmetry breaking)")
    print("=" * 70)
    
    solver_sym = M2CenterOutSolver(n=8, symmetry_breaking=True, seed=42)
    result_sym = solver_sym.solve(time_limit=30.0)
    
    print(f"\n{result_sym}")
    print(f"\nComparaison:")
    print(f"  Sans SB: {result.num_unique_solutions()} solutions")
    print(f"  Avec SB: {result_sym.num_unique_solutions()} solutions")
