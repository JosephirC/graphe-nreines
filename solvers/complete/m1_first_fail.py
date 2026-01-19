"""
M1 - Strict First-Fail (min domaine)
Méthode COMPLÈTE avec stratégie CHOOSE_MIN_DOMAIN_SIZE

Conforme au TODO:
- Type: stratégie de recherche (branching)
- Solveur: OR-Tools CP-SAT
- Complétude: OUI (backtracking exhaustif)
- search_branching = FIXED_SEARCH
- AddDecisionStrategy(x, CHOOSE_MIN_DOMAIN_SIZE, SELECT_MIN_VALUE)
- Paramètre: symmetry_breaking ON/OFF
"""

from ortools.sat.python import cp_model
from typing import Optional
import time

from ..base_solver import CompleteSolver, SolverResult


class M1StrictFirstFailSolver(CompleteSolver):
    """
    M1: Solveur avec stratégie Strict First-Fail
    
    Principe:
    - Toujours brancher sur la variable la plus contrainte (domaine minimal)
    - Détecter les échecs le plus tôt possible (principe first-fail)
    
    Avantages:
    - Élagage efficace de l'arbre de recherche
    - Détection rapide des branches infaisables
    
    Paramètres:
    - symmetry_breaking: x[0] <= (n-1)//2 (miroir gauche/droite)
    """
    
    @property
    def method_id(self) -> str:
        return "M1"
    
    @property
    def algorithm_name(self) -> str:
        return "Strict First-Fail (min domain)"
    
    def solve(self, time_limit: float = 30.0) -> SolverResult:
        """
        Résout avec stratégie First-Fail
        
        Args:
            time_limit: Limite de temps en secondes (30s benchmark)
            
        Returns:
            SolverResult avec métriques benchmark
        """
        self.logger.info(
            f"[M1] Démarrage First-Fail pour N={self.n}, "
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
            # Contrainte: x[0] <= (n-1)//2 (miroir gauche/droite)
            model.Add(queens[0] <= (self.n - 1) // 2)
            self.logger.debug(f"Symmetry breaking activé: x[0] <= {(self.n - 1) // 2}")
        
        # ========== STRATÉGIE DE RECHERCHE M1 ==========
        # First-Fail: brancher sur la variable avec le plus petit domaine
        model.AddDecisionStrategy(
            queens,
            cp_model.CHOOSE_MIN_DOMAIN_SIZE,  # Variable la plus contrainte
            cp_model.SELECT_MIN_VALUE         # Valeur minimale
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
        collector = _M1SolutionCollector(
            queens,
            self,
            start_time,
            time_limit
        )
        
        # ========== RECHERCHE ==========
        self.logger.info("[M1] Lancement de la recherche exhaustive...")
        status = solver.SearchForAllSolutions(model, collector)
        
        execution_time = time.time() - start_time
        
        # ========== STATISTIQUES ==========
        num_branches = solver.NumBranches()
        num_conflicts = solver.NumConflicts()
        
        self.logger.info(
            f"[M1] Terminé: {collector.num_unique} solutions uniques, "
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
                "method_id": "M1",
                "status": self._status_to_string(status),
                "num_branches": num_branches,
                "num_conflicts": num_conflicts,
                "symmetry_breaking": self.symmetry_breaking,
                "search_branching": "FIXED_SEARCH",
                "strategy": "CHOOSE_MIN_DOMAIN_SIZE + SELECT_MIN_VALUE"
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


class _M1SolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    Collecteur de solutions pour M1
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
                    f"[M1] Première solution trouvée en {self.time_to_first:.4f}s"
                )


# ========== EXEMPLE D'UTILISATION ==========
if __name__ == "__main__":
    # Test M1 sans symmetry breaking
    print("=" * 70)
    print("Test M1 - Strict First-Fail (sans symmetry breaking)")
    print("=" * 70)
    
    solver = M1StrictFirstFailSolver(n=8, symmetry_breaking=False, seed=42)
    result = solver.solve(time_limit=30.0)
    
    print(f"\n{result}")
    print(f"\nMétriques benchmark:")
    print(f"  - Solutions uniques en 30s: {result.num_unique_solutions()}")
    print(f"  - Temps première solution: {result.time_to_first_solution:.4f}s")
    
    if result.solutions:
        print(f"\nPremière solution: {result.solutions[0]}")
        solver.print_board(result.solutions[0])
    
    # Test M1 avec symmetry breaking
    print("\n" + "=" * 70)
    print("Test M1 - Strict First-Fail (avec symmetry breaking)")
    print("=" * 70)
    
    solver_sym = M1StrictFirstFailSolver(n=8, symmetry_breaking=True, seed=42)
    result_sym = solver_sym.solve(time_limit=30.0)
    
    print(f"\n{result_sym}")
    print(f"\nMétriques benchmark:")
    print(f"  - Solutions uniques en 30s: {result_sym.num_unique_solutions()}")
    print(f"  - Temps première solution: {result_sym.time_to_first_solution:.4f}s")
    
    print("\n" + "=" * 70)
    print("Comparaison symmetry breaking:")
    print(f"  Sans SB: {result.num_unique_solutions()} solutions")
    print(f"  Avec SB: {result_sym.num_unique_solutions()} solutions")
    print(f"  Ratio: {result.num_unique_solutions() / max(result_sym.num_unique_solutions(), 1):.1f}x")
