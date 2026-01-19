"""
I2 - CP-SAT + minimisation des conflits (contraintes molles)
Méthode INCOMPLÈTE par réparation / min-conflicts via optimisation

Conforme au TODO:
- Type: recherche par réparation / min-conflicts via optimisation (incomplète)
- Solveur: OR-Tools CP-SAT
- Incomplétude: OUI (optimisation interrompue, pas de preuve)
- Principe: autoriser solutions invalides, minimiser sum(conflicts)
- Objectif: sum(conflicts) == 0 → solution valide
- Timeout: 30s → méthode incomplète
"""

from ortools.sat.python import cp_model
from typing import List, Optional, Set
import time

from ..base_solver import IncompleteSolver, SolverResult


class I2MinConflictsSolver(IncompleteSolver):
    """
    I2: Solveur Min-Conflicts avec contraintes molles
    
    Principe:
    1. Autoriser des solutions invalides
    2. Introduire des booléens de violation (conflits):
       - Conflits colonnes: queens[i] == queens[j]
       - Conflits diagonales ↘: queens[i] + i == queens[j] + j
       - Conflits diagonales ↙: queens[i] - i == queens[j] - j
    3. Objectif: Minimize(sum(all_conflicts))
    4. Solution valide: objective == 0
    5. Couper au timeout (30s) → incomplète
    
    Paramètres exposés (TODO):
    - conflict_types: types de conflits pénalisés
    - conflict_weights: pondération des conflits
    - time_limit: temps max (30s benchmark)
    - symmetry_breaking: ON/OFF (optionnel)
    
    Sorties benchmark (TODO):
    - temps jusqu'à atteindre objectif = 0 (première solution valide)
    - nombre de solutions valides distinctes atteintes en 30s
    """
    
    def __init__(
        self,
        n: int,
        conflict_types: List[str] = ['columns', 'diagonals'],
        conflict_weights: Optional[dict] = None,
        symmetry_breaking: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialise le solveur I2 Min-Conflicts
        
        Args:
            n: Taille de l'échiquier
            conflict_types: Types de conflits à pénaliser
                - 'columns': conflits de colonnes
                - 'diagonals': conflits de diagonales (↘ et ↙)
            conflict_weights: Pondération des conflits (None = poids égaux)
            symmetry_breaking: Activer symmetry breaking
            seed: Graine pour reproductibilité
        """
        super().__init__(n, symmetry_breaking, seed)
        
        self.conflict_types = conflict_types
        self.conflict_weights = conflict_weights or {
            'columns': 1,
            'diag_desc': 1,  # Diagonales descendantes ↘
            'diag_asc': 1    # Diagonales ascendantes ↙
        }
        
    @property
    def method_id(self) -> str:
        return "I2"
    
    @property
    def algorithm_name(self) -> str:
        return "CP-SAT + Min-Conflicts (soft constraints)"
    
    def solve(self, time_limit: float = 30.0) -> SolverResult:
        """
        Résout avec Min-Conflicts (contraintes molles)
        
        Args:
            time_limit: Limite de temps globale (30s benchmark)
            
        Returns:
            SolverResult avec métriques benchmark
        """
        self.logger.info(
            f"[I2] Démarrage Min-Conflicts pour N={self.n}, "
            f"conflict_types={self.conflict_types}, "
            f"time_limit={time_limit}s"
        )
        
        start_time = time.time()
        self.reset_seen_solutions()
        
        # Tracking des solutions valides uniques
        unique_solutions: Set[str] = set()
        all_solutions: List[List[int]] = []
        time_to_first: Optional[float] = None
        
        # ========== CRÉATION DU MODÈLE CP-SAT ==========
        model = cp_model.CpModel()
        
        # Variables: queens[i] = colonne de la reine dans la ligne i
        queens = [model.NewIntVar(0, self.n - 1, f'q{i}') for i in range(self.n)]
        
        # ========== CONTRAINTES MOLLES (CONFLICTS) ==========
        conflict_vars = []
        
        # 1. Conflits de COLONNES
        if 'columns' in self.conflict_types:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    # Créer un booléen: conflict[i,j] = (queens[i] == queens[j])
                    conflict = model.NewBoolVar(f'col_conflict_{i}_{j}')
                    model.Add(queens[i] == queens[j]).OnlyEnforceIf(conflict)
                    model.Add(queens[i] != queens[j]).OnlyEnforceIf(conflict.Not())
                    
                    # Ajouter avec poids
                    weight = self.conflict_weights.get('columns', 1)
                    conflict_vars.append((conflict, weight))
        
        # 2. Conflits de DIAGONALES DESCENDANTES ↘ (row + col)
        if 'diagonals' in self.conflict_types:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    # Conflit si: queens[i] + i == queens[j] + j
                    conflict_desc = model.NewBoolVar(f'diag_desc_conflict_{i}_{j}')
                    model.Add(queens[i] + i == queens[j] + j).OnlyEnforceIf(conflict_desc)
                    model.Add(queens[i] + i != queens[j] + j).OnlyEnforceIf(conflict_desc.Not())
                    
                    weight = self.conflict_weights.get('diag_desc', 1)
                    conflict_vars.append((conflict_desc, weight))
                    
                    # Conflit si: queens[i] - i == queens[j] - j (diagonales ↙)
                    conflict_asc = model.NewBoolVar(f'diag_asc_conflict_{i}_{j}')
                    model.Add(queens[i] - i == queens[j] - j).OnlyEnforceIf(conflict_asc)
                    model.Add(queens[i] - i != queens[j] - j).OnlyEnforceIf(conflict_asc.Not())
                    
                    weight = self.conflict_weights.get('diag_asc', 1)
                    conflict_vars.append((conflict_asc, weight))
        
        # ========== OBJECTIF: MINIMISER LES CONFLITS ==========
        # Somme pondérée des conflits
        if conflict_vars:
            weighted_conflicts = []
            for conflict_var, weight in conflict_vars:
                if weight == 1:
                    weighted_conflicts.append(conflict_var)
                else:
                    # Multiplier par le poids
                    weighted = model.NewIntVar(0, weight, f'weighted_{conflict_var.Name()}')
                    model.Add(weighted == conflict_var * weight)
                    weighted_conflicts.append(weighted)
            
            # Objectif: minimiser la somme
            total_conflicts = model.NewIntVar(0, sum(w for _, w in conflict_vars), 'total_conflicts')
            model.Add(total_conflicts == sum(weighted_conflicts))
            model.Minimize(total_conflicts)
        
        # ========== SYMMETRY BREAKING (optionnel) ==========
        if self.symmetry_breaking:
            model.Add(queens[0] <= (self.n - 1) // 2)
            self.logger.debug(f"Symmetry breaking activé: x[0] <= {(self.n - 1) // 2}")
        
        # ========== CONFIGURATION SOLVEUR ==========
        solver = cp_model.CpSolver()
        
        # Limite de temps
        solver.parameters.max_time_in_seconds = time_limit
        
        # num_workers = 1 pour reproductibilité (TODO)
        solver.parameters.num_search_workers = 1
        
        # Seed
        if self.seed is not None:
            solver.parameters.random_seed = self.seed
        
        # Log search progress pour capturer solutions intermédiaires
        solver.parameters.log_search_progress = False
        
        # ========== CALLBACK POUR CAPTURER SOLUTIONS INTERMÉDIAIRES ==========
        callback = _I2SolutionCallback(
            queens,
            self,
            start_time,
            unique_solutions,
            all_solutions
        )
        
        # ========== RÉSOLUTION ==========
        self.logger.info("[I2] Lancement de l'optimisation...")
        
        # Utiliser SolveWithSolutionCallback pour capturer les solutions intermédiaires
        status = solver.Solve(model, callback)
        
        execution_time = time.time() - start_time
        
        # ========== STATISTIQUES ==========
        num_branches = solver.NumBranches()
        num_conflicts_solver = solver.NumConflicts()
        objective_value = solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None
        
        # Récupérer le temps de la première solution valide
        time_to_first = callback.time_to_first
        
        self.logger.info(
            f"[I2] Terminé: {len(unique_solutions)} solutions valides uniques, "
            f"objectif final={objective_value}, "
            f"{execution_time:.4f}s"
        )
        
        return SolverResult(
            solutions=all_solutions,
            unique_solutions=unique_solutions,
            execution_time=execution_time,
            time_to_first_solution=time_to_first,
            iterations=len(all_solutions),
            nodes_explored=num_branches,
            success=len(unique_solutions) > 0,
            metadata={
                "method_id": "I2",
                "status": self._status_to_string(status),
                "num_branches": num_branches,
                "num_conflicts_solver": num_conflicts_solver,
                "objective_value": objective_value,
                "conflict_types": self.conflict_types,
                "conflict_weights": self.conflict_weights,
                "symmetry_breaking": self.symmetry_breaking
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


class _I2SolutionCallback(cp_model.CpSolverSolutionCallback):
    """
    Callback pour capturer les solutions intermédiaires pendant l'optimisation
    """
    
    def __init__(self, variables, solver_instance, start_time, unique_solutions, all_solutions):
        super().__init__()
        self._variables = variables
        self._solver = solver_instance
        self._start_time = start_time
        self._unique_solutions = unique_solutions
        self._all_solutions = all_solutions
        self.time_to_first = None
        
    def OnSolutionCallback(self):
        """Appelé à chaque solution trouvée (même avec conflits)"""
        solution = [self.Value(v) for v in self._variables]
        
        # Vérifier si c'est une solution VALIDE (0 conflits)
        num_conflicts = self._solver.count_conflicts(solution)
        
        if num_conflicts == 0:
            # Solution valide!
            solution_hash = self._solver.hash_solution(solution)
            
            if solution_hash not in self._unique_solutions:
                self._unique_solutions.add(solution_hash)
                self._all_solutions.append(solution)
                
                # Première solution valide?
                if self.time_to_first is None:
                    self.time_to_first = time.time() - self._start_time
                    self._solver.logger.info(
                        f"[I2] Première solution VALIDE trouvée en {self.time_to_first:.4f}s "
                        f"(objectif={self.ObjectiveValue()})"
                    )
                else:
                    self._solver.logger.debug(
                        f"[I2] Nouvelle solution valide unique! Total: {len(self._unique_solutions)}"
                    )


# ========== EXEMPLE D'UTILISATION ==========
if __name__ == "__main__":
    print("=" * 70)
    print("Test I2 - CP-SAT + Min-Conflicts (contraintes molles)")
    print("=" * 70)
    
    # Test 1: Tous types de conflits
    print("\n1. Test avec tous les types de conflits")
    print("-" * 70)
    
    solver1 = I2MinConflictsSolver(
        n=8,
        conflict_types=['columns', 'diagonals'],
        conflict_weights={'columns': 1, 'diag_desc': 1, 'diag_asc': 1},
        symmetry_breaking=False,
        seed=42
    )
    
    result1 = solver1.solve(time_limit=30.0)
    
    print(f"\n{result1}")
    print(f"\nMétriques benchmark:")
    print(f"  - Solutions valides uniques en 30s: {result1.num_unique_solutions()}")
    print(f"  - Temps première solution valide: {result1.time_to_first_solution:.4f}s" if result1.time_to_first_solution else "  - Aucune solution valide trouvée")
    print(f"  - Objectif final: {result1.metadata.get('objective_value')}")
    
    if result1.solutions:
        print(f"\nPremière solution valide: {result1.solutions[0]}")
        solver1.print_board(result1.solutions[0])
    
    # Test 2: Seulement diagonales (colonnes automatiquement OK avec AllDifferent implicite)
    print("\n\n2. Test avec seulement conflits de diagonales")
    print("-" * 70)
    
    solver2 = I2MinConflictsSolver(
        n=8,
        conflict_types=['diagonals'],
        conflict_weights={'diag_desc': 2, 'diag_asc': 2},  # Poids plus élevé
        symmetry_breaking=False,
        seed=42
    )
    
    result2 = solver2.solve(time_limit=30.0)
    
    print(f"\n{result2}")
    print(f"\nMétriques benchmark:")
    print(f"  - Solutions valides uniques en 30s: {result2.num_unique_solutions()}")
    print(f"  - Temps première solution valide: {result2.time_to_first_solution:.4f}s" if result2.time_to_first_solution else "  - Aucune solution valide trouvée")
    
    # Test 3: Avec symmetry breaking
    print("\n\n3. Test avec Symmetry Breaking")
    print("-" * 70)
    
    solver3 = I2MinConflictsSolver(
        n=8,
        conflict_types=['columns', 'diagonals'],
        symmetry_breaking=True,
        seed=42
    )
    
    result3 = solver3.solve(time_limit=30.0)
    
    print(f"\nMétriques benchmark:")
    print(f"  - Solutions valides uniques en 30s: {result3.num_unique_solutions()}")
    print(f"  - Temps première solution valide: {result3.time_to_first_solution:.4f}s" if result3.time_to_first_solution else "  - Aucune solution valide trouvée")
    
    print("\n" + "=" * 70)
    print("Comparaison:")
    print(f"  Sans SB: {result1.num_unique_solutions()} solutions")
    print(f"  Avec SB: {result3.num_unique_solutions()} solutions")
