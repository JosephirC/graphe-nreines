from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
import logging

Solution = List[int]
SolutionKey = Tuple[int, ...]


@dataclass
class SolverResult:
    """
    Résultat d'une exécution de solveur

    Attributes:
        solutions: Liste des solutions trouvées (liste de permutations)
        unique_solutions: Ensemble de solutions uniques (tuples)
        execution_time: Temps d'exécution total en secondes
        time_to_first_solution: Temps pour trouver la première solution
        iterations: Nombre d'itérations effectuées
        nodes_explored: Nombre de nœuds explorés (branches CP-SAT)
        success: Au moins une solution valide trouvée
        metadata: Informations supplémentaires spécifiques au solveur
    """
    solutions: List[Solution]
    unique_solutions: Set[SolutionKey] = field(default_factory=set)
    execution_time: float = 0.0
    time_to_first_solution: Optional[float] = None
    iterations: int = 0
    nodes_explored: int = 0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def num_unique_solutions(self) -> int:
        """Retourne le nombre de solutions uniques (anti-doublons)"""
        return len(self.unique_solutions)

    def __str__(self) -> str:
        """Représentation lisible du résultat"""

        ttf = (
            f"{self.time_to_first_solution:.4f}s"
            if self.time_to_first_solution is not None
            else "None"
        )

        return (
            f"SolverResult(\n"
            f"  solutions_found={len(self.solutions)},\n"
            f"  unique_solutions={self.num_unique_solutions()},\n"
            f"  execution_time={self.execution_time:.4f}s,\n"
            f"  time_to_first={ttf},\n"
            f"  success={self.success}\n"
            f")"
        )


class BaseSolver(ABC):
    """
    Classe de base abstraite pour tous les solveurs N-Reines

    Tous les solveurs doivent implémenter solve() et les propriétés abstraites
    """

    def __init__(self, n: int, symmetry_breaking: bool = False, seed: Optional[int] = None):
        """
        Initialise le solveur

        Args:
            n: Taille de l'échiquier (nombre de reines)
            symmetry_breaking: Activer la réduction de symétrie (x[0] <= (n-1)//2)
            seed: Graine aléatoire pour reproductibilité

        Raises:
            ValueError: Si n < 1
        """
        if n < 1:
            raise ValueError(f"n doit être >= 1, reçu: {n}")

        self.n = n
        self.symmetry_breaking = symmetry_breaking
        self.seed = seed
        self.logger = logging.getLogger(self.__class__.__name__)

        # Pour tracking des solutions uniques (anti-doublons)
        self._seen_solutions: Set[SolutionKey] = set()

    @abstractmethod
    def solve(self, **kwargs) -> SolverResult:
        """
        Résout le problème des N-Reines

        Returns:
            SolverResult contenant les solutions et métriques benchmark
        """
        pass

    @property
    @abstractmethod
    def solver_type(self) -> str:
        """Retourne 'COMPLETE' ou 'INCOMPLETE'"""
        pass

    @property
    @abstractmethod
    def method_id(self) -> str:
        """Retourne l'ID de la méthode (ex: M1, M2, I1, I2)"""
        pass

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Retourne le nom descriptif de l'algorithme"""
        pass

    def solution_key(self, solution: Solution) -> SolutionKey:
        """
        Convertit une solution (liste mutable) en clé immuable et hashable.

        Utilisée pour le dédoublonnage (sets/dicts). La convention est :
        solution[i] = colonne de la reine placée sur la ligne i.

        Args:
            solution: Liste des positions des reines (solution[i] = colonne).

        Returns:
            Tuple immuable correspondant à la solution.
        """
        return tuple(solution)

    def is_unique_solution(self, solution: Solution) -> bool:
        """
        Vérifie si une solution n'a pas déjà été trouvée

        Args:
            solution: Liste des positions des reines

        Returns:
            True si la solution est nouvelle, False sinon
        """
        key = self.solution_key(solution)
        if key in self._seen_solutions:
            return False
        self._seen_solutions.add(key)
        return True

    def reset_seen_solutions(self):
        """Réinitialise le tracking des solutions uniques"""
        self._seen_solutions.clear()

    def verify_solution(self, solution: Solution) -> bool:
        """
        Vérifie qu'une solution est valide (0 conflits)

        Args:
            solution: Liste des positions des reines

        Returns:
            True si la solution est valide, False sinon
        """
        if len(solution) != self.n:
            return False

        # Vérifier que toutes les valeurs sont dans la plage
        if not all(0 <= pos < self.n for pos in solution):
            return False

        # Vérifier les colonnes (pas de doublons)
        if len(set(solution)) != self.n:
            return False

        # Vérifier les diagonales
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Même diagonale descendante ou montante
                if abs(solution[i] - solution[j]) == abs(i - j):
                    return False

        return True

    def count_conflicts(self, solution: Solution) -> int:
        """
        Compte le nombre de paires de reines en conflit

        Args:
            solution: Liste des positions des reines

        Returns:
            Nombre de conflits (0 = solution valide)
        """
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Même colonne
                if solution[i] == solution[j]:
                    conflicts += 1
                # Même diagonale
                elif abs(solution[i] - solution[j]) == abs(i - j):
                    conflicts += 1
        return conflicts

    def print_board(self, solution: Solution, show_indices: bool = False) -> None:
        """
        Affiche une solution sous forme d'échiquier

        Args:
            solution: Liste des positions des reines
            show_indices: Si True, affiche les indices de ligne/colonne
        """
        if show_indices:
            print("   " + " ".join(str(i) for i in range(self.n)))

        for i in range(self.n):
            row = []
            if show_indices:
                row.append(f"{i} ")

            for j in range(self.n):
                if j == solution[i]:
                    row.append("♛")
                else:
                    row.append("⬜" if (i + j) % 2 == 0 else "⬛")

            print(" ".join(row))

    def get_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le solveur (pour benchmarks)

        Returns:
            Dictionnaire avec les informations du solveur
        """
        return {
            "method_id": self.method_id,
            "algorithm": self.algorithm_name,
            "type": self.solver_type,
            "n": self.n,
            "symmetry_breaking": self.symmetry_breaking,
            "seed": self.seed,
            "class": self.__class__.__name__
        }

    def __repr__(self) -> str:
        """Représentation du solveur"""
        return (
            f"{self.__class__.__name__}(n={self.n}, "
            f"method={self.method_id}, "
            f"symmetry_breaking={self.symmetry_breaking})"
        )


class CompleteSolver(BaseSolver):
    """
    Classe de base pour les solveurs COMPLETS

    Les solveurs complets garantissent de trouver toutes les solutions
    ou de prouver qu'il n'en existe pas (backtracking exhaustif)
    """

    @property
    def solver_type(self) -> str:
        return "COMPLETE"

    @abstractmethod
    def solve(self, time_limit: float = 30.0) -> SolverResult:
        """
        Résout le problème de manière complète

        Args:
            time_limit: Limite de temps en secondes (30s par défaut pour benchmark)

        Returns:
            SolverResult avec toutes les solutions trouvées dans la limite de temps
        """
        pass


class IncompleteSolver(BaseSolver):
    """
    Classe de base pour les solveurs INCOMPLETS

    Les solveurs incomplets ne garantissent pas de trouver une solution,
    même si elle existe. Exploration partielle, pas de preuve.
    """

    @property
    def solver_type(self) -> str:
        return "INCOMPLETE"

    @abstractmethod
    def solve(self, time_limit: float = 30.0, **kwargs) -> SolverResult:
        """
        Résout le problème de manière incomplète

        Args:
            time_limit: Limite de temps en secondes (30s par défaut pour benchmark)
            **kwargs: Paramètres spécifiques au solveur

        Returns:
            SolverResult avec les solutions trouvées (peut être vide)
        """
        pass
