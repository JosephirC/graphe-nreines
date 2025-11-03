from ortools.sat.python import cp_model


def solve_n_queens(n: int, max_solutions=None):

    # Création du modèle
    model = cp_model.CpModel()

    # Variables : position des reines dans chaque ligne x[i] = colonne de la reine dans la ligne i
    queens = [model.NewIntVar(0, n - 1, f'x{i}') for i in range(n)]

    # Contraintes : colonnes et diagonales
    model.AddAllDifferent(queens)

    # Diagonales descendantes vers la droite ↘
    model.AddAllDifferent([queens[i] + i for i in range(n)])

    # Diagonales descendantes vers la gauche ↙
    model.AddAllDifferent([queens[i] - i for i in range(n)])

    solver = cp_model.CpSolver()

    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables, max_solutions=None):
            super().__init__()
            self.__variables = variables
            self.max_solutions = max_solutions
            self.solutions = []

        def OnSolutionCallback(self):
            self.solutions.append([self.Value(v) for v in self.__variables])
            if self.max_solutions and len(self.solutions) >= self.max_solutions:
                self.StopSearch()

    collector = SolutionCollector(queens, max_solutions)
    solver.SearchForAllSolutions(model, collector)

    return collector.solutions, solver.WallTime()


if __name__ == "__main__":

    # Taille de l'échiquier et nombre de reines
    n = 10

    # Nombre maximum de solutions à trouver
    max_solutions = None

    solutions, wall_time = solve_n_queens(n, max_solutions)

    print(f"Nombre de solutions trouvées: {len(solutions)}")
    for idx, solution in enumerate(solutions):
        print(f"Solution {idx + 1}: {solution}")
    print(f"Temps d'exécution: {wall_time} ms")