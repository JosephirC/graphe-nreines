from ortools.sat.python import cp_model
import time
import pandas as pd
import matplotlib.pyplot as plt


def solve_n_queens(n: int, strategy="DEFAULT", max_solutions=None):
    """Résout le problème des N reines avec OR-Tools et renvoie (nb_solutions, temps)."""

    model = cp_model.CpModel()
    queens = [model.NewIntVar(0, n - 1, f'x{i}') for i in range(n)]

    # Contraintes : colonnes et diagonales
    model.AddAllDifferent(queens)
    model.AddAllDifferent([queens[i] + i for i in range(n)])
    model.AddAllDifferent([queens[i] - i for i in range(n)])

    # Choix de la stratégie de recherche
    if strategy == "HEURISTIC":
        # CHOOSE_FIRST : ordre fixe | SELECT_MIN_VALUE : valeurs croissantes
        model.AddDecisionStrategy(
            queens,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE
        )

    solver = cp_model.CpSolver()

    if strategy == "HEURISTIC":
        solver.parameters.search_branching = 1  # FIXED_SEARCH
    else:
        solver.parameters.search_branching = 0  # DEFAULT_SEARCH

    # Collecteur de solutions

    class Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables, max_solutions=None):
            super().__init__()
            self.variables = variables
            self.max_solutions = max_solutions
            self.count = 0

        def OnSolutionCallback(self):
            self.count += 1
            if self.max_solutions and self.count >= self.max_solutions:
                self.StopSearch()

    collector = Collector(queens, max_solutions)

    start = time.time()
    solver.SearchForAllSolutions(model, collector)
    elapsed = time.time() - start

    return collector.count, elapsed


def benchmark(strategies=("DEFAULT", "HEURISTIC"), ns=range(4, 15)):
    """Lance le benchmark pour plusieurs tailles de N et stratégies."""
    results = []

    for strategy in strategies:
        print(f"\n--- Stratégie : {strategy} ---")
        for n in ns:
            nb_solutions, duration = solve_n_queens(n, strategy=strategy)
            print(f"N={n:2d} | Solutions={nb_solutions:8d} | Temps={duration:.3f}s")
            results.append({
                "N": n,
                "Strategy": strategy,
                "Solutions": nb_solutions,
                "Time(s)": duration
            })

    df = pd.DataFrame(results)
    df.to_csv("benchmark_nqueens.csv", index=False)
    print("\nRésultats enregistrés dans benchmark_nqueens.csv")

    # Graphique
    plt.figure(figsize=(8, 5))
    for strategy in strategies:
        subset = df[df["Strategy"] == strategy]
        plt.plot(subset["N"], subset["Time(s)"], marker="o", label=strategy)
    plt.xlabel("Taille N")
    plt.ylabel("Temps (s)")
    plt.title("Benchmark N-Queens (OR-Tools)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("benchmark_nqueens.png", dpi=150)
    plt.show()

    return df


if __name__ == "__main__":
    df = benchmark()
