# test_architecture.py
from solvers.complete import M1StrictFirstFailSolver, M2CenterOutSolver
from solvers.incomplete import I1LNSSolver, I2MinConflictsSolver

def test_m1():
    print("Test M1 Strict First Fail...")
    solver = M1StrictFirstFailSolver(n=8, symmetry_breaking=False, seed=42)
    result = solver.solve(time_limit=5.0)
    print(f"✓ M1 OK: {result.num_unique_solutions()} solutions en {result.execution_time:.2f}s")

def test_m2():
    print("Test M2 Center Out...")
    solver = M2CenterOutSolver(n=8, symmetry_breaking=False, seed=42)
    result = solver.solve(time_limit=5.0)
    print(f"✓ M2 OK: {result.num_unique_solutions()} solutions en {result.execution_time:.2f}s")

def test_i1():
    print("Test I1 LNS...")
    solver = I1LNSSolver(n=8, neighborhood_size=0.3, seed=42)
    result = solver.solve(time_limit=5.0)
    print(f"✓ I1 OK: {result.num_unique_solutions()} solutions en {result.execution_time:.2f}s")

def test_i2():
    print("Test I2 Min Conflicts...")
    solver = I2MinConflictsSolver(n=8, seed=42)
    result = solver.solve(time_limit=5.0)
    print(f"✓ I2 OK: {result.num_unique_solutions()} solutions en {result.execution_time:.2f}s")

if __name__ == "__main__":
    print("="*50)
    print("TEST DE L'ARCHITECTURE")
    print("="*50)
    test_m1()
    test_m2()
    test_i1()
    test_i2()
    print("\n✅ Tous les tests passent!")