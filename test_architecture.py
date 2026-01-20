# test_architecture.py
from solvers.incomplete.cp_sat_min_conflicts import CPSatMinConflictsSolver
from solvers.incomplete.cp_sat_lns import CPSatLNSSolver
from solvers.complete.cp_sat_fixed_search_center_out import CPSatCenterOutSolver
from solvers.complete.cp_sat_fixed_search_first_fail import CPSatFirstFailSolver
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

# Imports directs (évite les soucis d'exports dans __init__.py)


def test_m1():
    print("Test M1 Strict First-Fail.")
    solver = CPSatFirstFailSolver(n=8, symmetry_breaking=False, seed=42)
    result = solver.solve(time_limit=5.0)
    print(
        f"✓ M1 OK: {result.num_unique_solutions()} solutions en {result.execution_time:.2f}s")


def test_m2():
    print("Test M2 Center-Out.")
    solver = CPSatCenterOutSolver(n=8, symmetry_breaking=False, seed=42)
    result = solver.solve(time_limit=5.0)
    print(
        f"✓ M2 OK: {result.num_unique_solutions()} solutions en {result.execution_time:.2f}s")


def test_i1():
    print("Test I1 LNS.")
    solver = CPSatLNSSolver(n=8, neighborhood_size=0.3, seed=42)
    result = solver.solve(time_limit=5.0)
    print(
        f"✓ I1 OK: {result.num_unique_solutions()} solutions en {result.execution_time:.2f}s")


def test_i2():
    print("Test I2 Min-Conflicts.")
    solver = CPSatMinConflictsSolver(n=8, seed=42)
    result = solver.solve(time_limit=5.0)
    print(
        f"✓ I2 OK: {result.num_unique_solutions()} solutions en {result.execution_time:.2f}s")


if __name__ == "__main__":
    print("=" * 50)
    print("TEST DE L'ARCHITECTURE")
    print("=" * 50)
    test_m1()
    test_m2()
    test_i1()
    test_i2()
    print("\n✅ Tous les tests passent!")
