# FEniCS FEM Solver for Biharmonic Equation

This repository contains a Python implementation using the **FEniCS** computing platform to solve a fourth-order biharmonic equation via a mixed finite element formulation.

## Mathematical Formulation
We consider the biharmonic problem with simply supported boundary conditions:
$$\Delta^2 u = f \quad \text{in } \Omega$$,
$$u = 0$$ and $$\Delta u = 0 \text{on } \partial\Omega$$

## Numerical Approach
To avoid the need for complex $C^1$ continuous elements, the fourth-order problem is decoupled into a system of two second-order elliptic equations using the **Ciarlet-Raviart mixed method**:
$$-\Delta u = v \ \text{in } \Omega$$ and 
$$-\Delta v = f \ \text{in } \Omega$$

This formulation allows the system to be computed efficiently using standard continuous Lagrange finite elements ($P_k$ spaces).
