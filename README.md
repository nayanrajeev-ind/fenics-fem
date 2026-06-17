## Repository Directory

### 1. 1D Poisson Equation with Dirichlet BCs (P1 Elements)
* **File:** 1DPoissonDirichlet.py
* **Formulation:** Solves the 1D boundary value problem under non-homogeneous Dirichlet boundary conditions across the entire boundary.
* **Element Type:** Linear Lagrange elements (P1 continuous Galerkin).
* **Visualization:** Using Matplotlib to map the discrete numerical solution against the exact analytical solution.

### 2. 1D Poisson Equation with Mixed BCs (P2 Elements)
* **File:** 1dNeumann.py
* **Formulation:** Solves a 1D problem on a unit interval with a Dirichlet condition on the left boundary and a Neumann flux condition on the right.
* **Element Type:** Quadratic Lagrange elements (P2 continuous Galerkin).
* **Verification:** Computes and logs L2 and H1 error norms across consecutive mesh refinements to verify optimal convergence orders: O(h^3) in L2 and O(h^2) in H1.

### 3. 2D Poisson Equation with Mixed Dirichlet-Neumann Boundaries
* **File:** 2DNeumann.py
* **Formulation:** Scales the elliptic system to a two-dimensional spatial domain. It applies non-homogeneous Dirichlet conditions on the left/right boundaries while handling boundary integrals for Neumann flux conditions along the top and bottom edges.
* **Element Type:** 2D Linear Lagrange elements (P1) on a triangular mesh topology.
* **Verification:** Validates multi-dimensional gradient assembly via rigorous L2 and H1 error tracking over consecutive refinement cycles.

### 4. 2D Biharmonic Equation via Ciarlet-Raviart Mixed Method
* **File:** biharmonic.py
* **Formulation:** Solves a complex fourth-order fluid/structural PDE. 
* **Numerical Method:** Implements the Ciarlet-Raviart mixed finite element approach, decoupling the fourth-order problem into a coupled system of two second-order elliptic equations to bypass the need for complex C1 continuous elements.
