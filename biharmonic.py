from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Total refinement levels
nkmax = 5


u_str = 'sin(pi*x[0])*sin(pi*x[1])'  # Exact u in H_0^2(Omega)
w_str = '2*pow(pi, 2)*sin(pi*x[0])*sin(pi*x[1])' # Auxiliary variable  w = -Delta(u)
f_str = '4*pow(pi, 4)*sin(pi*x[0])*sin(pi*x[1])' # Source term f = Delta^2(u)

# Initialize arrays to store experimental data for the Convergence Table
hh = []; nn = [];   # Mesh size (h) and Degrees of Freedom (DoF)
e0u = []; r0u = []; # L2-norm error and its corresponding convergence rate
e1u = []; r1u = []; # H1-seminorm error and its corresponding convergence rate
r0u.append(0.0); r1u.append(0.0) # Rates require two points so need initial as 0

# Discretize a 1D cross-section at x=0.5 to observe pointwise convergence
points_x = np.linspace(0, 1, 100) 
slice_pts = [Point(0.5, y) for y in points_x] # Probe locations for the peak of the sine wave

# Persistent storage for discrete solutions u_h across refinement levels
u_h_list = []

######### COMPUTING THE SOLUTIONS ###########
for nk in range(nkmax):
    print("....... Processing Refinement level : nk = ", nk)
    
    # Calculate grid resolution: nps = 2, 4, 8, 16, 32...
    nps = pow(2, nk+1)
    # Generate a structured triangulation of the domain Omega = [0,1]x[0,1]; 
    mesh = UnitSquareMesh(nps, nps) #FEniCS automatically splits every square into 2 triangles to create a simplex mesh.
    # Track the diameter of the largest element (h) for the O(h^p) calculation
    hh.append(mesh.hmax())
    
    # Map analytical strings u_str to high-order (degree 6) quadrature spaces
    # This minimizes interpolation error to ensure the benchmark is "exact"
    u_ex = Expression(u_str, domain=mesh, degree=1)
    w_ex = Expression(w_str, domain=mesh, degree=1)
    f_ex = Expression(f_str, domain=mesh, degree=1)

# *********** Mixed Finite Element spaces ************* #
    # Define local P1 (Linear Lagrange) elements for both variables
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    # Construct the Product Space Sigma_h = P1 x P1 (Ciarlet-Raviart mixed method)
    element = MixedElement([P1, P1])
    # Instantiate the global Trial Space W_h subset of H1 x H1
    W = FunctionSpace(mesh, element)
    # Record the dimension of the global system (Total Degrees of Freedom)
    nn.append(W.dim())
    
    # Define Trial functions (u, w) and Test functions (v, phi) for the weak form
    (u, w) = TrialFunctions(W)
    (v, phi) = TestFunctions(W)
    
    # Apply Dirichlet Boundary Conditions: u=0 and w=0 on the boundary Gamma
    # This enforces the "Simply Supported" physical condition
    #W.sub(0) refers to the first subspace (the trial space for displacement u).
    #W.sub(1) refers to the second subspace (the trial space for the auxiliary variable w).
    bcs = [DirichletBC(W.sub(0), Constant(0.0), "on_boundary"),
           DirichletBC(W.sub(1), Constant(0.0), "on_boundary")] 

    # The Mixed Bilinear Form a((u,w), (v,phi)):
    a = (inner(grad(u), grad(phi)) - w*phi + inner(grad(w), grad(v))) * dx
    # The Linear Functional L(v):
    l = f_ex * v * dx
    
    # wh represents the combined solution vector in the mixed space
    #wh is the actual coefficient vector U=[Ui] \in R^N wrt basis of W_h that the computer will store.
    wh = Function(W)
    # Solve the linear system using the MUMPS multifrontal direct solver
    solve(a == l, wh, bcs, solver_parameters={"linear_solver":"mumps"}) #whether a=l is true not then run further
    
    # Extract displacement u_h from the mixed solution for error analysis
    u_h_curr, w_h_curr = wh.split(True)
    u_h_list.append(u_h_curr)

    # Error Analysis: Compute ||u_ex - u_h|| in the L2 norm (Omega)
    e0u.append(pow(assemble((u_ex - u_h_curr)**2 * dx), 0.5))
    # Error Analysis: Compute |u_ex - u_h| in the H1 semi-norm (Energy norm)
    e1u.append(pow(assemble((grad(u_ex - u_h_curr))**2 * dx), 0.5))
    
    # Compute the Experimental Order of Convergence (EOC): log(E_i/E_{i-1}) / log(h_i/h_{i-1})
    if(nk > 0):
        r0u.append(np.log(e0u[nk]/e0u[nk-1]) / np.log(hh[nk]/hh[nk-1]))
        r1u.append(np.log(e1u[nk]/e1u[nk-1]) / np.log(hh[nk]/hh[nk-1]))

    ######### PHASE 1: SEQUENTIAL MESH PLOTTING ################
    # Visualizing the geometric discretization process
    plt.figure()
    plot(mesh, linewidth=0.5)
    plt.title(f"MESH: Refinement level {nk}\n(Close window to see next level)")
    plt.show() 

print("\n--- All meshes shown. Starting Visual Convergence Phase ---")

######### PHASE 2: SEQUENTIAL VISUAL CONVERGENCE ################
# Demonstrating that u_h -> u_ex as h -> 0
plt.figure(figsize=(8, 6))
# Sample the high-fidelity analytical solution at slice points
exact_vals = [Expression(u_str, degree=6)(p) for p in slice_pts]
plt.plot(points_x, exact_vals, 'k-', label='Exact Analytical', linewidth=2.5)

for nk in range(nkmax):
    # Interpolate the numerical solution at the slice points for comparison
    u_vals = [u_h_list[nk](p) for p in slice_pts]
    plt.plot(points_x, u_vals, '--', label=f'Numerical nk={nk}')
    plt.draw()
    plt.pause(1.0) 

plt.show()

######### PHASE 3: FINAL 3D PLOTS ################
# Qualitative comparison of the solution fields over the 2D domain
plt.figure()
# Project u_ex onto a visualization space for the 3D surface plot
u_ex_final = interpolate(Expression(u_str, degree=6), FunctionSpace(mesh, "CG", 1))
p1 = plot(u_ex_final)
plt.title("PHASE 3: Exact Solution (u)")

plt.figure()
# Plot the final, most refined numerical displacement field
p2 = plot(u_h_list[-1])
plt.title(f"PHASE 3: Final Discrete Solution (u_h) at nk={nkmax-1}")
plt.show()

# ******** Numerical History Table **** #
# This table proves the theoretical O(h^2) L2-convergence of the P1-Mixed Method
print ('===============================================================')
print ('{:5} & {:^8} & {:^10} & {:^5} & {:^10} & {:^5}'.format('nn','hh','e0(u)','r0(u)','e1(u)','r1(u)'))
print ('===============================================================')
for nk in range(nkmax):
    print ('{:5d} & {:0.4f} & {:1.2e} & {:.3f} & {:1.2e} & {:.3f}'.format(nn[nk], hh[nk], e0u[nk], r0u[nk], e1u[nk], r1u[nk]))
print ('===============================================================')