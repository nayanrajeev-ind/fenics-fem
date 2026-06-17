from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
nkmax = 5;
# ******* Exact solutions and forcing terms for error analysis ****** #
u_str ='(sin(pi*x[0]))'

hh = []; nn = [];
######### COMPUTING THE SOLUTION ###########
for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1)
    mesh = UnitIntervalMesh(nps) #UnitSquareMesh(nps,nps)
    hh.append(mesh.hmax())
    # instantiation of exact solutions
    u_ex = Expression(u_str, domain=mesh, degree = 3)
    f_ex = -((div(grad(u_ex))))
    # *********** Finite Element spaces ************* #
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    Hh = FunctionSpace(mesh, P1)
    nn.append(Hh.dim())
    print("....... DoF = ", Hh.dim())
    u_h = Function(Hh)
    u = TrialFunction(Hh)
    v = TestFunction(Hh)
    bcU = DirichletBC(Hh, u_ex, "on_boundary")
    # ******** Weak form - linear problem ************* #
    a = inner(grad((u)),grad((v)))*dx
    l = f_ex * v * dx
    solve(a == l, u_h, bcU, solver_parameters={"linear_solver":"mumps"})
     ######### Solution Plot ################
    # Get mesh coordinates and numerical solution values
    coords = mesh.coordinates()
    u_h_vals = u_h.compute_vertex_values(mesh)

    # Sort by x-coordinate
    idx = np.argsort(coords[:, 0])
    x_mesh = coords[idx, 0]
    u_h_vals = u_h_vals[idx]

    # Create fine grid for exact solution
    x_fine = np.linspace(0, 1, 200)
    u_exact_vals = np.array([u_ex(xi) for xi in x_fine])

    # Plot
    plt.figure()
    plt.plot(x_mesh, u_h_vals, 'o-', label=r'$u_h$')
    plt.plot(x_fine, u_exact_vals, '--', label=r'$u_{ex}$')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Numerical vs Exact Solution (1D) - nk={nk}')
    plt.legend()
    plt.grid(True)
    plt.show()
