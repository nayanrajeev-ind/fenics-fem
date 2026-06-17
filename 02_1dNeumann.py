from fenics import *
import numpy as np
import matplotlib.pyplot as plt
nkmax = 5;
# ******* Exact solutions and forcing terms for error analysis ****** #
u_str ='(sin(x[0]))'

hh = []; nn = []; du = []
e0u = []; r0u = [];
e1u = []; r1u = [];
r0u.append(0.0); r1u.append(0.0)
######### COMPUTING THE SOLUTION ###########
for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1)
    #mesh = UnitIntervalMesh(nps,nps)
    mesh = UnitIntervalMesh(nps)
    hh.append(mesh.hmax())
    n = FacetNormal(mesh)
    # instantiation of exact solutions
    u_ex = Expression(u_str, domain=mesh, degree = 6)
    f_ex = -((div(grad(u_ex))))
    # *********** Finite Element spaces ************* #
    P1 = FiniteElement("CG", mesh.ufl_cell(), 2)
    Hh = FunctionSpace(mesh, P1)
    nn.append(Hh.dim())
    # Define a Dirichlet BC at one point (e.g., x=0)
    def boundary_left(x, on_boundary):
        return on_boundary and near(x[0], 0.0)

    bc = DirichletBC(Hh, u_ex, boundary_left)
    print("....... DoF = ", Hh.dim())
    u_h = Function(Hh)
    u = TrialFunction(Hh)
    v = TestFunction(Hh)
   # bcU = DirichletBC(Hh, u_ex, "on_boundary")
    # ******** Weak form - linear problem ************* #
    a = inner(grad((u)),grad((v)))*dx
    l = f_ex * v * dx+ dot(grad(u_ex), n)*v*ds  
    solve(a == l, u_h, bc, solver_parameters={"linear_solver":"mumps"})
     ######### Solution Plot ################
    # Get mesh coordinates and numerical solution values
    L2_error = assemble((u_ex-u_h)**2*dx)
    H1_error = assemble((grad(u_ex-u_h))**2*dx)

    e0u.append(pow(L2_error,0.5))
    e1u.append(pow(H1_error,0.5))
#save data
    if(nk>0):
        r0u.append(ln(e0u[nk]/e0u[nk-1])/ln(hh[nk]/hh[nk-1]))
        r1u.append(ln(e1u[nk]/e1u[nk-1])/ln(hh[nk]/hh[nk-1]))
# ********  Generating error history **** #
#print ('{:5} & {:^8} & {:^5} & {:^8} & {:^5} & {:^8}'.format('nn','hh','e0(u)','r0(u)','e2(u)','r2(u)'))
print ('===============================================================')
for nk in range(nkmax):
    print ('{:5d} & {:0.4f} & {:1.2e} & {:.3f} & {:1.2e} & {:.3f}'.format(nn[nk], hh[nk], e0u[nk], r0u[nk], e1u[nk], r1u[nk]))
print ('===============================================================') 
