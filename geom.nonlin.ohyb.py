import os
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

#-----[ zatizeni ]-----

F_load = 1000  # [N]

#-----[ materialove konstanty]-----

E = 210000  # [MPa]
c = 50  # [mm]
nu = 0.3  

#-----[ dopocitane veliciny ]-----
mu = E / (2 * (1 + nu))
lambda_ = E * nu / ((1 + nu) * (1 - nu))

print('nacitam sit')
mesh = Mesh("prut.xml")
boundaries = MeshFunction("size_t", mesh, "prut_facet_region.xml")

#-----[definice indexovani oblasti a hranic]-----

ds = Measure("ds", domain=mesh, subdomain_data=boundaries, metadata={"quadrature_degree": 2})
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 2})

#-----[ prostory funkci ]-----
print('Generuji prostory funkci')
V = VectorFunctionSpace(mesh, "CG", 2)
u_sol = Function(V)  
w = TestFunction(V)  
du = TrialFunction(V)  

#-----[ okrajove podminky ]-----
print('Sestavuji okrajove podminky')
eps = 1

class PointLoad(UserExpression):
    def eval(self, value, x):
        if near(x[0], 0.0, eps) and near(x[1], 0.0, eps):
            value[0] = 0.0
            value[1] = -F_load
        else:
            value[0] = 0.0
            value[1] = 0.0
    def value_shape(self):
        return (2,)

class Tractions(UserExpression):
    def eval(self, value, x):
        value[0] = 0.0
        value[1] = -F_load
    def value_shape(self):
        return (2,)

trs = Tractions(element=V.ufl_element())
pl = PointLoad(element=V.ufl_element())
null = Constant((0.0, 0.0))

bcs = [
    DirichletBC(V, null, boundaries, 2), 
]

#-----[ Nelinearni hyperelasticky model ]-----
def F(displ):
    "deformacni gradient"
    return Identity(2) + grad(displ)

def C(displ):
    "pravy Cauchy-Greenuv tenzor"
    return F(displ).T * F(displ)

def E(displ):
    "green-lagrangeuv deformacni tenzor"
    return 0.5 * (C(displ) - Identity(2))

def J_det(displ):
    "jakobian deformace"
    return det(F(displ)) + DOLFIN_EPS  

def S(displ):
    "druhy piola-kirchhoffuv napetovy tenzor (Saint Venant-Kirchhoff)"
    E_ = E(displ)  # Green-Lagrangeuv tenzor
    return lambda_ * tr(E_) * Identity(2) + 2 * mu * E_

def P(displ):
    "prvni piola-kirchhoffův napetovy tenzor"
    return F(displ) * S(displ)

def sigma(displ):
    "cauchyuv napetovy tenzor"
    F_ = F(displ)
    J_ = J_det(displ)
    return (1/J_) * F_ * S(displ) * F_.T

#-----[ nelinearni slaba formulace ]-----
F_int = c * inner(P(u_sol), grad(w)) * dx
F_ext = dot(pl, w) * ds(4)
F_res = F_int - F_ext

# jakobian slabe formulace
J = derivative(F_res, u_sol, du)

#-----[ newtonova metoda ]-----
print('resim nelinearni problem')
problem = NonlinearVariationalProblem(F_res, u_sol, bcs, J)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1e-5 
prm["newton_solver"]["relative_tolerance"] = 1e-5
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["report"] = True
prm["newton_solver"]["error_on_nonconvergence"] = True
prm["newton_solver"]["linear_solver"] = "mumps"

solver.solve()

#-----[ Export výsledků ]-----

V_scalar = FunctionSpace(mesh, 'CG', 1)
u_y = project(u_sol.sub(1), V_scalar)
file_u_y = File("uy_prut3d_nonlinear.pvd")
file_u_y << u_y

uy_vals = u_y.compute_vertex_values(mesh)
coords = mesh.coordinates()
x_coords = coords[:, 0]
sorted_indices = np.argsort(x_coords)
x_sorted = x_coords[sorted_indices]
uy_sorted = uy_vals[sorted_indices]

plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.9,
    "lines.linewidth": 2.2,
})

fig, ax = plt.subplots(figsize=(6, 6))
plt.plot(x_sorted, -uy_sorted, label='w(x) [mm]')
plt.xlabel("Souřadnice $x$ [mm]").set_fontsize(16)
plt.ylabel("Průhyb $w(x)$ [mm]").set_fontsize(16)
plt.grid(True)
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
ax.set_ylim(11,-1)
plt.savefig('pruhyb_nonlinear_fenics1.png')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
F_plot = FunctionSpace(mesh, "CG", 1)
T_plot = TensorFunctionSpace(mesh, "CG", 1)

plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.9,
    "lines.linewidth": 2.2,
})

p = project(sigma(u_sol), T_plot)
psxx = project(sigma(u_sol)[0, 0], F_plot)
psxy = project(sigma(u_sol)[0, 1], F_plot)
psyy = project(sigma(u_sol)[1, 1], F_plot)

fig, ax = plt.subplots(figsize=(6, 6))
# ukladani hodnot
values = []
values1 = []
values2 = []
values3 = []
values4 = []
for v in mesh.coordinates():
    x = v[0]
    y = v[1]
    
    sxx = psxx(x, y)
    sxy = psxy(x, y)
    syy = psyy(x, y)
    if x >= 0 - eps and x <= 0 + eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx, sxy, syy))
        values.append((x, y, sxx, sxy, syy))
    dtype = [('x', float), ('y', float), ('sxx', float), ('sxy', float), ('syy', float)]
    data_plot = np.sort(np.array(values, dtype=dtype), order='y')
    if x >= 250 - eps and x <= 250 + eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx, sxy, syy))
        values1.append((x, y, sxx, sxy, syy))
    dtype = [('x1', float), ('y1', float), ('sxx1', float), ('sxy1', float), ('syy1', float)]
    data_plot1 = np.sort(np.array(values1, dtype=dtype), order='y1')
    if x >= 500 - eps and x <= 500 + eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx, sxy, syy))
        values2.append((x, y, sxx, sxy, syy))
    dtype = [('x2', float), ('y2', float), ('sxx2', float), ('sxy2', float), ('syy2', float)]
    data_plot2 = np.sort(np.array(values2, dtype=dtype), order='y2')
    if x >= 750 - eps and x <= 750 + eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx, sxy, syy))
        values3.append((x, y, sxx, sxy, syy))
    dtype = [('x3', float), ('y3', float), ('sxx3', float), ('sxy3', float), ('syy3', float)]
    data_plot3 = np.sort(np.array(values3, dtype=dtype), order='y3') 
    if x >= 1000 - eps and x <= 1000 + eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx, sxy, syy))
        values4.append((x, y, sxx, sxy, syy))
    dtype = [('x4', float), ('y4', float), ('sxx4', float), ('sxy4', float), ('syy4', float)]
    data_plot4 = np.sort(np.array(values4, dtype=dtype), order='y4')

ax.plot(data_plot1['sxx1'], -data_plot1['y1'], c='tab:blue', ls='--', label='$x=0.25L$')
ax.plot(data_plot2['sxx2'], -data_plot2['y2'], c='tab:orange', ls='-.', label='$x=0.5L$')
ax.plot(data_plot3['sxx3'], -data_plot3['y3'], c='tab:green', ls=':', label='$x=0.75L$')
ax.plot(data_plot4['sxx4'], -data_plot4['y4'], c='tab:red', ls=(5, (10, 3)), label='$x=L$')

ax.set_xlabel(r'$\sigma_{x}$ [MPa]').set_fontsize(16)
ax.set_ylabel(r'$\xi$ [mm]').set_fontsize(16)
ax.legend(loc='best')
ax.grid(True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Napeti_GXX_fenics_saint_venant.png', dpi=600)
