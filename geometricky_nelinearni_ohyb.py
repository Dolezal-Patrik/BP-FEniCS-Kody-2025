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
u = Function(V)
w = TestFunction(V)
du = TrialFunction(V)

#-----[ okrajove podminky ]-----
eps = 1
class PointLoad(UserExpression):
    def eval(self, value, x):
        if near(x[0], 0.0, eps) and near(x[1], 0.0, eps):
            value[0] = 0.0
            value[1] = -F_load
        else:
            value[0] = value[1] = 0.0
    def value_shape(self):
        return (2,)
pl = PointLoad(element=V.ufl_element())
bcs = [DirichletBC(V, Constant((0, 0)), boundaries, 2)]


def F(u):
    "deformacni gradient"
    return Identity(2) + grad(u)

def E(u):
    "green-lagrangeuv deformacni tenzor"
    F_ = F(u)
    return 0.5 * (F_.T * F_ - Identity(2))

def S(u):
    "druhy piola-kirchhoffuv napetovy tenzor (Saint Venant-Kirchhoff)"
    E_ = E(u)
    return lambda_ * tr(E_) * Identity(2) + 2 * mu * E_

def sigma(u):
    "cauchyuv napetovy tenzor"
    F_ = F(u)
    J_ = det(F_) + DOLFIN_EPS
    return (1/J_) * F_ * S(u) * F_.T

#-----[ nelinearni slaba formulace ]-----
F_ = F(u)
delta_E = 0.5 * (F_.T * grad(w) + grad(w).T * F_)
form = c * inner(S(u), delta_E) * dx - dot(pl, w) * ds(1)

#-----[ jakobian slabe formulace ]-----
J = derivative(form, u, du)

#-----[ newtonova metoda ]-----
problem = NonlinearVariationalProblem(form, u, bcs, J)
solver = NonlinearVariationalSolver(problem)
solver.parameters["nonlinear_solver"] = "newton"
solver.parameters["newton_solver"]["linear_solver"] = "mumps"
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-5
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-5
solver.parameters["newton_solver"]["maximum_iterations"] = 25
solver.solve()

#-----[ Export výsledků ]-----


V_scalar = FunctionSpace(mesh, "CG", 1)
u_y = project(u.sub(1), V_scalar)
File("uy_prut.pvd") << u_y

import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.9,
    "lines.linewidth": 2.2,
})

uy_vals = u_y.compute_vertex_values(mesh)
x_coords = mesh.coordinates()[:, 0]
idx = np.argsort(x_coords)
plt.figure(figsize=(6, 6))
plt.plot(x_coords[idx], -uy_vals[idx], label="w(x) [mm]")
plt.xlabel("x [mm]").set_fontsize(16)
plt.ylabel("Průhyb [mm]").set_fontsize(16)
plt.grid(True)
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("pruhyb.png")


T_plot = TensorFunctionSpace(mesh, "CG", 1)
F_plot = FunctionSpace(mesh, "CG", 1)
sxx = project(sigma(u)[0, 0], F_plot)
sxy = project(sigma(u)[0, 1], F_plot)
syy = project(sigma(u)[1, 1], F_plot)

values0, values1, values2, values3, values4 = [], [], [], [], []
for x, y in mesh.coordinates():
    sxx_val = sxx(x, y)
    sxy_val = sxy(x, y)
    syy_val = syy(x, y)
    if abs(x - 0) <= eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx_val, sxy_val, syy_val))
        values0.append((x, y, sxx_val, sxy_val, syy_val))
    if abs(x - 250) <= eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx_val, sxy_val, syy_val))
        values1.append((x, y, sxx_val, sxy_val, syy_val))
    if abs(x - 500) <= eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx_val, sxy_val, syy_val))
        values2.append((x, y, sxx_val, sxy_val, syy_val))
    if abs(x - 750) <= eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx_val, sxy_val, syy_val))
        values3.append((x, y, sxx_val, sxy_val, syy_val))
    if abs(x - 1000) <= eps:
        print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x, y, sxx_val, sxy_val, syy_val))
        values4.append((x, y, sxx_val, sxy_val, syy_val))

dtype = [('x', float), ('y', float), ('sxx', float), ('sxy', float), ('syy', float)]
data_plot0 = np.sort(np.array(values0, dtype=dtype), order='y')
data_plot1 = np.sort(np.array(values1, dtype=dtype), order='y')
data_plot2 = np.sort(np.array(values2, dtype=dtype), order='y')
data_plot3 = np.sort(np.array(values3, dtype=dtype), order='y')
data_plot4 = np.sort(np.array(values4, dtype=dtype), order='y')

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(data_plot1['sxx'], -data_plot1['y'], c='tab:blue', ls='--', label='$x=0.25L$')
ax.plot(data_plot2['sxx'], -data_plot2['y'], c='tab:orange', ls='-', label='$x=0.5L$')
ax.plot(data_plot3['sxx'], -data_plot3['y'], c='tab:green', ls=':', label='$x=0.75L$')
ax.plot(data_plot4['sxx'], -data_plot4['y'], c='tab:red', ls=(5, (10, 3)), label='$x=L$')
ax.set_xlabel(r'$\sigma_x$ [MPa]').set_fontsize(16)
ax.set_ylabel(r'y [mm]').set_fontsize(16)
ax.grid(True)
ax.legend()
plt.gca().invert_yaxis()
ax.set_xlim(-100,100)
ax.set_ylim(42,-2)
plt.tight_layout()
plt.savefig("napeti_sigmaxx_nonlin.png")
