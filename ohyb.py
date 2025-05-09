import os
from dolfin import *

#-----[ zatizeni ]-----
F=1000
#-----[ materialove charakteristiky ]-----

E=210000
nu=0.3
c = 50

#-----[ dopocitane veliciny ]-----

mu=E/2/(1+nu)
lambda_=E*nu/(1+nu)/(1-nu)#/(1+nu)/(1-nu)



print('nacitam sit')

mesh=Mesh("prut.xml")
boundaries=MeshFunction("size_t",mesh,"prut_facet_region.xml")

#-----[definice indexovani oblasti a hranic]-----

ds=Measure("ds",domain=mesh,subdomain_data=boundaries)

#-----[ prostory funkci ]-----

print('generuji prostory funkci')

V=VectorFunctionSpace(mesh,"CG",2)

u=TrialFunction(V)
w=TestFunction(V)

#-----[ okrajove podminky ]-----

print('sestavuji okrajove podminky')
eps=1
class pointload(UserExpression):
  def eval(self,value,x):

    if near(x[0],0.0,eps) and near(x[1],0.0,eps):
      value[0]=0.0
      value[1]=-F
    else:
      value[0]=0.
      value[1]=0.

  def value_shape(self):
    return (2,)

class tractions(UserExpression):
  def eval(self,value,x):
    value[0]=0.
    value[1]=-F
  def value_shape(self):
    return (2,)

trs=tractions(element=V.ufl_element())
pl=pointload(element=V.ufl_element())

null=Constant(0.0)

bcs=[DirichletBC(V.sub(0),null,boundaries,2),
     DirichletBC(V.sub(1),null,boundaries,2)]

#-----[ definice tenzoru napeti a deforamace ]-----

def epsilon(u):
  return 0.5*(grad(u) + grad(u).T)

def sigma(u):
    du = grad(u)
    trdu = tr(du)
    return lambda_ * trdu * Identity(len(u)) + 2 * mu * epsilon(u)

#-----[ slaba formulace ]-----
a=c*inner(sigma(u),epsilon(w))*dx
L=dot(pl,w)*ds(4)

#-----[ reseni ]-----

u=Function(V)
solve(a==L,u,bcs)

#------[ export reseni do pvd formatu ]-----

file1=File('u_prut3d0stupnu.pvd')
file1 << u

V_scalar = FunctionSpace(mesh,'CG',1)
u_y = project(u.sub(1), V_scalar)
file_u_y = File("uy_prut3dd.pvd")
file_u_y << u_y

V_scalar = FunctionSpace(mesh,'CG',1)
u_x = project(u.sub(0), V_scalar)
file_u_y = File("ux_prut3ddxxxxx.pvd")
file_u_y << u_x
import matplotlib.pyplot as plt
import numpy as np

uy_vals = u_y.compute_vertex_values(mesh)
coords = mesh.coordinates()
x_coords = coords[:, 0] 

# Seřazení podle x pro správné vykreslení
sorted_indices = np.argsort(x_coords)
x_sorted = x_coords[sorted_indices]
uy_sorted = uy_vals[sorted_indices]

from matplotlib.ticker import FormatStrFormatter

# Vykreslení grafu
plt.figure(figsize=(6, 6))
plt.plot(x_sorted, -uy_sorted, label='w(x) [mm]')  
plt.xlabel("Souřadnice $x$ [mm]").set_fontsize(16)
plt.ylabel("Průhyb $w(x)$ [mm]").set_fontsize(16)
plt.grid(True)
ax = plt.gca()  
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.savefig('průhyb fenics')

s=sigma(u)-(1./3)*tr(sigma(u))*Identity(len(u))  # deviatoric stress
von_Mises=sqrt(3./2*inner(s, s))
V=FunctionSpace(mesh,'P',1)
von_Mises=project(von_Mises, V)
file2=File('von_Mises_prut3d.pvd')
file2 << von_Mises


import numpy as np
import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
F_plot=FunctionSpace(mesh,"CG",1)
T_plot=TensorFunctionSpace(mesh,"CG",1)

plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.9,
    "lines.linewidth": 2.2,
})

p=project(sigma(u),T_plot)
psxx=project(sigma(u)[0,0],F_plot)
psxy=project(sigma(u)[0,1],F_plot)
psyy=project(sigma(u)[1,1],F_plot)


fig, ax=plt.subplots(figsize=(6,6))
# Vytvoření listu pro ukládání hodnot
values = []
values1 = []
values2 = []
values3 = []
values4 = []
for v in mesh.coordinates():
  x=v[0]
  y=v[1]

  
  sxx=psxx(x,y)
  sxy=psxy(x,y)
  syy=psyy(x,y)
  if x>=0-eps and x<=0+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values.append((x,y,sxx,sxy,syy))
  dtype=[('x',float),('y',float),('sxx',float),('sxy',float),('syy',float)]
  data_plot=np.sort(np.array(values,dtype=dtype),order='y')
  if x>=250-eps and x<=250+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values1.append((x,y,sxx,sxy,syy))
  dtype=[('x1',float),('y1',float),('sxx1',float),('sxy1',float),('syy1',float)]
  data_plot1=np.sort(np.array(values1,dtype=dtype),order='y1')
  if x>=500-eps and x<=500+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values2.append((x,y,sxx,sxy,syy))
  dtype=[('x2',float),('y2',float),('sxx2',float),('sxy2',float),('syy2',float)]
  data_plot2=np.sort(np.array(values2,dtype=dtype),order='y2')
  if x>=750-eps and x<=750+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values3.append((x,y,sxx,sxy,syy))
  dtype=[('x3',float),('y3',float),('sxx3',float),('sxy3',float),('syy3',float)]
  data_plot3=np.sort(np.array(values3,dtype=dtype),order='y3') 
  if x>=1000-eps and x<=1000+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values4.append((x,y,sxx,sxy,syy))
  dtype=[('x4',float),('y4',float),('sxx4',float),('sxy4',float),('syy4',float)]
  data_plot4=np.sort(np.array(values4,dtype=dtype),order='y4')


ax.plot(data_plot1['sxx1'], -data_plot1['y1'], c='tab:blue', ls='--', label='$x=0.25L$')
ax.plot(data_plot2['sxx2'], -data_plot2['y2'], c='tab:orange', ls='-.', label='$x=0.5L$')
ax.plot(data_plot3['sxx3'], -data_plot3['y3'], c='tab:green', ls=':', label='$x=0.75L$')
ax.plot(data_plot4['sxx4'], -data_plot4['y4'], c='tab:red', ls=(5, (10, 3)), label='$x=L$')

ax.set_xlabel(r'$\sigma_{x}$ [MPa]').set_fontsize(16)
ax.set_ylabel(r'$\xi$ [mm]').set_fontsize(16)
ax.legend(loc='best')
ax.grid(True)
plt.tight_layout()
plt.savefig('Napeti GXX fenics',dpi=600)

fig, ax=plt.subplots(figsize=(6,6))
values = []
values1 = []
values2 = []
values3 = []
values4 = []
for v in mesh.coordinates():
  x=v[0]
  y=v[1]

  
  sxx=psxx(x,y)
  sxy=psxy(x,y)
  syy=psyy(x,y)
  if x>=0-eps and x<=0+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values.append((x,y,sxx,sxy,syy))
  dtype=[('x',float),('y',float),('sxx',float),('sxy',float),('syy',float)]
  data_plot=np.sort(np.array(values,dtype=dtype),order='y')
  if x>=250-eps and x<=250+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values1.append((x,y,sxx,sxy,syy))
  dtype=[('x1',float),('y1',float),('sxx1',float),('sxy1',float),('syy1',float)]
  data_plot1=np.sort(np.array(values1,dtype=dtype),order='y1')
  if x>=500-eps and x<=500+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values2.append((x,y,sxx,sxy,syy))
  dtype=[('x2',float),('y2',float),('sxx2',float),('sxy2',float),('syy2',float)]
  data_plot2=np.sort(np.array(values2,dtype=dtype),order='y2')
  if x>=750-eps and x<=750+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values3.append((x,y,sxx,sxy,syy))
  dtype=[('x3',float),('y3',float),('sxx3',float),('sxy3',float),('syy3',float)]
  data_plot3=np.sort(np.array(values3,dtype=dtype),order='y3') 
  if x>=1000-eps and x<=1000+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values4.append((x,y,sxx,sxy,syy))
  dtype=[('x4',float),('y4',float),('sxx4',float),('sxy4',float),('syy4',float)]
  data_plot4=np.sort(np.array(values4,dtype=dtype),order='y4')

    
ax.plot(data_plot['sxy'],-data_plot['y'],c='#9467bd',ls=('-'), label='$x=0$')
ax.plot(data_plot1['sxy1'], -data_plot1['y1'], c='tab:blue', ls='--', label='$x=0.25L$')
ax.plot(data_plot2['sxy2'], -data_plot2['y2'], c='tab:orange', ls='-.', label='$x=0.5L$')
ax.plot(data_plot3['sxy3'], -data_plot3['y3'], c='tab:green', ls=':', label='$x=0.75L$')

ax.set_xlabel(r'$\tau_{xz}$ [MPa]').set_fontsize(16)
ax.set_ylabel(r'$\xi$ [mm]').set_fontsize(16)
ax.legend(loc='best')
ax.grid(True)
ax.set_xticks([0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Napeti Txz fenics',dpi=600)

fig, ax=plt.subplots(figsize=(6,6))
values = []
values1 = []
values2 = []
values3 = []
values4 = []
for v in mesh.coordinates():
  x=v[0]
  y=v[1]

  
  sxx=psxx(x,y)
  sxy=psxy(x,y)
  syy=psyy(x,y)
  if x>=0-eps and x<=0+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values.append((x,y,sxx,sxy,syy))
  dtype=[('x',float),('y',float),('sxx',float),('sxy',float),('syy',float)]
  data_plot=np.sort(np.array(values,dtype=dtype),order='y')
  if x>=250-eps and x<=250+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values1.append((x,y,sxx,sxy,syy))
  dtype=[('x1',float),('y1',float),('sxx1',float),('sxy1',float),('syy1',float)]
  data_plot1=np.sort(np.array(values1,dtype=dtype),order='y1')
  if x>=500-eps and x<=500+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values2.append((x,y,sxx,sxy,syy))
  dtype=[('x2',float),('y2',float),('sxx2',float),('sxy2',float),('syy2',float)]
  data_plot2=np.sort(np.array(values2,dtype=dtype),order='y2')
  if x>=750-eps and x<=750+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values3.append((x,y,sxx,sxy,syy))
  dtype=[('x3',float),('y3',float),('sxx3',float),('sxy3',float),('syy3',float)]
  data_plot3=np.sort(np.array(values3,dtype=dtype),order='y3') 
  if x>=1000-eps and x<=1000+eps:
    print("{:1.3e} {:+1.3e} {:+1.3e} {:+1.3e} {:+1.3e}".format(x,y,sxx,sxy,syy))
    values4.append((x,y,sxx,sxy,syy))
  dtype=[('x4',float),('y4',float),('sxx4',float),('sxy4',float),('syy4',float)]
  data_plot4=np.sort(np.array(values4,dtype=dtype),order='y4')

    

ax.plot(data_plot1['syy1'], -data_plot1['y1'], c='tab:blue', ls='--', label='$x=0.25L$')
ax.plot(data_plot2['syy2'], -data_plot2['y2'], c='tab:orange', ls='-.', label='$x=0.5L$')
ax.plot(data_plot3['syy3'], -data_plot3['y3'], c='tab:green', ls=':', label='$x=0.75L$')

ax.set_xlabel(r'$\sigma_{z}$ [MPa]').set_fontsize(16)
ax.set_ylabel(r'$\xi$ [mm]').set_fontsize(16)
ax.legend(loc='best')
ax.grid(True)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('Napeti Gz fenics',dpi=600)

# --- Výpočet maximálních hodnot složek napětí ---
sigmax_vals = []
sigmay_vals = []
tauxy_vals = []

for v in mesh.coordinates():
    x = v[0]
    y = v[1]
    sigmax_vals.append(psxx(x, y))
    sigmay_vals.append(psyy(x, y))
    tauxy_vals.append(psxy(x, y))


max_sigmax = np.max(np.abs(psxx.vector().get_local()))
max_sigmay = np.max(np.abs(psyy.vector().get_local()))
max_tauxy = np.max(np.abs(psxy.vector().get_local()))

print("\n--- Maximální hodnoty napětí ---")
print(f"Maximální |σₓ| = {max_sigmax:.3f} MPa")
print(f"Maximální |σ_y| = {max_sigmay:.3f} MPa")
print(f"Maximální |τ_xy| = {max_tauxy:.3f} MPa")

