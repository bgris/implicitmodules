# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:58:34 2019

@author: gris
"""


import scipy .optimize
import scipy.interpolate as si
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
sys.path.append("../../")


import src.DeformationModules.SilentLandmark as defmodsil
import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.ElasticOrder1 as defmod1
import src.DeformationModules.Combination as comb_mod

import src.Forward.shooting as shoot
import src.Backward.Backward as bckwrd
#%%
#from implicitmodules.src import constraints_functions as con_fun, field_structures as fields, rotation as rot, shooting as shoot_old, \
#    useful_functions as fun, modules_operations as modop, functions_eta as fun_eta, visualisation as visu
from implicitmodules.src.visualisation import my_close
from implicitmodules.src import rotation as rot
import implicitmodules.src.data_attachment.varifold as var

import implicitmodules.src.Optimisation.ScipyOpti as opti

import pickle
#%%  source


# %%
xmin, xmax = -10, 10
ymin, ymax = -5, 5
nx, ny = 11, 5

X0 = np.linspace(xmin, xmax, nx)
Y0 = np.linspace(ymin, ymax, ny)
Z0 = np.meshgrid(X0, Y0)

Z = np.reshape(np.swapaxes(Z0, 0, 2), [-1, 2])

Z_c = np.concatenate([np.array([X0, np.zeros([nx]) + ymin]).transpose(),
                      np.array([np.zeros([ny]) + xmax, Y0]).transpose(),
                      np.array([np.flip(X0), np.zeros([nx]) + ymax]).transpose(),
                      np.array([np.zeros([ny]) + xmin, np.flip(Y0)]).transpose()])

# %%
#plt.plot(Z[:, 0], Z[:, 1], '.')
#plt.plot(Z_c[:, 0], Z_c[:, 1], '-')
#plt.axis('equal')
#plt.show()


# %%
x1 = Z.copy()
xs = Z_c.copy()
x_s = x1.copy()
# %% parameter for module of order 1
th = 0. * np.pi
th = th * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

dimh = 2
C = np.zeros((x1.shape[0], 2, dimh))
L = 38.
K = 100
a, b = -2 / L ** 3, 3 / L ** 2


def define_C1(x, y):
    return y.copy()


#indi_ll = np.where(x1[:,0] < -5)
#indi_lr = np.where(x1[:,0] > -5 and x1[:,0] < 0)

#indi_rl = np.where(x1[:,0] < 5 and x1[:,0] > 0)
#indi_rr = np.where(x1[:,0] > 5)

indi_l = np.where(x1[:,0] < 0)
indi_r = np.where(x1[:,0] > 0)

C[indi_l, 0, 0] = define_C1(x1[indi_l, 0], x1[indi_l, 1]) 
C[indi_r, 1, 0] = 0.

C[indi_r, 0, 1] = define_C1(x1[indi_r, 0], x1[indi_r, 1]) 
C[indi_l, 1, 1] = 0. 

# Define C by spline interpolation:
C = np.ones([x1.shape[0], 2, dimh])
x_spli_init = np.linspace(xmin, xmax, 4)
valc_spli_init = np.array([1, 0.9, 0.1, 0])
spli_valc = si.CubicSpline(x_spli_init, valc_spli_init, bc_type=((1,0.0), (1,0.0)))
valc_x = spli_valc(x1[:,0]) 
C[:,0,0] = valc_x * x1[:,1]
C[:,1,0 ] = 0.
valc_x =1 -  spli_valc(x1[:,0]) 
C[:,0,1] = valc_x * x1[:,1]
C[:,1,1] = 0.




ZX = define_C1(X0, np.zeros([nx]))
ZY = define_C1(np.zeros([ny]), Y0)
name_exp = 'linear_angle05pi'

xfigmin = -10
xfigmax = 10
yfigmin = -5
yfigmax = 55

xfigmin = -10
xfigmax = 10
yfigmin = -10
yfigmax = 10

#%%
x00 = np.array([[0., 0.]])
coeffs = [1., 0.01]
sig0 = 20
sig00 = 200
sig1 = 5
nu = 0.001
dim = 2
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
#Model0 = defmod0.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = defmod0.ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
#%% 

#Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])

#Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model1])

Mod_el_init = comb_mod.CompoundModules([Sil, Model1])


#%%
p00 = np.zeros([1, 2])

#p0 = np.zeros(x0.shape)
ps = np.zeros(xs.shape)
ps[0, :] = -5., -5.
ps[10, :] = -5., 5.
ps[27, :] = 5., -5.
ps[16, :] = 5., 5.


ps[0, :] = 5., 5.
ps[10, :] = -5., 5.
ps[27, :] = -5., 5.
ps[16, :] = 5., 5.


ps *= 0.03

plt.figure()
plt.plot(xs[:,0], xs[:,1], '-g')
plt.plot(x1[:,0], x1[:,1], '.b')
plt.quiver(xs[:,0], xs[:,1], ps[:,0], ps[:,1])
c0_rep = np .transpose(np.array([C[:,0,0], C[:,0,0]]))
c1_rep = np .transpose(np.array([C[:,0,1], C[:,0,1]]))
vec_C0 = R[:,:,0] * c0_rep
vec_C1 = R[:,:,0] * c1_rep
plt.quiver(x1[:,0], x1[:,1], vec_C0[:,0], vec_C0[:,1], color='r')
plt.quiver(x1[:,0], x1[:,1], vec_C1[:,0], vec_C1[:,1], color='b')
plt.axis('equal')
plt.show()

(p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))

param_sil = (xs, ps)
#param_0 = (x0, p0)
param_00 = (np.zeros([1, 2]), p00)
param_1 = ((x1, R), (p1, PR))

#%%
param = [param_sil, param_1]
#param = [param_sil, param_00, param_1]
GD = Mod_el_init.GD.copy()

#%%
Mod_el_init.GD.fill_cot_from_param(param)


#%%
Mod_el = Mod_el_init.copy_full()

N=10
Modlist_opti_tot = shoot.shooting_traj(Mod_el, N)


#%% Visualisation
#xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xs_i = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
    x1_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD[0]
    R_i =  Modlist_opti_tot[2 * i].GD.GD_list[1].GD[1]
    cont_i = Modlist_opti_tot[2*i].Cont[1]
    vec_C0_i = cont_i[0] * R_i[:,:,0] * c0_rep
    vec_C1_i = cont_i[1] * R_i[:,:,0] * c1_rep
    plt.quiver(x1_i[:,0], x1_i[:,1], vec_C0_i[:,0], vec_C0_i[:,1], color='r')
    plt.quiver(x1_i[:,0], x1_i[:,1], vec_C1_i[:,0], vec_C1_i[:,1], color='b')
    plt.plot(xs_i[:,0], xs_i[:,1], '-g')
    plt.plot(x1_i[:,0], x1_i[:,1], '.b')
    plt.axis('equal')
plt.show()
 
# Visualise grid
nxgrid = 21
nygrid = 21
(a, b, c, d) = (xfigmin, xfigmax, yfigmin, yfigmax)
[xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))
(nxgrid, nygrid) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid =  defmodsil.SilentLandmark(grid_points.shape[0], dim)

param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)


Mod_el_init.GD.fill_cot_from_param(param)
Mod_tot = comb_mod.CompoundModules([Sil_grid, Mod_el_init])
# Mod_tot
# %%
Modlist_opti_tot_grid = shoot.shooting_traj(Mod_tot, N)



#%% Visualisation
#xst_c = my_close(xst)
#xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xgrid_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
    xsx = xgrid_i[:, 0].reshape((nxgrid, nygrid))
    xsy = xgrid_i[:, 1].reshape((nxgrid, nygrid)) 
    plt.plot(xsx, xsy, color='lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
    xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
    x1_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[1].GD[0]
    R_i =  Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[1].GD[1]
    cont_i = Modlist_opti_tot_grid[2*i].Cont[1][1]
    vec_C0_i = cont_i[0] * R_i[:,:,0] * c0_rep
    vec_C1_i = cont_i[1] * R_i[:,:,0] * c1_rep
    plt.quiver(x1_i[:,0], x1_i[:,1], vec_C0_i[:,0], vec_C0_i[:,1], color='r')
    plt.quiver(x1_i[:,0], x1_i[:,1], vec_C1_i[:,0], vec_C1_i[:,1], color='b')
    plt.plot(xs_i[:,0], xs_i[:,1], '-g')
    plt.plot(x1_i[:,0], x1_i[:,1], '.b')
    plt.axis('equal')
plt.show()








