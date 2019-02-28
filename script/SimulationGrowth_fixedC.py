# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:26:14 2019

@author: gris
"""
import sys
sys.path.append("../")
sys.path.append("../../")


import scipy .optimize
import numpy as np
import matplotlib.pyplot as plt

from implicitmodules.src import rotation as rot
import src.DeformationModules.SilentLandmark as defmodsil
import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.ElasticOrder1 as defmod1
import src.DeformationModules.ElasticOrder1C as defmod1C
import src.DeformationModules.Combination as comb_mod
import implicitmodules.src.data_attachment.varifold as var

import src.Forward.shooting as shoot

import implicitmodules.src.Optimisation.ScipyOpti as opti

#%%

path_data = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/data/CorticalFolds/Simulations/'

#%%
source = np.loadtxt(path_data + 'deformation1-orig.txt')[:,:-1]
target = np.loadtxt(path_data + 'deformation1.txt')[:,:-1]

#%%

plt.plot(source[:,0], source[:,1], 'x')
plt.plot(target[:,0], target[:,1], 'x')

#%%
x1 = source.copy()
xs = source.copy()

source_layer = np.reshape(source, [-1, 17, 2])
up = source_layer[0]
down = source_layer[-1]
left = source_layer[1:-1, 0]
right = source_layer[1:-1, -1]
xs = np.concatenate([up, down, left, right])

target_layer = np.reshape(target, [-1, 17, 2])
up = target_layer[0]
down = target_layer[-1]
left = target_layer[1:-1, 0]
right = target_layer[1:-1, -1]
target_boundary = np.concatenate([up, down, left, right])


#%% Initialize GD for module of order 1
th = 0.2*np.pi
th = th*np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for  i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

dim = 2
dimCont = 1
C = np.zeros((x1.shape[0], dim, dimCont))
C[source[:,1]>0,0,0] = 0. 
indi = source[:,1]<=0
C[indi,0,0] = -source[indi,1] 
C[indi,1,0] = -source[indi,1] 

x00 = np.array([[0., 0.]])
coeffs = [1., 1.]
sig00 = 30
sig1 = 1
nu = 0.001
dim = 2
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1],C,  nu)
Model00 = defmod0.ElasticOrderO(sig00, x00.shape[0], dim, 1., nu)
#%%

Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model1])

#%%
ps = np.zeros(xs.shape)
p00 = np.zeros([1, 2])
(p1, PR) = (np.zeros(x1.shape), np.zeros(R.shape))

param_sil = (xs, ps)
param_00 = (x00, p00)
param_1 = ((x1, R), (p1, PR))
#%%
param = [param_sil, param_00, param_1]
#param = [param_sil, param_00, param_1]
GD = Mod_el_init.GD.copy()

#%%
Mod_el_init.GD.fill_cot_from_param(param)


#%%
Mod_el = Mod_el_init.copy_full()

#%%
N=5


#%%
Mod_el_opti = Mod_el_init.copy_full()
P0 = opti.fill_Vector_from_GD(Mod_el_opti.GD)
#%%
flag_xs = np.zeros(xs.shape).flatten()
flag_x00 = np.zeros([1, 2]).flatten()
(flag_x1, flag_xR) = (np.zeros(x1.shape).flatten(), np.zeros(R.shape).flatten())
flag_x1tot = np.concatenate((flag_x1, flag_xR))

flag_ps = np.ones(xs.shape).flatten()
flag_p00 = np.zeros([1, 2]).flatten()
(flag_p1, flag_PR) = (np.zeros(x1.shape).flatten(), np.zeros(R.shape).flatten())
flag_p1tot = np.concatenate((flag_p1, flag_PR))

#flag_x = np.concatenate
flag_x = np.concatenate((flag_xs, flag_x00, flag_x1tot))
flag_P = np.concatenate((flag_ps, flag_p00, flag_p1tot))
flag = np.concatenate((flag_x, flag_P))
# %%
lam_var = 1000.
N = 5
def attach_fun(x,y):
    return np.sum( (x-y)**2 ), 2*(x-y)

sigma_var = 1.
def attach_fun(x,y):
    return var.my_dxvar_cost(x, y, sigma_var)

args = (Mod_el_opti, target_boundary, lam_var, N, 0.001, flag, attach_fun)

res = scipy.optimize.minimize(opti.fun, P1,
                              args=args,
                              method='L-BFGS-B', jac=opti.jac, bounds=None, tol=None, callback=None,
                              options={'disp': True, 'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03,
                                       'eps': 1e-08, 'maxfun': 100, 'maxiter': 50, 'iprint': -1, 'maxls': 20})

P1 = res['x']

# %%
opti.fill_Mod_from_Vector(P1, Mod_el_opti)                                       
#%%
Modlist_opti_tot = shoot.shooting_traj(Mod_el_opti, N)

#%% Visualisation
xst_c = target.copy()
xs_c = xs.copy()
for i in range(N + 1):
    plt.figure()
    xs_i = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
    xs_ic = xs_i.copy()
    plt.plot(xst_c[:, 0], xst_c[:, 1], 'xk', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], 'xr', linewidth=5)
#    plt.plot(xs_c[:, 0], xs_c[:, 1], 'xb', linewidth=1)
    plt.axis('equal')
plt.figure()
xs_i = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
xs_ic = xs_i.copy()
plt.plot(xs_ic[:, 0], xs_ic[:, 1], 'xr', linewidth=5)
plt.axis('equal')


im0 = np.reshape(C[:,0,0], [-1, 17])
im1 = np.reshape(C[:,1,0], [-1, 17])
plt.figure()
plt.imshow(im0)
plt.colorbar()
plt.figure()
plt.imshow(im1)
plt.colorbar()



plt.figure()
xs_i = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
x1_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD[0]
xs_ic = xs_i.copy()
plt.plot(xst_c[:, 0], xst_c[:, 1], 'xk', linewidth=1)
plt.plot(xs_ic[:, 0], xs_ic[:, 1], 'xr', linewidth=5)
plt.plot(x1_i[:, 0], x1_i[:, 1], 'xr', linewidth=5)
#    plt.plot(xs_c[:, 0], xs_c[:, 1], 'xb', linewidth=1)
plt.axis('equal')

