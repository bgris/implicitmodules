# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:42:41 2019

@author: gris
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import src.DeformationModules.Combination as comb_mod
import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.ElasticOrder1 as defmod1
import src.DeformationModules.SilentLandmark as defmodsil
import src.Forward.Shooting as shoot
import src.Optimisation.ScipyOpti_attach as opti
import src.DataAttachment.Varifold as var
from src.Utilities import Rotation as rot
from src.Utilities.visualisation import my_close


# helper function
def my_plot(x, title="", col='*b'):
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], col)
    plt.title(title)
    plt.axis('equal')
    plt.show()
    

name_exp = 'acro_pure_parametric'
#name_exp = 'acro_pure_nonparametric'
#name_exp = 'acro_semi_parametric'


flag_show = False
#  common options
nu = 0.001
dim = 2
N=10
maxiter = 100
 
lam_var = 40.
sig_var = [50., 10.]

# define attachment_function
#def attach_fun(xsf, xst):
#    return var.my_dxvar_cost(xsf, xst, sig_var)
                   
def attach_fun(xsf, xst):
    (varcost0, dxvarcost0) = var.my_dxvar_cost(xsf, xst, sig_var[0])
    (varcost1, dxvarcost1) = var.my_dxvar_cost(xsf, xst, sig_var[1])
    costvar = varcost0 + varcost1
    dcostvar = dxvarcost0 + dxvarcost1
    return (lam_var * costvar, lam_var * dcostvar )
                   

coeffs =[0.01, 100, 0.01]
coeffs_str = '0_01__100__0_01'

# Source
path_data = '../data/'
with open(path_data + 'basi2btemp.pkl', 'rb') as f:
    _, lx = pickle.load(f)
    
Dx = 0.
Dy = 0.
height_source = 90.
height_target = 495.

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = height_source / (lmax - lmin)

nlx[:, 1] = Dy-scale * (nlx[:, 1] - lmax)
nlx[:, 0] = Dx+scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

# %% target
with open(path_data + 'basi2target.pkl', 'rb') as f:
    _, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = height_target / (lmax - lmin)
nlxt[:, 1] = - scale * (nlxt[:, 1] - lmax) 
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0])) 

xst = nlxt[nlxt[:, 2] == 2, 0:2]

# %% Silent Module
xs = nlx[nlx[:, 2] == 2, 0:2]
#xs = np.delete(xs, 3, axis=0)
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
ps = np.zeros(xs.shape)
param_sil = (xs, ps)
if(flag_show):
    my_plot(xs, "Silent Module", '*b')

# %% Modules of Order 0
sig0 = 10.
x0 = nlx[nlx[:, 2] == 1, 0:2]
Model0 = defmod0.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[1], nu)
p0 = np.zeros(x0.shape)
param_0 = (x0, p0)

if(flag_show):
    my_plot(x0, "Module order 0", 'or')

# %% Modules of Order 0
sig00 = 800.
x00 = np.array([[0., 0.]])
Model00 = defmod0.ElasticOrderO(sig00, x00.shape[0], dim, coeffs[0], nu)
p00 = np.zeros([1, 2])
param_00 = (x00, p00)

if(flag_show):
    my_plot(x00, "Module order 00", '+r')

# %% Modules of Order 1
sig1 = 60.

x1 = nlx[nlx[:, 2] == 1, 0:2]
C = np.zeros((x1.shape[0], 2, 1))
K, L = 10, height_source
a,b = 1/L, 3.
z = a*(x1[:,1]-Dy)
C[:,1,0] = K*((1-b)*z**2+b*z)
C[:,0,0] = 0.7*C[:,1,0]

Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[2], C, nu)

th = 0 * np.pi * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])

(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_1 = ((x1, R), (p1, PR))

if(flag_show):
    my_plot(x1, "Module order 1", 'og')

# %% Full model

if name_exp == 'acro_pure_nonparametric':
    Module = comb_mod.CompoundModules([Sil, Model0])
    Module.GD.fill_cot_from_param([param_sil, param_0])  
elif name_exp == 'acro_pure_parametric':
    Module = comb_mod.CompoundModules([Sil, Model00, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_00, param_1])
elif name_exp == 'acro_semi_parametric':
    Module = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_00, param_0, param_1])
else:
    print('unknown experiment type')

P0 = opti.fill_Vector_from_GD(Module.GD)



# %%

args = (Module, xst, attach_fun, N, 1e-7)

res = scipy.optimize.minimize(opti.fun, P0,
                              args=args,
                              method='L-BFGS-B',
                              jac=opti.jac,
                              bounds=None,
                              tol=None,
                              callback=None,
                              options={
                                  'maxcor': 10,
                                  'ftol': 1.e-09,
                                  'gtol': 1e-03,
                                  'eps': 1e-08,
                                  'maxfun': 100,
                                  'maxiter': maxiter,
                                  'iprint': 1,
                                  'maxls': 25
                              })

P1 = res['x']
opti.fill_Mod_from_Vector(P1, Module)
Module_optimized = Module.copy_full()
Modules_list = shoot.shooting_traj(Module, N)

# %% Visualisation
xst_c = my_close(xst)
xs_c = my_close(xs)
if(flag_show):
    for i in range(N + 1):
        plt.figure()
        xs_i = Modules_list[2 * i].GD.GD_list[0].GD
        xs_ic = my_close(xs_i)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    
        x0_i = Modules_list[2 * i].GD.GD_list[1].GD
        plt.plot(x0_i[:, 0], x0_i[:, 1], '*r', linewidth=2)
    
        x00_i = Modules_list[2 * i].GD.GD_list[2].GD
        plt.plot(x00_i[:, 0], x00_i[:, 1], 'or', linewidth=2)
    
        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
        plt.axis('equal')
        
        plt.show()

# %% With grid
#nxgrid, nygrid = (21, 21)  # create a grid for visualisation purpose
#xfigmin, xfigmax, yfigmin, yfigmax = -20, 20, 0, 40
#(a, b, c, d) = (xfigmin, xfigmax, yfigmin, yfigmax)
#[xx, xy] = np.meshgrid(np.linspace(xfigmin, xfigmax, nxgrid), np.linspace(yfigmin, yfigmax, nygrid))

hxgrid = 9
hsl = 1.2*height_source/2
a, b, c, d = (Dx-hsl/2, Dx+hsl/2, Dy, Dy+2*hsl) 
hygrid = np.round(hxgrid*(d-c)/(b-a))
nxgrid, nygrid = (2*hxgrid+1, 2*hygrid+1) # create a grid for visualisation purpose
[xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))



(nxgrid, nygrid) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid = defmodsil.SilentLandmark(grid_points.shape[0], dim)

param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)

Mod_tot = comb_mod.CompoundModules([Sil_grid, Module_optimized])

# %%
Modlist_opti_tot_grid = shoot.shooting_traj(Mod_tot, N)
# %% Plot with grid
xs_c = my_close(xs)
xst_c = my_close(xst)
if(flag_show):
    for i in range(N + 1):
        plt.figure()
        xgrid = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
        xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
        xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
        plt.plot(xsx, xsy, color='lightblue')
        plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
        xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
        xs_ic = my_close(xs_i)
        # plt.plot(xs[:,0], xs[:,1], '-b', linewidth=1)
        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
        plt.axis('equal')
        # plt.axis([-10,10,-10,55])
        # plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
        # plt.axis('off')
        plt.show()
    # plt.savefig(path_res + name_exp + '_t_' + str(i) + '.png', format='png', bbox_inches='tight')
#%% save figure for last time
i=N

plt.figure(figsize=(5,8))
xgrid = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
xs_ic = my_close(xs_i)
xs_c = my_close(xs)
plt.plot(xs_c[:,0], xs_c[:,1], '-b', linewidth=1)
plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
plt.axis('equal')
#plt.axis('off')
#plt.axis([-50,50,-10,120])
#plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
plt.axis('off')
plt.tight_layout()
if(flag_show):
    plt.show()
path_fig = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/Implicit/'
plt.savefig(path_fig + name_exp + coeffs_str + '.pdf', bbox_inches = 'tight')
print(coeffs)
print(coeffs_str)
plt.close()
## %% Shooting from controls
#
#Contlist = []
#for i in range(len(Modules_list)):
#    Contlist.append(Modules_list[i].Cont)
#
## %%
#Mod_cont_init = Modules_list[0].copy_full()
#Modlist_cont = shoot.shooting_from_cont_traj(Mod_cont_init, Contlist, 5)
#
## %% Visualisation
#xst_c = my_close(xst)
#xs_c = my_close(xs)
#for i in range(N + 1):
#    plt.figure()
#    xs_i = Modlist_cont[2 * i].GD.GD_list[0].GD
#    xs_ic = my_close(xs_i)
#    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
#    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
#    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
#    plt.axis('equal')
