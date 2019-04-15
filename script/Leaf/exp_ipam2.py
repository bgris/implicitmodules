import sys
sys.path.append('/home/trouve/Implicit/implicitmodules')

import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np
import scipy.optimize
import os

import src.DeformationModules.Combination as comb_mod
import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.ElasticOrder1 as defmod1
import src.DeformationModules.SilentLandmark as defmodsil
import src.Forward.Shooting as shoot
import src.Optimisation.ScipyOpti as opti
from src.Utilities import Rotation as rot
from src.Utilities.visualisation import my_close

import src.ExpIpam.utils as ipam


    
################################################################################
#########



# %%  Basi experiments
flag = 'basi'
(source, target) = ('basi1b.pkl', 'basi1t.pkl')
(height_source, height_target, Dx, Dy) = (38., 100., 0., 0.)
dir_res = '/home/trouve/Dropbox/Talks/Pics/ResImp/ResIpam/'
dir_Mysh = '/home/trouve/Dropbox/Talks/2019_04_IPAM/tex/Mysh/'

#%%
# parametric

name_exp = 'leaf_ba2_p'
coeffs = [0.01, 100, 0.001]
nu = 0.001
(sig00, sig0, sig1) = (1000., 15., 30.)
(lam_var, sig_var) = (10., 30.)
attach_var = (lam_var, sig_var)

(P0, outvar) = ipam.exp_ipam_init(flag = flag, 
           source = source, target = target, name_exp = name_exp, 
           dir_res = dir_res,  
           Dx = Dx, Dy = Dy, height_source = height_source,
           height_target = height_target, sig00 = sig00, sig0 = sig0, sig1 = sig1, 
           coeffs = coeffs, nu = nu, lam_var = lam_var, sig_var = sig_var)
          # outvar = (Module, xs, xst, opti.fun, opti.jac)
          

maxiter = 30
max_fun = 100
N = 10
P1 = ipam.exp_ipam_optim((P0, outvar), 
                         attach_var, maxiter = maxiter, maxfun = 200, N=N)

# Save the result in case
filepkl = dir_res + name_exp + ".pkl"
with open(filepkl, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([P0, P1, outvar, attach_var, maxiter, max_fun, N], f)

# plottings and savings
N = 15
invar = (N, flag, name_exp, height_source, height_target, Dx, Dy)
(Module, xs, xst, fun, jac) = outvar
outvarplot = (Module, P1, xs, xst, dir_res, dir_Mysh)
ipam.exp_ipam_plot(invar, outvarplot)
#%%

from_past = True
if (from_past):
    with open(filepkl, 'rb') as f:  # Python 3: open(..., 'rb')
     P0, P1, outvar, attach_var, maxiter, max_fun, N = pickle.load(f)
else:
    P0 = P1.copy()
         
P1 = ipam.exp_ipam_optim((P0, outvar), attach_var, maxiter = 1, maxfun = 200, N=10)

# Save  again the result in case
filepkl = dir_res + name_exp + ".pkl"
with open(filepkl, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([P0, P1, outvar, attach_var, maxiter, max_fun, N], f)
    
# plottings and savings
N = 15
invar = (N, flag, name_exp, height_source, height_target, Dx, Dy)
(Module, xs, xst, fun, jac) = outvar
outvarplot = (Module, P1, xs, xst, dir_res, dir_Mysh)
ipam.exp_ipam_plot(invar, outvarplot)
 
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leafaa', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.001, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)

#%%
## LDDMM
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leafaar', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 2000, sig0 = 15, sig1 = 30, 
           coeffs =[10, 0.01, 10], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leafa', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.01, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf0', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.05, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf1', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.1, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf2', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.1, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 100.)

#%%
my_exp(source = 'leafbasi.pkl', target = 'leafbasit.pkl', name_exp = 'leaf3', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 10, sig1 = 30, 
           coeffs =[0.01, 1, 0.01], nu = 0.001, lam_var = 100., sig_var = 15.,
                   N=10, maxiter = 100.)

#%% Experiences on basipetal

#%%
# lddmm
my_exp(source = 'leafbasi.pkl', target = 'leafbasit.pkl', name_exp = 'leaf_ba1_ld', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 1000, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.01, 100], nu = 0.001, lam_var = 10., sig_var = 30.,
                   N=10, maxiter = 100)

# parametric
my_exp(source = 'leafbasi.pkl', target = 'leafbasit.pkl', name_exp = 'leaf_ba1_p', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 1000, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 100, 0.001], nu = 0.001, lam_var = 10., sig_var = 30.,
                   N=10, maxiter = 100.)
#%%
# semi- param
my_exp(source = 'leafbasi.pkl', target = 'leafbasit.pkl', name_exp = 'leaf_ba1_sp', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 1000, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.01, 0.001], nu = 0.001, lam_var = 10., sig_var = 30.,
                   N=10, maxiter = 100.)
#%%
my_exp(source = 'leafbasi.pkl', target = 'leafbasit.pkl', name_exp = 'leaf3s', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.01, 0.001], nu = 0.001, lam_var = 10., sig_var = 30.,
                   N=10, maxiter = 50.)
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf4', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 100, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 90.)
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf4b', 
       Dx = 0., Dy = 30., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 10, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf4c', 
       Dx = 0., Dy = -30., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 10, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)
#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf4d', 
       Dx = -20., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 10, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,#%%
           N=10, maxiter = 50.)

#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf5', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 100, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)

#%%
my_exp(source = 'basi2btemp.pkl', target = 'basi2target.pkl', name_exp = 'leafacro', 
       Dx = 0., Dy = 0., height_source = 90., flag = 'acro',
           height_target = 495., sig00 = 800, sig0 = 30, sig1 = 60, 
           coeffs =[0.01, 1000, 0.01], nu = 0.001, lam_var = 40., sig_var = 50.,
                   N=10, maxiter = 90)
#%%

my_exp(source = 'basi2btemp.pkl', target = 'basi2target.pkl', name_exp = 'leafacro2', 
       Dx = 0., Dy = 0., height_source = 90., flag = 'acro',
           height_target = 495., sig00 = 800, sig0 = 30, sig1 = 60, 
           coeffs =[0.01, 1, 0.01], nu = 0.001, lam_var = 40., sig_var = 50.,
                   N=10, maxiter = 90.)

#%%

my_exp(source = 'basi2btemp.pkl', target = 'basi2target.pkl', name_exp = 'leafacro3', 
       Dx = 0., Dy = 0., height_source = 90., flag = 'acro',
           height_target = 495., sig00 = 800, sig0 = 60, sig1 = 60, 
           coeffs =[0.01, 0.01, 10.], nu = 0.001, lam_var = 40., sig_var = 50.,
                   N=10, maxiter = 90.)
#%%

my_exp(source = 'diffuse.pkl', target = 'diffuset.pkl', name_exp = 'leafdiffuse', 
       Dx = 0., Dy = 0., height_source = 32., flag = 'diffuse',
           height_target = 136., sig00 = 300, sig0 = 15, sig1 = 15, 
           coeffs =[0.01, 0.01, 10.], nu = 0.001, lam_var = 40., sig_var = 25.,
                   N=10, maxiter = 100.)

#%%

my_exp(source = 'diffuse.pkl', target = 'diffuset.pkl', name_exp = 'leafdiffuse2', 
       Dx = 0., Dy = 0., height_source = 32., flag = 'diffuse',
           height_target = 136., sig00 = 300, sig0 = 15, sig1 = 15, 
           coeffs =[0.01, 1000, 0.01], nu = 0.001, lam_var = 40., sig_var = 25.,
                   N=10, maxiter = 100.)


#%%
my_exp(source = 'diffuse.pkl', target = 'diffuset.pkl', name_exp = 'leafdiffuse3', 
       Dx = 0., Dy = 0., height_source = 32., flag = 'diffuse',
           height_target = 136., sig00 = 300, sig0 = 15, sig1 = 15, 
           coeffs =[0.01, 1, 0.01], nu = 0.001, lam_var = 40., sig_var = 25.,
                   N=10, maxiter = 100.)

# %% Shooting from controls

Contlist = []
for i in range(len(Modules_list)):
    Contlist.append(Modules_list[i].Cont)

# %%
Mod_cont_init = Modules_list[0].copy_full()
Modlist_cont = shoot.shooting_from_cont_traj(Mod_cont_init, Contlist, 5)

# %% Visualisation
xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xs_i = Modlist_cont[2 * i].GD.GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')
