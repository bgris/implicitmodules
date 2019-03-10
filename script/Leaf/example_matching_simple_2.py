import pickle

import matplotlib.pyplot as plt
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


# helper function
def my_plot(x, title="", col='*b'):
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], col)
    plt.title(title)
    plt.axis('equal')
    plt.show()

# %%
def my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf1',
           Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.1, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 100.):
    with open('./data/' + source, 'rb') as f:
        _, lx = pickle.load(f)
    
    nlx = np.asarray(lx).astype(np.float32)
    mean_nlx = np.mean(nlx[:, 1])
    (lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
    scale = height_source / (lmax - lmin)
    
    nlx[:, 1] = Dy-scale * (nlx[:, 1] - mean_nlx)
    nlx[:, 0] = Dx+scale * (nlx[:, 0] - np.mean(nlx[:, 0]))
    
    #  target
    with open('./data/' + target, 'rb') as f:
        _, lxt = pickle.load(f)
    
    nlxt = np.asarray(lxt).astype(np.float32)
    (lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
    scale = height_target / (lmax - lmin)
    
    nlxt[:, 1] = - scale * (nlxt[:, 1] - np.mean(nlxt[:, 1])) 
    nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0])) 
    
    xst = nlxt[nlxt[:, 2] == 2, 0:2]
    
    #  common options
    # nu = 0.001
    dim = 2
    
    #  Silent Module
    xs = nlx[nlx[:, 2] == 2, 0:2]
    xs = np.delete(xs, 3, axis=0)
    Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
    ps = np.zeros(xs.shape)
    param_sil = (xs, ps)
    
    my_plot(xs, "Silent Module", '*b')
    
    #  Modules of Order 0
    # sig0 = 15
    x0 = nlx[nlx[:, 2] == 1, 0:2]
    Model0 = defmod0.ElasticOrder0(sig0, x0.shape[0], dim, 0.1, nu)
    p0 = np.zeros(x0.shape)
    param_0 = (x0, p0)
    
    my_plot(x0, "Module order 0", 'or')
    
    #  Modules of Order 0
    # sig00 = 200
    x00 = np.array([[0., 0.]])
    Model00 = defmod0.ElasticOrder0(sig00, x00.shape[0], dim, 0.01, nu)
    p00 = np.zeros([1, 2])
    param_00 = (x00, p00)
    
    my_plot(x00, "Module order 00", '+r')
    
    #  Modules of Order 1
    # sig1 = 30
    x1 = nlx[nlx[:, 2] == 1, 0:2]
    C = np.zeros((x1.shape[0], 2, 1))
    K, L = 10, height_source
    a, b = -2 / L ** 3, 3 / L ** 2
    C[:, 1, 0] = (K * (a * (L/2 - x1[:, 1]-Dy) ** 3 
         + b * (L/2 - x1[:, 1]-Dy) ** 2))
    C[:, 0, 0] = 1. * C[:, 1, 0]
    Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, 0.01, C, nu)
    
    th = 0 * np.pi * np.ones(x1.shape[0])
    R = np.asarray([rot.my_R(cth) for cth in th])
    
    (p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
    param_1 = ((x1, R), (p1, PR))
    
    my_plot(x1, "Module order 1", 'og')
    
    #  Full model
    
    Module = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_00, param_0, param_1])
    P0 = opti.fill_Vector_from_GD(Module.GD)
    
    # 
    #lam_var = 40.
    #sig_var = 30.
    #N = 10
    args = (Module, xst, lam_var, sig_var, N, 1e-7)
    
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
                                      'maxfun': 200,
                                      'maxiter': maxiter,
                                      'iprint': 1,
                                      'maxls': 25
                                  })
    
    P1 = res['x']
    opti.fill_Mod_from_Vector(P1, Module)
    Module_optimized = Module.copy_full()
    Modules_list = shoot.shooting_traj(Module, N)
    
    #  Visualisation
    xst_c = my_close(xst)
    xs_c = my_close(xs)
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
    
    #  With grid
    hxgrid = 9
    hsl = 1.1*height_source/2
    a, b, c, d = (Dx-hsl/2, Dx+hsl/2, Dy-hsl, Dy+hsl) 
    hygrid = np.round(hxgrid*(d-c)/(b-a))
    nxgrid, nygrid = (2*hxgrid+1, 2*hygrid+1) # create a grid for visualisation purpose
    
    [xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))
    (nxgrid, nygrid) = xx.shape
    grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()
    
    Sil_grid = defmodsil.SilentLandmark(grid_points.shape[0], dim)
    
    param_grid = (grid_points, np.zeros(grid_points.shape))
    Sil_grid.GD.fill_cot_from_param(param_grid)
    
    Mod_tot = comb_mod.CompoundModules([Sil_grid, Module_optimized.copy_full()])
    
    # 
    Modlist_opti_tot_grid = shoot.shooting_traj(Mod_tot, N)
    #  Plot with grid
    xs_c = my_close(xs)
    xst_c = my_close(xst)
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
        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=2)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=1)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=2)
        plt.axis('equal')
        # plt.axis([-10,10,-10,55])
        
        plt.axis( [- 1.1*height_target/2, + 1.1*height_target/2, 
                   - 1.1*height_target/2, + 1.1*height_target/2])
        # plt.axis('off')
        #plt.show()
        plt.savefig('./results/' + name_exp + '_t_' + '{:02d}'.format(i) + '.png', 
                    format='png', bbox_inches='tight')
        
    filepng = './results/' + name_exp + '_t_' + "*.png"      
    os.system("convert " + filepng + " -set delay 0 -reverse " 
              + filepng + " -loop 0 " + './results/' + name_exp + ".gif")  

#%%
my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf1', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.1, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
                   N=10, maxiter = 50.)

my_exp(source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = 'leaf2', 
       Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.1, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.,
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
