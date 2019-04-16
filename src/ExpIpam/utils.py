# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:15:43 2019

@author: trouve
"""

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


def exp_ipam_init(flag = 'basi', 
           source = 'basi1b.pkl', target = 'basi1t.pkl', name_exp = None, 
           dir_res = '/home/trouve/Dropbox/Talks/Pics/ResImp/ResIpam/',  
           Dx = 0., Dy = 0., height_source = 38.,
           height_target = 100., sig00 = 200, sig0 = 15, sig1 = 30, 
           coeffs =[0.01, 0.1, 0.01], nu = 0.001, lam_var = 40., sig_var = 30.):

    with open('./data/' + source, 'rb') as f:
        _, lx = pickle.load(f)
    
    nlx = np.asarray(lx).astype(np.float32)
    (lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
    scale = height_source / (lmax - lmin)
    
    nlx[:, 1] = Dy-scale * (nlx[:, 1] - lmax)
    nlx[:, 0] = Dx+scale * (nlx[:, 0] - np.mean(nlx[:, 0]))
    
    #  target
    with open('./data/' + target, 'rb') as f:
        _, lxt = pickle.load(f)
    
    nlxt = np.asarray(lxt).astype(np.float32)
    (lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
    scale = height_target / (lmax - lmin)
    
    nlxt[:, 1] = - scale * (nlxt[:, 1] - lmax) 
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
    
    #  Modules of Order 0
    # sig00 = 200
    x00 = np.array([[0., Dy + height_source/2]])
    Model00 = defmod0.ElasticOrder0(sig00, x00.shape[0], dim, coeffs[0], nu)
    p00 = np.zeros([1, 2])
    param_00 = (x00, p00)
    
    
    #my_plot(x00, "Module order 00", '+r')
    
    #  Modules of Order 0
    # sig0 = 15

    x0 = np.concatenate((nlx[nlx[:, 2] == 1, 0:2], xs), axis = 0)
    Model0 = defmod0.ElasticOrder0(sig0, x0.shape[0], dim, coeffs[1], nu)
    p0 = np.zeros(x0.shape)
    param_0 = (x0, p0)
    
    
    
    #my_plot(x0, "Module order 0", 'or')
    
    #  Modules of Order 1
    # sig1 = 30
    
    if (flag == 'basi'):
        x1 = nlx[nlx[:, 2] == 1, 0:2]
        C = np.zeros((x1.shape[0], 2, 1))
        K, L = 10, height_source
        a, b = -2 / L ** 3, 3 / L ** 2
        C[:, 1, 0] = (K * (a * (L - x1[:, 1]+Dy) ** 3 
             + b * (L - x1[:, 1]+Dy) ** 2))
        C[:, 0, 0] = 1. * C[:, 1, 0]
    elif (flag == 'acro'):
        x1 = nlx[nlx[:, 2] == 1, 0:2]
        C = np.zeros((x1.shape[0], 2, 1))
        K, L = 10, height_source
        a,b = 1/L, 3.
        z = a*(x1[:,1]-Dy)
        C[:,1,0] = K*((1-b)*z**2+b*z)
        C[:,0,0] = 0.8*C[:,1,0]
    elif (flag == 'diffuse'):
        x1 = nlx[nlx[:, 2] == 1, 0:2]
        C = np.zeros((x1.shape[0], 2, 1))
        K = 10
        C[:,1,0] = K
        C[:,0,0] = K
        
    
    Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[2], C, nu)
    
    th = 0 * np.pi * np.ones(x1.shape[0])
    R = np.asarray([rot.my_R(cth) for cth in th])
    
    (p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
    param_1 = ((x1, R), (p1, PR))
    
    plt.figure()
    xs_c = my_close(xs)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-g', linewidth=2)
    plt.plot(xs[:, 0], xs[:, 1], 'og', linewidth=2)
    plt.plot(x00[:, 0], x00[:, 1], 'ok', linewidth=2)
    
    plt.plot(x1[:, 0], x1[:, 1], 'ob', linewidth=2, markersize = 8)
    plt.plot(x0[:, 0], x0[:, 1], 'or', linewidth=2, markersize = 3)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(dir_res + name_exp + '_modules_' + flag + '.png', 
                    format='png', bbox_inches='tight')
    plt.show()
    
    #my_plot(x1, "Module order 1", 'og')
    
    
    #  Full model
    
    Module = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_00, param_0, param_1])
    P0 = opti.fill_Vector_from_GD(Module.GD)
    
    outvar = (Module, xs, xst, opti.fun, opti.jac)
  
    return (P0, outvar)
    

def exp_ipam_optim(invar, attach_var, maxiter = 100, maxfun = 200, N=10):
    (P0, (Module, xs, xst, fun, jac)) = invar
    (lam_var, sig_var) = attach_var
    args = (Module, xst, lam_var, sig_var, N, 1e-7)
    
    res = scipy.optimize.minimize(fun, P0,
                                  args=args,
                                  method='L-BFGS-B',
                                  jac= jac,
                                  bounds=None,
                                  tol=None,
                                  callback=None,
                                  options={
                                      'maxcor': 10,
                                      'ftol': 1.e-09,
                                      'gtol': 1e-03,
                                      'eps': 1e-08,
                                      'maxfun': maxfun,
                                      'maxiter': maxiter,
                                      'iprint': 1,
                                      'maxls': 25
                                  })
                                  
    P1 = res['x']
    return P1    
    
def exp_ipam_plot(invar, outvar):

    (N, flag, name_exp, height_source, height_target, Dx, Dy) = invar
    (Module, P1, xs, xst, dir_res, dir_Mysh) = outvar
    dim = 2

    opti.fill_Mod_from_Vector(P1, Module)
    Module_optimized = Module.copy_full()
    Modules_list = shoot.shooting_traj(Module, N)
    
    # Show data
    
    plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    plt.figure()
    plt.subplot(gs[0])
    if (flag == 'basi'):
        img = mpimg.imread('./data/Model1.png')
    elif (flag == 'acro'):
        img = mpimg.imread('./data/Model2.png')
    elif (flag == 'diffuse'):
        img = mpimg.imread('./data/Model3.png')
        
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(gs[1])
    z = np.linspace(0,1,100)
    if (flag == 'basi'):
        plt.plot(-2*(1-z)**3+3*(1-z)**2,z)
        axes = plt.gca()
        axes.set_xlim([0.,1.])
        axes.set_ylim([0.,1.])
        plt.axis('tight')
    elif (flag == 'acro'):
        img = mpimg.imread('./data/Model2.png')
        plt.plot((1-3)*z**2+3*z,z)
        plt.axis('tight')
    elif (flag == 'diffuse'):
        img = mpimg.imread('./data/Model3.png')
        plt.plot(np.ones(z.shape),z)
        plt.axis('tight')
    plt.tight_layout()
    plt.show()
    plt.savefig(dir_res + name_exp + '_model_' + flag + '.png', 
                    format='png', bbox_inches='tight')
    
    #  Visualisation
    xst_c = my_close(xst)
    xs_c = my_close(xs)
    for i in range(N + 1):
        plt.figure()
        xs_i = Modules_list[2 * i].GD.GD_list[0].GD
        xs_ic = my_close(xs_i)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
        
        x00_i = Modules_list[2 * i].GD.GD_list[2].GD
        plt.plot(x00_i[:, 0], x00_i[:, 1], 'ok', linewidth=2)
        
        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=2)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=2)
        
        x0_i = Modules_list[2 * i].GD.GD_list[1].GD
        plt.plot(x0_i[:, 0], x0_i[:, 1], 'or', linewidth=2, markersize = 3)
        plt.axis('equal')
        plt.show()
    
    #  With grid
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
    
    Mod_tot = comb_mod.CompoundModules([Sil_grid, Module_optimized.copy_full()])
    
    # 
    N = 15
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
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=2)
        plt.axis('equal')
        # plt.axis([-10,10,-10,55])
        
        plt.axis( [- 1.1*height_target/2, + 1.1*height_target/2, 
                   - 0.05*height_target, + 1.05*height_target])
        # plt.axis('off')
        #plt.show()
        plt.savefig(dir_res + name_exp + '_t_' + '{:02d}'.format(i) + '.png', 
                    format='png', bbox_inches='tight',dpi=300)
        
    filepng = dir_res + name_exp + '_t_' + "*.png"    
    filegif = dir_res + name_exp + ".gif"
    filegifr = dir_res + name_exp + "r.gif"
     
    #os.system("convert " + filepng + " -coalesce -duplicate 1,-2--1" +
    #          "-quiet -layers OptimizePlus  -loop 0 " + filegif) 

    os.system("convert " + filepng + " -reverse -set delay 0 " + filegifr)  
    os.system("convert " + filepng + " -set delay 0 " 
               + filegifr + "  -loop 0 " + filegif)  
    
    # create the mysh file for inclusion in the beamer latex
    filemysh = dir_Mysh + name_exp + ".mysh"
    file = open(filemysh,"w")
    file.write("#!/bin/bash -x\n\n")
    file.write("gwenview " + filegif + " &")
    file.close()  
