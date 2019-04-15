# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:20:37 2019

@author: trouve
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np


import src.DeformationModules.Combination as comb_mod
import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.ElasticOrder1 as defmod1
import src.DeformationModules.SilentLandmark as defmodsil
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
    x0 = nlx[nlx[:, 2] == 1, 0:2]
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
    plt.show()
    plt.savefig(dir_res + name_exp + '_modules_' + flag + '.png', 
                    format='png', bbox_inches='tight')
    #my_plot(x1, "Module order 1", 'og')
    
    
    #  Full model
    
    Module = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_00, param_0, param_1])
    P0 = opti.fill_Vector_from_GD(Module.GD)
    
    outvar = (Module, xs, xst, opti.fun, P0, opti.jac)
  
    return outvar
                                                     