# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:11:07 2019

@author: gris
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:26:14 2019

@author: gris
"""


import scipy .optimize
import numpy as np
import matplotlib.pyplot as plt

from implicitmodules.src import rotation as rot
import src.DeformationModules.SilentLandmark as defmodsil
import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.ElasticOrder1 as defmod1
import src.DeformationModules.ElasticOrder1C as defmod1C
import src.DeformationModules.Combination as comb_mod

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


#%% Initialize GD for module of order 1
th = 0.2*np.pi
th = th*np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for  i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

dim = 2
dimCont = 1
C = np.ones((x1.shape[0], dim, dimCont))


x00 = np.array([[0., 0.]])
coeffs = [1., 0.01]
sig00 = 10
sig1 = 2
nu = 0.001
dim = 2
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
Model1 = defmod1C.ElasticOrder1C(sig1, x1.shape[0], dim, coeffs[1], nu, dimCont)
Model00 = defmod0.ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
#%%

Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model1])

#%%
ps = np.zeros(xs.shape)
p00 = np.zeros([1, 2])
(p1, PR, pC) = (np.zeros(x1.shape), np.zeros(R.shape), np.zeros(C.shape))
ps = np.random.rand(*xs.shape)
p00 = np.random.rand(*[1, 2])
(p1, PR, pC) = (np.random.rand(*x1.shape), np.random.rand(*R.shape), np.random.rand(*C.shape))

param_sil = (xs, ps)
param_00 = (x00, p00)
param_1 = ((x1, R, C), (p1, PR, pC))
#%%
param = [param_sil, param_00, param_1]
#param = [param_sil, param_00, param_1]
GD = Mod_el_init.GD.copy()

#%%
Mod_el_init.GD.fill_cot_from_param(param)


#%%
Mod_el = Mod_el_init.copy_full()
#%%

Model1.GD.fill_cot_from_param(param_1)
Model1.update()
Model1.GeodesicControls_curr(Model1.GD)
Cont = Model1.Cont
Model1.Cost_curr()
Model12 = Model1.copy_full()
v1 = Model1.field_generator_curr()
co_init = Model12.p_Ximv_curr(v1,0)
der_cost = Model1.cot_to_innerprod_curr(Model12.GD,1).cotan
co_init = Model12.GD.inner_prod_v(v1)
#%%

der_cost_man = np.zeros(C.shape)
eps = 1e-5
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        for k  in range(C.shape[2]):
            Model1_cop = Model1.copy()
            C_cop = C.copy()
            C_cop[i,j,k] += eps
            param_cop = ((x1, R, C_cop), (p1, PR, pC))
            Model1_cop.GD.fill_cot_from_param(param_cop)
            Model1_cop.SKS = Model1.SKS
            #Model1_cop.GeodesicControls_curr(Model1.GD)
            Model1_cop.Cont = Cont
            Model1_cop.compute_mom_from_cont_curr()
            Model1_cop.Cost_curr()
            v1_cop = Model1_cop.field_generator_curr()
            co_init_cop = Model12.p_Ximv_curr(v1_cop,0)
            der_cost_man[i,j,k] = (co_init_cop - co_init)/eps
#%%
print(der_cost - der_cost_man)
#%%
def attach_fun(x,y):
    return np.sum( (x-y)**2 ), 2*(x-y)

N=10
#%%
flag_xs = np.ones(xs.shape).flatten()
flag_x00 = np.ones([1, 2]).flatten()

flag_ps = np.ones(xs.shape).flatten()
flag_p00 = np.ones([1, 2]).flatten()

(flag_x1, flag_xR, flag_xC) = (np.ones(x1.shape).flatten(), np.ones(R.shape).flatten(), np.ones(C.shape).flatten())
(flag_p1, flag_PR, flag_pC) = (np.ones(x1.shape).flatten(), np.ones(R.shape).flatten(), np.ones(C.shape).flatten())
flag_x1tot = np.concatenate((flag_x1, flag_xR, flag_xC))
flag_p1tot = np.concatenate((flag_p1, flag_PR, flag_pC))

flag = np.concatenate((flag_xs, flag_x1tot, flag_ps, flag_p1tot)) 
Mod_comb = comb_mod.CompoundModules([Sil, Model1])
args = (Mod_comb, target, 1., N, 0.001, flag, attach_fun)

ps = np.zeros(xs.shape)
p00 = np.zeros([1, 2])
(p1, PR, pC) = (np.zeros(x1.shape), np.zeros(R.shape), np.zeros(C.shape))
ps = 0.01*np.random.rand(*xs.shape)
p00 = 0.01*np.random.rand(*[1, 2])
(p1, PR, pC) = (0.01*np.random.rand(*x1.shape), 0.01*np.random.rand(*R.shape), 0.01*np.random.rand(*C.shape))

param_sil = (xs, ps)
param_00 = (x00, p00)
param_1 = ((x1, R, C), (p1, PR, pC))
param = [param_sil, param_1]
Mod_comb.GD.fill_cot_from_param(param)
P0 = opti.fill_Vector_from_GD(Mod_comb.GD)

co_init = opti.fun(P0, *args)
jac_init = opti.jac(P0, *args)
#der_cost = Model1.DerCost_curr().cotan[2]
#%%

der_cost_man = np.zeros(x1.shape)
eps = 1e-9
for i in range(5):
    for j in range(x1.shape[1]):
        #for k  in range(x1.shape[2]):
            x1_cop = x1.copy()
            x1_cop[i,j] += eps
            param_cop = ((x1_cop, R, C), (p1, PR, pC))
            param = [param_sil, param_cop]
            Mod_comb.GD.fill_cot_from_param(param)
            P0_cop = opti.fill_Vector_from_GD(Mod_comb.GD)
            co_init_cop = opti.fun(P0_cop, *args)
            der_cost_man[i,j] = (co_init_cop - co_init)/eps#%%
#%%
dim0 = flag_xs.shape[0]
print(der_cost_man[:5,:] + dx.GD_list[1].cotan[0][:5,:])
print(np.reshape(jac_init[dim0 :dim0 +5*2], [-1, 2]))
#print(jac_init[dim0 :dim0 +5*2])
#print(dx.GD_list[1].cotan[0][:5,:])
#%%
import src.Forward.Hamiltonianderivatives as HamDer  
Mod_comb.GD.fill_cot_from_param(param)
Mod_comb.update()
Mod_comb.GeodesicControls_curr(Mod_comb.GD)

dx = HamDer.dxH(Mod_comb)




#%%
der_cost_man = np.zeros(C.shape)
eps = 1e-5
for i in range(5):
    for j in range(C.shape[1]):
        for k  in range(C.shape[2]):
            Model1_cop = Model1.copy()
            C_cop = C.copy()
            C_cop[i,j,k] += eps
            param_cop = ((x1, R, C_cop), (p1, PR, pC))
            param = [param_sil, param_cop]
            Mod_comb.GD.fill_cot_from_param(param)
            P0_cop = opti.fill_Vector_from_GD(Mod_comb.GD)
            co_init_cop = opti.fun(P0_cop, *args)
            der_cost_man[i,j,k] = (co_init_cop - co_init)/eps
#%%
dim0 = flag_xs.shape[0]*2 + flag_xR.shape[0]
print(der_cost_man[:5,:,:])
print(jac_init[dim0 :dim0 +5*2])
print(der_cost_man[:5,:,:].reshape(-1)-jac_init[dim0 :dim0 +5*2])
#%%

#%%


der_cost_man = np.zeros(x1.shape)
eps = 1e-5
for i in range(5):
    for j in range(x1.shape[1]):
            Model1_cop = Model1.copy()
            p1_cop = p1.copy()
            p1_cop[i,j] += eps
            param_cop = ((x1, R, C), (p1_cop, PR, pC))
            param = [param_sil, param_cop]
            Mod_comb.GD.fill_cot_from_param(param)
            P0_cop = opti.fill_Vector_from_GD(Mod_comb.GD)
            co_init_cop = opti.fun(P0_cop, *args)
            der_cost_man[i,j] = (co_init_cop - co_init)/eps

#%%
dim0 = flag_xs.shape[0]*3 +  flag_xR.shape[0] + flag_xC.shape[0]
print(der_cost_man[:5,:])
print(jac_init[dim0 :dim0 +5*2])
print(der_cost_man[:5,:].reshape(-1)-jac_init[dim0 :dim0 +5*2])

#%%
Model1.GD.fill_cot_from_param(param_1)
Model1.update()
Model1.GeodesicControls_curr(Model1.GD)
Cont = Model1.Cont
Model1.Cost_curr()
co_init = Model1.cost
der_cost = Model1.DerCost_curr().cotan
#%%

der_cost_man = np.zeros(C.shape)
eps = 1e-8
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        for k  in range(C.shape[2]):
            Model1_cop = Model1.copy()
            C_cop = C.copy()
            C_cop[i,j,k] += eps
            param_cop = ((x1, R, C_cop), (p1, PR, pC))
            Model1_cop.GD.fill_cot_from_param(param_cop)
            Model1_cop.SKS = Model1.SKS
            #Model1_cop.GeodesicControls_curr(Model1.GD)
            Model1_cop.Cont = Cont
            Model1_cop.compute_mom_from_cont_curr()
            Model1_cop.Cost_curr()
            v1_cop = Model1_cop.field_generator_curr()
            #co_init_cop = Model1_cop.cost
            co_init_cop = Model12.GD.inner_prod_v(v1_cop)
            der_cost_man[i,j,k] = (co_init_cop - co_init)/eps
#%%
print(der_cost[2] - der_cost_man)

#%%
der_cost_man = np.zeros(R.shape[0])
eps = 1e-6
th_dir = np.random.rand(x1.shape[0])
for i in range(R.shape[0]):
    #for j in range(R.shape[1]):
    #    for k  in range(R.shape[2]):
            Model1_cop = Model1.copy()
            R_cop = R.copy()
            R_cop[i,:,:] += eps * rot.my_R(th_dir[i])
            param_cop = ((x1, R_cop, C), (p1, PR, pC))
            Model1_cop.GD.fill_cot_from_param(param_cop)
            Model1_cop.SKS = Model1.SKS
            #Model1_cop.GeodesicControls_curr(Model1.GD)
            Model1_cop.Cont = Cont
            Model1_cop.compute_mom_from_cont_curr()
            Model1_cop.Cost_curr()
            #co_init_cop = Model1_cop.cost
            v1_cop = Model1_cop.field_generator_curr()
            #co_init_cop = Model12.p_Ximv_curr(v1_cop,0)
            co_init_cop = Model12.GD.inner_prod_v(v1_cop)
            der_cost_man[i] = (co_init_cop - co_init)/eps
#%%
dir_R= np.asarray([rot.my_R(eps) for i in range(R.shape[0])])     
#%%       
der_cost_dir =  np.sum(np.sum(der_cost[1] * dir_R, axis=1), axis=1)

print(der_cost_dir - der_cost_man)


#%%
der_cost_man = np.zeros(x1.shape)
eps = 1e-6
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
            Model1_cop = Model1.copy()
            x1_cop = x1.copy()
            x1_cop[i,j] += eps
            param_cop = ((x1_cop, R, C), (p1, PR, pC))
            Model1_cop.GD.fill_cot_from_param(param_cop)
            #Model1_cop.SKS = Model1.SKS
            Model1_cop.update()
            #Model1_cop.GeodesicControls_curr(Model1.GD)
            Model1_cop.Cont = Cont
            Model1_cop.compute_mom_from_cont_curr()
            Model1_cop.Cost_curr()
            #co_init_cop = Model1_cop.cost
            v1_cop = Model1_cop.field_generator_curr()
            co_init_cop = Model12.GD.inner_prod_v(v1_cop)
            der_cost_man[i,j] = (co_init_cop - co_init)/eps
#%%
print(der_cost[0] - der_cost_man)
            
            
            
#%%
eps = 1e-2
Model1_cop = Model1.copy()
C_cop = C.copy()
C_cop[i,j,k] += eps
param_cop = ((x1, R, C_cop), (p1, PR, pC))
Model1_cop.GD.fill_cot_from_param(param_cop)
Model1_cop.SKS = Model1.SKS
#Model1_cop.GeodesicControls_curr(Model1.GD)
Model1_cop.Cont = Cont
Model1_cop.compute_mom_from_cont_curr()


v = Model1_cop.field_generator_curr()
inner = Model1.p_Ximv_curr(v,0)


v = Model1.field_generator_curr()
inner_init = Model1.p_Ximv_curr(v,0)

print(inner- inner_init)