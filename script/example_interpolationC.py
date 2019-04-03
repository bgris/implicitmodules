import os.path
#path_res = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'Results' + os.path.sep
#os.makedirs(path_res, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from src.DeformationModules.Combination import CompoundModules
import src.Utilities.Rotation as rot
from src.DeformationModules.ElasticOrder1 import ElasticOrder1
from src.DeformationModules.SilentLandmark import SilentLandmark
import src.Forward.Shooting as shoot
#%%
dim = 2
Nb_layers = 7
Nb_pts_layer = 40

theta = np.linspace(0, 2 * np.pi, Nb_pts_layer +1)
theta = theta[:-1]
#%%

radius_ext = 30
radius_int = 25
step_layer = (radius_ext - radius_int) / (Nb_layers -1)
#%%
Points = []
for i in range(Nb_layers):
    r = radius_int + i*step_layer
    Points.append(np.array([r*np.cos(theta), r*np.sin(theta)]).transpose())

Points = np.concatenate(Points)

plt.plot(Points[:,0], Points[:,1], 'xr')
plt.axis('equal')
#%%

x1 = Points.copy()
N_x1 = Nb_layers * Nb_pts_layer
R = np.asarray([rot.my_R(0) for i in range(N_x1)])


for i in range(Nb_layers):
    for j in range(Nb_pts_layer):
        R[i * Nb_pts_layer + j] = rot.my_R(theta[j])
#%% Plot R
        

ax = plt.subplot(111, aspect='equal')

for i in range(Nb_layers):
    for j in range(Nb_pts_layer):
        indi = Nb_pts_layer * i + j
        ell = Ellipse(xy=Points[indi],
                  width=2, height=1,
                  angle=np.rad2deg(theta[j]))
        ax.add_artist(ell)
plt.axis([-40, 40, -40, 40])
plt.show()




#%% Dim Cont = 1
dim_h = 1
C = np.zeros((N_x1, dim, dim_h))

N_per = 4
 #scipy.interpolate.spline(xk, yk, xnew, order=3, kind='smoothest', conds=None)[source]
 #z = scipy.interpolate.CubicSpline(xk, yk)
for i in range(Nb_layers):
    for j in range(Nb_pts_layer):
        C[i * Nb_pts_layer + j, 0, 0] = 0.
        #print(np.mod(i,Nb_layers) - 0.5*(Nb_layers-1))
#        if i>0.5*(Nb_layers-1):
#            C[i * Nb_pts_layer + j, 1, 0] = 1.
#        else:
#                C[i * Nb_pts_layer + j, 1, 0] = 0.
        C[i * Nb_pts_layer + j, 1, 0] = np.cos(N_per * theta[j]) *  (np.mod(i,Nb_layers) - 0.5*(Nb_layers-1))
        
        

#%% Dim Cont = 2
dim_h = 2
C = np.zeros((N_x1, dim, dim_h))

N_per0 = 4
N_per1 = 8
 #scipy.interpolate.spline(xk, yk, xnew, order=3, kind='smoothest', conds=None)[source]
 #z = scipy.interpolate.CubicSpline(xk, yk)
for i in range(Nb_layers):
    for j in range(Nb_pts_layer):
        C[i * Nb_pts_layer + j, 0, 0] = 0.
        #print(np.mod(i,Nb_layers) - 0.5*(Nb_layers-1))
#        if i>0.5*(Nb_layers-1):
#            C[i * Nb_pts_layer + j, 1, 0] = 1.
#        else:
#                C[i * Nb_pts_layer + j, 1, 0] = 0.
        C[i * Nb_pts_layer + j, 1, 0] = np.cos(N_per0 * theta[j]) *  (np.mod(i,Nb_layers) - 0.5*(Nb_layers-1))
        
        
        C[i * Nb_pts_layer + j, 0, 1] = 0.
        C[i * Nb_pts_layer + j, 1, 1] = np.cos(N_per1 * theta[j]) *  (np.mod(i,Nb_layers) - 0.5*(Nb_layers-1))
        
        
        
#%% Plot C
epsix = 0.1 
epsiy = 0.001 
ax = plt.subplot(111, aspect='equal')

for i in range(Nb_layers):
    for j in range(Nb_pts_layer):
        indi = Nb_pts_layer * i + j
        C_i = C[indi, : , 0]
        ell = Ellipse(xy=Points[indi],
                  width=C_i[0] + epsix, height=C_i[1] + epsiy,
                  angle=np.rad2deg(theta[j]))
        ax.add_artist(ell)
plt.axis([-40, 40, -40, 40])
plt.show()
        

# %%
coeffs = [1.]
sig1 = 10.
nu = 0.001
dim = 2

#Sil = SilentLandmark(xs.shape[0], dim)
Model1 = ElasticOrder1(sig1, x1.shape[0], dim, coeffs[0], C, nu)

(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
p1[:11] += 200.
param_1 = ((x1, R), (p1, PR))
Model1.GD.fill_cot_from_param(param_1)
#%%
N=10
Modlist_opti_tot = shoot.shooting_traj(Model1, N)

#%%%
i=N
plt.figure()
ax = plt.subplot(111, aspect='equal')
x1_i = Modlist_opti_tot[2 * i].GD.GD[0]
R_i =  Modlist_opti_tot[2 * i].GD.GD[1]
for k in range(Nb_layers):
    for j in range(Nb_pts_layer):
        indi = Nb_pts_layer * k + j
        angle_kj = np.arctan(R_i[indi][1][0]/R_i[indi][0][0])
        C_kj = C[indi, : , 0] 
        ell = Ellipse(xy=x1_i[indi],
                  width=C_kj[0] + epsix, height=C_kj[1] + epsiy,
                  angle=np.rad2deg(angle_kj))
        ax.add_artist(ell)

plt.axis([-40, 40, -40, 40])
plt.show()




#%%

for i in range(N+1):
    plt.figure()
    ax = plt.subplot(111, aspect='equal')
    x1_i = Modlist_opti_tot[2 * i].GD.GD[0]
    R_i =  Modlist_opti_tot[2 * i].GD.GD[1]
    for k in range(Nb_layers):
        for j in range(Nb_pts_layer):
            indi = Nb_pts_layer * k + j
            angle_kj = np.arctan(R_i[indi][1][0]/R_i[indi][0][0])
            C_kj = C[indi] 
            ell = Ellipse(xy=x1_i[indi],
                      width=C_kj[0] + epsix, height=C_kj[1] + epsiy,
                      angle=np.rad2deg(angle_kj))
            ax.add_artist(ell)
    
    plt.axis([-40, 40, -40, 40])
    plt.show()

























      
        
        
        