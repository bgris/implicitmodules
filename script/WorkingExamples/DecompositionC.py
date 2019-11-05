

import os.path

#path_res = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'results' + os.path.sep
#os.makedirs(path_res, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from implicitmodules.numpy.DeformationModules.Combination import CompoundModules
from implicitmodules.numpy.DeformationModules.ElasticOrder1 import ElasticOrder1
from implicitmodules.numpy.DeformationModules.SilentLandmark import SilentLandmark
import implicitmodules.numpy.Forward.Shooting as shoot
import implicitmodules.numpy.Utilities.Rotation as rot

from implicitmodules.numpy.Utilities.Visualisation import my_close, my_plot


#%%

xmin, xmax = -2, 2
ymin, ymax = -30, 30
nx, ny = 5, 60

X0 = np.linspace(xmin, xmax, nx)
Y0 = np.linspace(ymin, ymax, ny)
Z0 = np.meshgrid(X0, Y0)

Z = np.reshape(np.swapaxes(Z0, 0, 2), [-1, 2])

Z_c = np.concatenate([np.array([X0, np.zeros([nx]) + ymin]).transpose(),
                      np.array([np.zeros([ny]) + xmax, Y0]).transpose(),
                      np.array([np.flip(X0), np.zeros([nx]) + ymax]).transpose(),
                      np.array([np.zeros([ny]) + xmin, np.flip(Y0)]).transpose()])

plt.plot(Z[:, 0], Z[:, 1], '.')
plt.plot(Z_c[:, 0], Z_c[:, 1], '-')
plt.axis('equal')
plt.show()

####################################################################
# Modules
# ^^^^^^^

x1 = Z.copy()
xs = Z_c.copy()

####################################################################
# parameter for module of order 1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

th = 0. * np.pi
th = th * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

#%%
dimh = 2


eps = 0.1
C = np.zeros((x1.shape[0], 2, dimh)) + eps
w = 2 * np.pi / (ymax - ymin)

n0 = 1
C[:,1,0] = np.cos(n0 * w * x1[:,1]) * x1[:,0]

n1 = 2
C[:,1,1] = np.cos(n1 * w * x1[:,1]) * x1[:,0]


#%%

my_plot(x1, ellipse=C, angles=th)
#plt.plot(xs[:,0], xs[:,1], '-g')    


#%%
coeffs = [5., 0.05]
sig1 = 5
nu = 0.001
dim = 2

Sil = SilentLandmark(xs.shape[0], dim)
Model1 = ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)


# %%

Mod_el_init = CompoundModules([Sil, Model1])

# %%
ps = np.zeros(xs.shape)
#ps[nx + ny:2 * nx + ny, 1] = 0.5
ps[:10, :] = -5.
# 1ps[:,:,] = 1.
(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_sil = (xs, 1 * ps)
param_1 = ((x1, R), (p1, PR))

# %%

xfigmin = -10
xfigmax = 10
yfigmin = -30
yfigmax = 30

param = [param_sil, param_1]
# param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()
#%%
N=5
Modlist_shoot = shoot.shooting_traj(Mod_el, N)
#%%


i=N
plt.figure()
xs_i = Modlist_shoot[2 * i].GD.GD_list[0].GD
x1_i = Modlist_shoot[2 * i].GD.GD_list[1].GD[0]
plt.plot(x1_i[:, 0], x1_i[:, 1], '.b')
plt.plot(xs_i[:, 0], xs_i[:, 1], '-g', linewidth=2)
plt.axis('equal')
plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
plt.show()



#%%


N = 5
height = 55
nxgrid, nygrid = (41, 81)  # create a grid for visualisation purpose
u = height / 38.

(a, b, c, d) = (xfigmin, xfigmax, yfigmin, yfigmax)
[xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))
(nxgrid, nygrid) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid = SilentLandmark(grid_points.shape[0], dim)

param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)

Mod_tot = CompoundModules([Sil_grid, Mod_el])

# %%
Modlist_opti_tot = shoot.shooting_traj(Mod_tot, N)


#%%

i=N
plt.figure()
xgrid = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
xs_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].GD
x1_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[1].GD[0]
plt.plot(x1_i[:, 0], x1_i[:, 1], '.b')
plt.plot(xs_i[:, 0], xs_i[:, 1], '-g', linewidth=2)
plt.axis('equal')
plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
plt.show()

# %% Plot with grid
for i in range(N + 1):
    plt.figure()
    xgrid = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
    xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
    xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
    plt.plot(xsx, xsy, color='lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
    xs_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].GD
    x1_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[1].GD[0]
    plt.plot(x1_i[:, 0], x1_i[:, 1], '.b')
    plt.plot(xs_i[:, 0], xs_i[:, 1], '-g', linewidth=2)
    plt.axis('equal')
    plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
    plt.show()


























