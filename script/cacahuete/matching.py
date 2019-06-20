
import pickle
import time

import torch

import matplotlib.pyplot as plt
import implicitmodules.torch as im
import implicitmodules.torch.DeformationModules.ConstrainedTranslations as constr_trans
import implicitmodules.torch.Attachment.attachement as attach

#%%
dty = torch.float32
source = torch.tensor([[-1., -1.], [1., 1.], [1., -1.], [-1., 1.]], requires_grad=True, dtype=dty)
#target = torch.tensor([[0., 0.], [2., 0.], [1., -1.], [1., 1.]], requires_grad=True, dtype=dty)

target = torch.tensor([[-2., -2.], [2., 2.], [2., -2.], [-2., 2.]], requires_grad=True, dtype=dty)
#target = 1.+ torch.tensor([[-1., -1.], [1., 1.], [1., -1.], [-1., 1.]], requires_grad=True, dtype=dty)

#%%


with open('implicitmodules/data/nutsdata.pickle', 'rb') as f:
    lines, sigv, sig = pickle.load(f)
source = torch.tensor(lines[0][::2], requires_grad=True, dtype=dty)[1:]
target = torch.tensor(lines[1][::2]  , requires_grad=True, dtype=dty)[1:]



pts_source = source.detach().numpy()
#%%


sigma_scaling = 1.
a = torch.sqrt(torch.tensor(3.))
direc_scaling_pts = torch.tensor([[1., 0.], [-0.5 , 0.5* a],  [-0.5, -0.5* a]], requires_grad=True, dtype=dty)
direc_scaling_vec =  torch.tensor([[1., 0.], [-0.5 , 0.5* a],  [-0.5, -0.5* a]], requires_grad=True, dtype=dty)
def f(x):
    centre = x.view(1,2).repeat(3,1)
    return centre + 0.3 * sigma_scaling * direc_scaling_pts

def g(x):
    return direc_scaling_vec

#%%
#gd0 = torch.tensor([[-1., 0.6]], requires_grad=True, dtype=dty)
gd0 = torch.tensor([[-1., 0.]], requires_grad=True, dtype=dty)
cotan0 = torch.tensor([[0., 0.]], requires_grad=True, dtype=dty)
#gd1 = torch.tensor([[0.8, 0.6]], requires_grad=True, dtype=dty)
gd1 = torch.tensor([[1., 0.]], requires_grad=True, dtype=dty)
cotan1 = torch.tensor([[0., 0.]], requires_grad=True, dtype=dty)
#%%
pts = f(gd0).detach().numpy()
vec = g(gd0).detach().numpy()
pts1 = f(gd1).detach().numpy()
vec1 = g(gd1).detach().numpy()


plt.quiver(pts[:,0], pts[:,1], vec[:,0], vec[:,1])
plt.quiver(pts1[:,0], pts1[:,1], vec1[:,0], vec1[:,1])
plt.plot(source.detach().numpy()[:,0], source.detach().numpy()[:,1], 'b')
plt.plot(target.detach().numpy()[:,0], target.detach().numpy()[:,1], 'r')
plt.axis('equal')
#%%
scaling0 = constr_trans.ConstrainedTranslations(im.Manifolds.Landmarks(2, 1, gd = gd0.view(-1), cotan = cotan0.view(-1)), f, g, sigma_scaling)
scaling1 = constr_trans.ConstrainedTranslations(im.Manifolds.Landmarks(2, 1, gd = gd1.view(-1), cotan = cotan1.view(-1)), f, g, sigma_scaling)


sigma00 = 400.
nu00 = 0.001
coeff00 = 10.
implicit00 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, 1, gd=torch.tensor([0., 0.], requires_grad=True)), sigma00, nu00, coeff00)


sigma0 = 0.2
nu0 = 0.001
coeff0 = 10.
implicit0 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, pts_source.shape[0], gd=torch.tensor(pts_source, requires_grad=True).view(-1)), sigma0, nu0, coeff0)


sigma1 = 0.2
nu1 = 0.001
coeff1 = 10.
implicit1 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, pts_source.shape[0], gd=torch.tensor(pts_source, requires_grad=True).view(-1)), sigma1, nu1, coeff1)



#%%
model_param = im.Models.ModelCompoundWithPointsRegistration(
    [[source, torch.ones(source.shape[0], requires_grad=True)]],
    [scaling0, scaling1, implicit00, implicit0],
    [False, False, True, True],
    [attach.VarifoldAttachement([1, 0.2])]
)


model_lddmm = im.Models.ModelCompoundWithPointsRegistration(
    [[source, torch.ones(source.shape[0], requires_grad=True)]],
    [implicit00, implicit1],
    [True, True],
    [attach.VarifoldAttachement([1, 0.2])]
)

#%%
model = model_lddmm
costs = model.fit([(target, torch.ones(target.shape[0], requires_grad=True))], max_iter=2000, l=100., lr=0.0001, log_interval=1)

#%%
model.compute([(target, torch.ones(target.shape[0], requires_grad=True))])
import numpy as np
grid_origin, grid_size, grid_resolution = [-2., -1.], [4., 2.], [40, 20]
def_grids = model.compute_deformation_grid(grid_origin, grid_size, grid_resolution, it=10)

final = model.shot_manifold[0].gd.view(-1,2).detach().numpy()
ax_c = plt.subplot(111, aspect='equal')
im.Utilities.usefulfunctions.plot_grid(ax_c, def_grids[-1][0].numpy(), def_grids[-1][1].numpy(), color='k')

plt.plot(source.detach().numpy()[:,0], source.detach().numpy()[:,1], 'b', label='source')
plt.plot(target.detach().numpy()[:,0], target.detach().numpy()[:,1], 'r', label='target')
plt.plot(final[:,0], final[:,1], 'G', label='final')
#GD0 = model.init_manifold[1].gd.detach().numpy()
#GD1 = model.init_manifold[2].gd.detach().numpy()
#GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
#plt.plot(GD_opt[:,0], GD_opt[:,1], '+b', label='optimized')
#names_exp = 'exp0'
#gd_init = np.array([[-1., 0.6], [0.8, 0.6]])
names_exp = 'exp_LDDMM_grande_translation'
#gd_init = np.array([[-1., 0.], [1., 0.]])
#plt.plot(gd_init[:,0], gd_init[:,1], '*b', label='initialisation')
plt.axis('equal')
plt.axis([-3, 3,-3,3])
plt.legend()
path = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/cacahuete/'
plt.savefig(path + names_exp)
#%% 
grid_origin, grid_size, grid_resolution = [-2., -1.], [4., 2.], [20, 10]
def_grids = model.compute_deformation_grid(grid_origin, grid_size, grid_resolution, it=10)

ax_c = plt.subplot(111, aspect='equal')
plt.axis('equal')
plt.title('Source')
plt.xlabel('x')
plt.ylabel('y')
im.Utilities.usefulfunctions.plot_grid(ax_c, def_grids[-1][0].numpy(), def_grids[-1][1].numpy(), color='k')

plt.plot(source.detach().numpy()[:,0], source.detach().numpy()[:,1], 'b', label='source')
plt.plot(target.detach().numpy()[:,0], target.detach().numpy()[:,1], 'r', label='target')
plt.plot(final[:,0], final[:,1], 'k', label='final')
GD0 = model.init_manifold[1].gd.detach().numpy()
GD1 = model.init_manifold[2].gd.detach().numpy()
GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
plt.plot(GD_opt[:,0], GD_opt[:,1], '+b', label='optimized')


#%%
source_shot = model.compute([(target, torch.ones(target.shape[0], dtype=dty))])



#%%
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot

from implicitmodules.torch.DeformationModules import CompoundModule
#%%

compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
shoot(h, 10, "torch_euler")
attach_fun = attach.VarifoldAttachement([1])
#att = attach_fun([source], [target])
att = attach_fun([compound.manifold.gd[0].view(-1, 2)], [target])

torch.autograd.grad(att, model.init_manifold.gd, allow_unused=True)
#%%

sigma00 = 400.
nu00 = 0.001
coeff00 = 0.01
implicit00 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, 1, gd=torch.tensor([0., 0.], requires_grad=True)), sigma00, nu00, coeff00)



#%%
#############################################################################
# Model fit
# ^^^^^^^^^^
#
# Setting up the model and start the fitting loop
#
#%%
#att = attach.PointwiseDistanceAttachement()((model.shot_manifold.gd[0], model.alpha), (target.view(-1),  model.alpha))
#att = torch.sum((model.shot_manifold.gd[0] - target.view(-1))**2)
model.compute([target])
attach_fun = attach.VarifoldAttachement([1])
#att = attach_fun([source], [target])
att = attach_fun([model.shot_manifold.gd[0].view(-1, 2)], [target])

torch.autograd.grad(attach_fun([model.shot_manifold.gd[0].view(-1, 2)], [target]), model.init_manifold.cotan, allow_unused=True)

#%%
compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
shoot(h, 10, "torch_euler")
self.shot_manifold = compound.manifold.copy()
self.deformation_cost = compound.cost()










