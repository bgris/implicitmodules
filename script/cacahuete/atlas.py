
import pickle
import time

import torch

import matplotlib.pyplot as plt
import implicitmodules.torch as im
import implicitmodules.torch.Models.atlas as atlas
import implicitmodules.torch.DeformationModules.ConstrainedTranslations as constr_trans
import implicitmodules.torch.Attachment.attachement as attach

#%%
dty = torch.float32



with open('implicitmodules/data/nutsdata.pickle', 'rb') as f:
    lines, sigv, sig = pickle.load(f)

source = torch.tensor(lines[0][::2], requires_grad=True, dtype=dty)[1:]
targets = []
nb_pop = 10
for k in range(nb_pop):
    targets.append([torch.tensor(lines[1 + k][::2]  , requires_grad=True, dtype=dty)[1:]])

source0 = torch.tensor(lines[0][::2], requires_grad=True, dtype=dty)[1:]

pts_source = source.detach().numpy()


#%%
plt.figure()
plt.plot(source.detach().numpy()[:,0], source.detach().numpy()[:,1],'or')
for k in range(nb_pop):
    plt.plot(targets[k][0].detach().numpy()[:,0], targets[k][0].detach().numpy()[:,1],'xb')

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
scaling0 = constr_trans.ConstrainedTranslations(im.Manifolds.Landmarks(2, 1, gd = gd0.view(-1), cotan = cotan0.view(-1)), f, g, sigma_scaling)
scaling1 = constr_trans.ConstrainedTranslations(im.Manifolds.Landmarks(2, 1, gd = gd1.view(-1), cotan = cotan1.view(-1)), f, g, sigma_scaling)


sigma00 = 400.
nu00 = 0.001
coeff00 = 0.1
implicit00 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, 1, gd=torch.tensor([1., 1.], requires_grad=True)), sigma00, nu00, coeff00)



sigma0 = 0.2
nu0 = 0.001
coeff0 = 10.
implicit0 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, pts_source.shape[0], gd=torch.tensor(pts_source, requires_grad=True).view(-1)), sigma0, nu0, coeff0)


#%%
model = atlas.ModelCompoundWithPointsAtlas(
    [[source]],
    [scaling0, scaling1, implicit00, implicit0],
    True,
    [False, False, True, True],
    targets,
    [attach.VarifoldAttachement([1, 0.2])]
)

#%%
fitter = im.Models.ModelFittingScipy(model, 1., 10.)
#%%

costs = fitter.fit(targets, 200, options={})
#costs = model.fit(max_iter=2000, l=1., lr=0.01, log_interval=1)
##%%
model.compute(targets)
template = model.init_manifolds[0].gd[0].view(-1, 2)
for k in range(nb_pop):
    plt.figure(str(k))
    final = model.shot_manifold[k][0].gd.view(-1,2).detach().numpy()
    print(final)
    plt.plot(source0.detach().numpy()[:,0], source0.detach().numpy()[:,1], 'ob', label='template initial')
    plt.plot(template.detach().numpy()[:,0], template.detach().numpy()[:,1], 'og', label='template optimized')
    plt.plot(targets[k][0].detach().numpy()[:,0], targets[k][0].detach().numpy()[:,1], 'xr', label='target')
    plt.plot(final[:,0], final[:,1], '+k', label='final')
    plt.legend()
    plt.axis('equal')
    plt.axis([-2, 5,-2,5])
    names_exp = 'exp_nuts_atlas_' + str(k)
    #gd_init = np.array([[-1., 0.], [1., 0.]])
    #plt.plot(gd_init[:,0], gd_init[:,1], '*b', label='initialisation')
    plt.axis('equal')
    plt.axis([-3, 3,-3,3])
    plt.legend()
    path = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/cacahuete/'
    plt.savefig(path + names_exp)
#%%

#%%
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot

from implicitmodules.torch.DeformationModules import CompoundModule
#%%

compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifolds[1])
h = Hamiltonian(compound)
shoot(h, 10, "torch_euler")




