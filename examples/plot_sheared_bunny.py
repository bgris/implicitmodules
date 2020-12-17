"""
Sheared bunny
=============

3D meshes registration using implicit modules of order 1. Normal frames and growth factor are optimized.
"""

###############################################################################
# Important relevant Python modules.
#

import sys
sys.path.append("../")

import torch
import meshio

import imodal

torch.set_default_dtype(torch.float64)
imodal.Utilities.set_compute_backend('keops')
# device = 'cuda'
device = 'cpu'

###############################################################################
# Load source and target data.
#

data_folder = "data/"
source_mesh = meshio.read(data_folder+"bunny_source.ply")
target_mesh = meshio.read(data_folder+"bunny_shear_ear.ply")

source_points = torch.tensor(source_mesh.points, dtype=torch.get_default_dtype())
target_points = torch.tensor(target_mesh.points, dtype=torch.get_default_dtype())
source_triangles = torch.tensor(source_mesh.cells_dict['triangle'], dtype=torch.long)
target_triangles = torch.tensor(target_mesh.cells_dict['triangle'], dtype=torch.long)


###############################################################################
# Rescaling source and target.
#

scale_factor = 100.
source_points = scale_factor*(source_points - torch.mean(source_points, dim=0))
target_points = scale_factor*(target_points - torch.mean(target_points, dim=0))


###############################################################################
# Generation of implicit module of order 1: points positions, initial growth
# factor and normal frames.
#

# Defining an AABB around the source
aabb_source = imodal.Utilities.AABB.build_from_points(1.8*source_points)

# Generation of growth points
implicit1_density = 0.1
implicit1_points = imodal.Utilities.fill_area_uniform_density(imodal.Utilities.area_convex_hull, aabb_source, implicit1_density, scatter=1.8*source_points)

# Placeholders for growth factor and normal frames
implicit1_r = torch.empty(implicit1_points.shape[0], 3, 3)
implicit1_c = torch.empty(implicit1_points.shape[0], 3, 1)

# Initial growth factor constants
growth_constants = torch.tensor([[[1.], [1.], [1.]]], requires_grad=True, device=device)

# Initial normal frames angles. Normal frames are rotation matrices and thus defined by 3 angles.
angles = torch.zeros(implicit1_points.shape[0], 3, requires_grad=True, device=device)


###############################################################################
# Create the deformation model with a combination of 3 modules : implicit module
# of order 1 (growth model), implicit module of order 0 (small corrections), global translation
# and a large scale rotation.
#


###############################################################################
# Create and initialize the global translation module.
#

global_translation = imodal.DeformationModules.GlobalTranslation(3, coeff=10.)


###############################################################################
# Create and initialize the growth module.
#

sigma1 = 2./implicit1_density**(1/3)

implicit1 = imodal.DeformationModules.ImplicitModule1(3, implicit1_points.shape[0], sigma1, implicit1_c, nu=1000., gd=(implicit1_points, implicit1_r), coeff=0.001)


###############################################################################
# Create and initialize the local translations module.
#

sigma0 = 3./implicit0_density**(1/3)

implicit0 = imodal.DeformationModules.ImplicitModule0(3, implicit0_points.shape[0], sigma0, nu=1., gd=implicit0_points, coeff=100.)


###############################################################################
# Create and initialize the local large scale rotation.
#

rotation = imodal.DeformationModules.LocalRotation(3, 30., gd=torch.tensor([[0., 0., 1.]], device=device, requires_grad=True), backend='torch', coeff=10.)


###############################################################################
# Define our growth factor model.
#

# Function that computes normal frames from angles.
def compute_basis(angles):
    rot_x = imodal.Utilities.rot3d_x_vec(angles[:, 0])
    rot_y = imodal.Utilities.rot3d_y_vec(angles[:, 1])
    rot_z = imodal.Utilities.rot3d_z_vec(angles[:, 2])
    return torch.einsum('nik, nkl, nlj->nij', rot_z, rot_y, rot_x)


# Function that computes growth factor from growth factor constants.
def compute_growth(growth_constants):
    return growth_constants.repeat(implicit1_points.shape[0], 1, 1)


# Callback used by the registration model to compute the new growth factor
# and normal frames.
def precompute(init_manifold, modules, parameters):
    init_manifold[1].gd = (init_manifold[1].gd[0], compute_basis(parameters['growth']['params'][0]))
    modules[1].C = compute_growth(parameters['growth']['params'][1])


###############################################################################
# Move the deformation modules on the right device (e.g. GPU) if necessary.
#

deformable_source.silent_module.to_(device)
deformable_target.silent_module.to_(device)
global_translation.to_(device)
implicit0.to_(device)
implicit1.to_(device)
rotation.to_(device)
if str(device) is not 'cpu':
    implicit0._ImplicitModule0_KeOps__keops_backend = 'GPU'
    implicit1._ImplicitModule1_KeOps__keops_backend = 'GPU'

###############################################################################
# Define deformables used by the registration model.
#

deformable_source = imodal.Models.DeformableMesh(source_points, source_triangles.to(device))
deformable_target = imodal.Models.DeformableMesh(target_points, target_triangles.to(device))

###############################################################################
# Define the registration model.
#

sigmas_varifold = [1., 5., 15.]
attachment = imodal.Attachment.VarifoldAttachment(3, sigmas_varifold)

model = imodal.Models.RegistrationModel(deformable_source, [implicit1, implicit0, global_translation, rotation], [attachment], fit_gd=None, lam=100., precompute_callback=precompute, other_parameters={'growth': {'params': [angles, growth_constants]}})


###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = 'euler'
shoot_it = 10
costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')

fitter.fit(deformable_target, 1000, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe', 'history_size': 250})


###############################################################################
# Compute optimized deformation trajectory.
#

import time
intermediates = {}
start = time.perf_counter()
with torch.autograd.no_grad():
    deformed = model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)[0][0].detach()
print("Elapsed={elapsed}".format(elapsed=time.perf_counter()-start))

basis = compute_basis(angles.detach()).cpu()
C = compute_growth(growth_constants.detach()).cpu()


