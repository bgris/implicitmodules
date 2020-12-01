"""
Acropetal growth model of a leaf
================================

In this example we will model acropetal growth of a leaf.

Acropetal growth is characterised by a development of the leaf from the basis towards the apex.

For this, we use an implicit module of order 1 with growth constants following
some quadratic law. We also add (heavily penalised) simple local translations to the border of the leaf to accelerate convergence.

"""

###############################################################################
# Module import.
#

import sys
sys.path.append("../../")

import pickle
import copy

import torch
import matplotlib.pyplot as plt

import imodal

torch.set_default_dtype(torch.float64)

###############################################################################
# We load the data, rescale it and zero it.
#

data = pickle.load(open("../../data/data_acropetal.pkl", 'rb'))

shape_source = imodal.Utilities.close_shape(torch.tensor(data['source_silent']).type(torch.get_default_dtype()))
shape_target = imodal.Utilities.close_shape(torch.tensor(data['target_silent']).type(torch.get_default_dtype()))

# Some rescaling for the source
height_source = 90.
height_target = 495.

smin, smax = torch.min(shape_source[:, 1]), torch.max(shape_source[:, 1])
sscale = height_source / (smax - smin)
shape_source[:, 0] = sscale * (shape_source[:, 0] - torch.mean(shape_source[:, 0]))
shape_source[:, 1] = - sscale * (shape_source[:, 1] - smax)

# Some rescaling for the target
tmin, tmax = torch.min(shape_target[:, 1]), torch.max(shape_target[:, 1])
tscale = height_target / (tmax - tmin)
shape_target[:, 0] = tscale * (shape_target[:, 0] - torch.mean(shape_target[:, 0]))
shape_target[:, 1] = - tscale * (shape_target[:, 1] - tmax)


###############################################################################
# We now sample the points that will be used by the deformation modules.
# We simply map the inner area of the source shape with uniformly spaced points.
#

# Points for our contour
points_small = shape_source.clone()

# Build AABB around the source shape and uniformly sample points for the implicit
# module of order 1
growth_density = 0.1
aabb_source = imodal.Utilities.AABB.build_from_points(shape_source)
points_growth = imodal.Utilities.fill_area_uniform_density(imodal.Utilities.area_shape, aabb_source, growth_density, shape=shape_source)

rot_growth = torch.stack([imodal.Utilities.rot2d(0.)]*points_growth.shape[0], axis=0)

###############################################################################
# Plot everything.
#

plt.subplot(1, 2, 1)
plt.title("Source leaf")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(points_small[:, 0].numpy(), points_small[:, 1].numpy(), 'x', color='red')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), 'o', color='blue')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.title("Target leaf")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='black')
plt.axis('equal')

plt.show()


###############################################################################
# Define and plot the quadratic model of the growth constants.
# TODO: d'où vient ce modèle ?
#
C = torch.zeros(points_growth.shape[0], 2, 1)
K, L = 10, height_source
a, b = 1./L, 3.
z = a*points_growth[:, 1]
C[:, 1, 0] = K * ((1 - b) * z**2 + b * z)
C[:, 0, 0] = 0.8 * C[:, 1, 0]

ax = plt.subplot()
imodal.Utilities.plot_C_ellipses(ax, points_growth, C, color='green', scale=0.2)
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.axis('equal')
plt.show()


###############################################################################
# We now build the deformation modules that will be used by the model.
#

nu = 0.1
scale0 = 5.
scale1 = 3./(growth_density**(1/2))
coeff0 = 100.
coeff1 = 0.1

global_translation = imodal.DeformationModules.GlobalTranslation(2)

small_scale_translation = imodal.DeformationModules.ImplicitModule0(
    2, points_small.shape[0], scale0, coeff=coeff0, nu=nu,
    gd=points_small)

growth = imodal.DeformationModules.ImplicitModule1(
    2, points_growth.shape[0], scale0, C, coeff=coeff1, nu=nu,
    gd=(points_growth, rot_growth))


###############################################################################
# We now define the model.
#

deformable_shape_source = imodal.Models.DeformablePoints(shape_source)
deformable_shape_target = imodal.Models.DeformablePoints(shape_target)

model = imodal.Models.RegistrationModel([deformable_shape_source], [global_translation, small_scale_translation, growth], [imodal.Attachment.VarifoldAttachment(2, [10., 150.])], lam=100.)


###############################################################################
# Fitting.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([deformable_shape_target], 50, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})

###############################################################################
# Plot results. Matching is very good.
#

intermediates = {}
model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)
intermediate_states = intermediates['states']

deformed_source = intermediate_states[-1][0].gd
deformed_small = intermediate_states[-1][2].gd
deformed_growth = intermediate_states[-1][3].gd[0]

aabb_target = imodal.Utilities.AABB.build_from_points(shape_target).scale(1.2)

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-')
plt.plot(points_small[:, 0].numpy(), points_small[:, 1].numpy(), 'x')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), '.')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_source[:, 0], deformed_source[:, 1], '-')
plt.plot(deformed_small[:, 0], deformed_small[:, 1], 'x')
plt.plot(deformed_growth[:, 0], deformed_growth[:, 1], '.')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-')
plt.plot(deformed_source[:, 0], deformed_source[:, 1], '-')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.show()


###############################################################################
# We now compute the influence on the deformation of the small scale module
# and of the growth module.
#
# We first need to compute the controls of each modules.
#

modules = imodal.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold.clone())

intermediates = {}
imodal.HamiltonianDynamic.shoot(imodal.HamiltonianDynamic.Hamiltonian(modules), shoot_solver, shoot_it, intermediates=intermediates)

intermediate_states = intermediates['states']
intermediate_controls = intermediates['controls']

ss_controls = [control[2] for control in intermediate_controls]
growth_controls = [control[3] for control in intermediate_controls]

###############################################################################
# We know compute the deformation grid of the small scale module.
#


# We extract the modules of the models and fill the right manifolds.
modules = imodal.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold)
silent = copy.copy(modules[0])
deformation_grid = imodal.DeformationModules.DeformationGrid(imodal.Utilities.AABB.build_from_points(shape_source).scale(1.2), [32, 32])
small_scale = copy.copy(modules[2])

# We construct the controls list we will give will shooting
controls = [[torch.tensor([]), torch.tensor([]), control] for control in ss_controls]

imodal.HamiltonianDynamic.shoot(imodal.HamiltonianDynamic.Hamiltonian([silent, deformation_grid, small_scale]), shoot_solver, shoot_it, controls=controls)

ss_deformed_source = silent.manifold.gd.detach()
ss_deformed_grid = deformation_grid.togrid()


###############################################################################
# We know compute the deformation grid of the growth module.
#

# We extract the modules of the models and fill the right manifolds.
modules = imodal.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold.clone())
silent = copy.copy(modules[0])
deformation_grid = imodal.DeformationModules.DeformationGrid(imodal.Utilities.AABB.build_from_points(shape_source).scale(1.2), [32, 32])
growth = copy.copy(modules[3])

# We construct the controls list we will give will shooting
controls = [[torch.tensor([]), torch.tensor([]), control] for control in growth_controls]

imodal.HamiltonianDynamic.shoot(imodal.HamiltonianDynamic.Hamiltonian([silent, deformation_grid, growth]), shoot_solver, shoot_it, controls=controls)

growth_deformed_source = silent.manifold.gd.detach()
growth_deformed_grid = deformation_grid.togrid()

###############################################################################
# We now plot both results. We see that most of the deformation comes from the
# growth module, as expected.
#

ax = plt.subplot(1, 2, 1)
plt.title("Growth module")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '--', color='black')
plt.plot(growth_deformed_source[:, 0].numpy(), growth_deformed_source[:, 1].numpy())
imodal.Utilities.plot_grid(ax, growth_deformed_grid[0], growth_deformed_grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')

ax = plt.subplot(1, 2, 2)
plt.title("Small scale module")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '--', color='black')
plt.plot(ss_deformed_source[:, 0].numpy(), ss_deformed_source[:, 1].numpy())
imodal.Utilities.plot_grid(ax, ss_deformed_grid[0], ss_deformed_grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')

plt.show()

