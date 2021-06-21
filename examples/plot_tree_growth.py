"""
Analyzing Differences Between Tree Images
=========================================

Image registration with an implicit module of order 1. Segmentations given by the data are used to initialize its points.
"""


###############################################################################
# Initialization
# --------------
#
# Import relevant Python modules.
#

import time
import pickle
import sys
sys.path.append("../")

import matplotlib.pyplot as plt
import torch

import imodal


device = 'cuda:1'
torch.set_default_dtype(torch.float64)
imodal.Utilities.set_compute_backend('keops')


###############################################################################
# Load source and target images, along with the source curve.
#

with open("../data/tree_growth.pickle", 'rb') as f:
    data = pickle.load(f)

source_shape = data['source_shape'].to(torch.get_default_dtype())
source_image = data['source_image'].to(torch.get_default_dtype())
target_image = data['target_image'].to(torch.get_default_dtype())

# Segmentations as Axis Aligned Bounding Boxes (AABB)
aabb_trunk = data['aabb_trunk']
aabb_crown = data['aabb_leaves']
extent = data['extent']


###############################################################################
# Display source and target images, along with the segmented source curve (orange
# for the trunk, green for the crown).
#

shape_is_trunk = aabb_trunk.is_inside(source_shape)
shape_is_crown = aabb_crown.is_inside(source_shape)

plt.subplot(1, 2, 1)
plt.title("Source")
plt.imshow(source_image, cmap='gray', origin='lower', extent=extent.totuple())
plt.plot(source_shape[shape_is_trunk, 0].numpy(), source_shape[shape_is_trunk, 1].numpy(), lw=2., color='orange')
plt.plot(source_shape[shape_is_crown, 0].numpy(), source_shape[shape_is_crown, 1].numpy(), lw=2., color='green')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Target")
plt.imshow(target_image, cmap='gray', origin='lower', extent=extent.totuple())
plt.axis('off')
plt.show()


###############################################################################
# Generating implicit modules of order 1 points and growth model tensor.
#

implicit1_density = 500.

# Lambda function defining the area in and around the tree shape
area = lambda x, **kwargs: imodal.Utilities.area_shape(x, **kwargs) | imodal.Utilities.area_polyline_outline(x, **kwargs)
polyline_width = 0.07

# Generation of the points of the initial geometrical descriptor
implicit1_points = imodal.Utilities.fill_area_uniform_density(area, imodal.Utilities.AABB(xmin=0., xmax=1., ymin=0., ymax=1.), implicit1_density, shape=source_shape, polyline=source_shape, width=polyline_width)

# Masks that flag points into either the trunk or the crown
implicit1_trunk_points = aabb_trunk.is_inside(implicit1_points)
implicit1_crown_points = aabb_crown.is_inside(implicit1_points)

implicit1_points = implicit1_points[implicit1_trunk_points | implicit1_crown_points]
implicit1_trunk_points = aabb_trunk.is_inside(implicit1_points)
implicit1_crown_points = aabb_crown.is_inside(implicit1_points)

assert implicit1_points[implicit1_trunk_points].shape[0] + implicit1_points[implicit1_crown_points].shape[0] == implicit1_points.shape[0]

# Initial normal frames
implicit1_r = torch.eye(2).repeat(implicit1_points.shape[0], 1, 1)

# Growth model tensor
implicit1_c = torch.zeros(implicit1_points.shape[0], 2, 4)

# Horizontal stretching for the trunk
implicit1_c[implicit1_trunk_points, 0, 0] = 1.
# Vertical stretching for the trunk
implicit1_c[implicit1_trunk_points, 1, 1] = 1.
# Horizontal stretching for the crown
implicit1_c[implicit1_crown_points, 0, 2] = 1.
# Vertical stretching for the crown
implicit1_c[implicit1_crown_points, 1, 3] = 1.


###############################################################################
# Plot the 4 dimensional growth model tensor.
#

plt.figure(figsize=[20., 5.])
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    plt.imshow(source_image, origin='lower', extent=extent, cmap='gray')
    imodal.Utilities.plot_C_ellipses(ax, implicit1_points, implicit1_c, c_index=i, color='blue', scale=0.03)
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.axis('off')

plt.show()


###############################################################################
# Create the deformation model with a combination of 2 modules : a global translation and the implicit module of order 1.
#


###############################################################################
# Create and initialize the global translation module **global_translation**.
#

global_translation_coeff = 1.
global_translation = imodal.DeformationModules.GlobalTranslation(2, coeff=global_translation_coeff)


###############################################################################
# Create and initialize the implicit module of order 1 **implicit1**.
#

sigma1 = 2./implicit1_density**(1/2)
implicit1_coeff = 0.1
implicit1_nu = 100.
implicit1 = imodal.DeformationModules.ImplicitModule1(2, implicit1_points.shape[0], sigma1, implicit1_c, nu=implicit1_nu, gd=(implicit1_points, implicit1_r), coeff=implicit1_coeff)
implicit1.eps = 1e-2


###############################################################################
# Define deformables used by the registration model.
#

source_image_deformable = imodal.Models.DeformableImage(source_image, output='bitmap', extent=extent)
target_image_deformable = imodal.Models.DeformableImage(target_image, output='bitmap', extent=extent)

source_image_deformable.to_device(device)
target_image_deformable.to_device(device)

###############################################################################
# Registration
# ------------
# Define the registration model.
#

attachment_image = imodal.Attachment.L2NormAttachment(weight=1e0)

model = imodal.Models.RegistrationModel([source_image_deformable], [implicit1, global_translation], [attachment_image], lam=1.)
model.to_device(device)

###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([target_image_deformable], 500, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe', 'history_size': 500})


###############################################################################
# Visualization
# -------------
# Compute optimized deformation trajectory.
#

deformed_intermediates = {}
start = time.perf_counter()
with torch.autograd.no_grad():
    deformed_image = model.compute_deformed(shoot_solver, shoot_it, intermediates=deformed_intermediates)[0][0].detach().cpu()
print("Elapsed={elapsed}".format(elapsed=time.perf_counter()-start))


###############################################################################
# Display deformed source image and target.
#

plt.figure(figsize=[15., 5.])
plt.subplot(1, 3, 1)
plt.title("Source")
plt.imshow(source_image, extent=extent.totuple(), origin='lower')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Deformed")
plt.imshow(deformed_image, extent=extent.totuple(), origin='lower')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Target")
plt.imshow(target_image, extent=extent.totuple(), origin='lower')
plt.axis('off')

plt.show()


###############################################################################
# We can follow the action of each part of the total deformation by setting all the controls components to zero but one.
#

###############################################################################
# Functions generating controls to follow one part of the deformation.
#

def generate_implicit1_controls(table):
    outcontrols = []
    for control in deformed_intermediates['controls']:
        outcontrols.append(control[1]*torch.tensor(table, dtype=torch.get_default_dtype(), device=device))

    return outcontrols


def generate_controls(implicit1_table, trans):
    outcontrols = []
    implicit1_controls = generate_implicit1_controls(implicit1_table)
    for control, implicit1_control in zip(deformed_intermediates['controls'], implicit1_controls):
        outcontrols.append([implicit1_control, control[2]*torch.tensor(trans, dtype=torch.get_default_dtype(), device=device)])

    return outcontrols


###############################################################################
# Function to compute a deformation given a set of controls up to some time point.
#

grid_resolution = [16, 16]


def compute_intermediate_deformed(it, controls, t1, intermediates=None):
    implicit1_points = deformed_intermediates['states'][0][1].gd[0]
    implicit1_r = deformed_intermediates['states'][0][1].gd[1]
    implicit1_cotan_points = deformed_intermediates['states'][0][1].cotan[0]
    implicit1_cotan_r = deformed_intermediates['states'][0][1].cotan[1]
    silent_cotan = deformed_intermediates['states'][0][0].cotan

    implicit1 = imodal.DeformationModules.ImplicitModule1(2, implicit1_points.shape[0], sigma1, implicit1_c.clone(), nu=implicit1_nu, gd=(implicit1_points.clone(), implicit1_r.clone()), cotan=(implicit1_cotan_points, implicit1_cotan_r), coeff=implicit1_coeff)
    global_translation = imodal.DeformationModules.GlobalTranslation(2, coeff=global_translation_coeff)

    implicit1.to_(device=device)
    global_translation.to_(device=device)

    source_deformable = imodal.Models.DeformableImage(source_image, output='bitmap', extent=extent)
    source_deformable.silent_module.manifold.cotan = silent_cotan

    grid_deformable = imodal.Models.DeformableGrid(extent, grid_resolution)

    source_deformable.to_device(device)
    grid_deformable.to_device(device)

    costs = {}
    with torch.autograd.no_grad():
        deformed = imodal.Models.deformables_compute_deformed([source_deformable, grid_deformable], [implicit1, global_translation], shoot_solver, it, controls=controls, t1=t1, intermediates=intermediates, costs=costs)

    return deformed[0][0]


###############################################################################
# Functions to generate the deformation trajectory given a set of controls.
#

def generate_images(table, trans, outputfilename):
    incontrols = generate_controls(table, trans)
    intermediates_shape = {}
    deformed = compute_intermediate_deformed(10, incontrols, 1., intermediates=intermediates_shape)

    trajectory_grid = [imodal.Utilities.vec2grid(state[1].gd, grid_resolution[0], grid_resolution[1]) for state in intermediates_shape['states']]

    trajectory = [source_image]
    t = torch.linspace(0., 1., 11)
    indices = [0, 3, 7, 10]
    print("Computing trajectories...")
    for index in indices[1:]:
        print("{}, t={}".format(index, t[index]))
        deformed = compute_intermediate_deformed(index, incontrols[:4*index], t[index])

        trajectory.append(deformed)

    print("Generating images...")
    plt.figure(figsize=[5.*len(indices), 5.])
    for deformed, i in zip(trajectory, range(len(indices))):
        ax = plt.subplot(1, len(indices), i + 1)

        grid = trajectory_grid[indices[i]]
        plt.imshow(deformed.cpu(), origin='lower', extent=extent, cmap='gray')
        imodal.Utilities.plot_grid(ax, grid[0].cpu(), grid[1].cpu(), color='xkcd:light blue', lw=1)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


###############################################################################
# Generate trajectory of the total optimized deformation.
#

generate_images([True, True, True, True], True, "deformed_all")

###############################################################################
# Generate trajectory following vertical elongation of the trunk.
#

generate_images([False, True, False, False], False, "deformed_trunk_vertical")

###############################################################################
# Generate trajectory following horizontal elongation of the crown.
#

generate_images([False, False, True, False], False, "deformed_crown_horizontal")


