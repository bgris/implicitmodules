import sys

sys.path.append("../../../")

import math
import pickle
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as dm

save_fig = True


def show(figname=None):
    if save_fig:
        plt.savefig(figname, dpi=300.)
        plt.close()
    else:
        plt.show()


with open("results.pt", 'rb') as f:
    results = torch.load(f)


    
path = '../../../../Results/ImplicitModules/Leaf/results_leaf_acropetal_model/0/'
    
ext = ".png"
figname = path + "images/" + results['experience_name']
source_shape = results['source_shape']
target_shape = results['target_shape']
source_dots = results['source_dots']
target_dots = results['target_dots']
deformed_shape = results['deformed_shape']
intermediates = results['intermediates']
intermediates_growth = results['intermediates_growth']
growth_grid_resolution = results['growth_grid_resolution']
abc = results['abc']


def polynomial(pos, a, b, c, d, e, f):
    return a + b*pos[:, 1] + c*pos[:, 0]**2 + d*pos[:, 1]**2 + e*pos[:, 0]**2*pos[:, 1] + f*pos[:, 1]**3


figsize = (3., 6.)

plt.subplots(figsize=figsize)
plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
plt.plot(target_shape[:, 0].numpy(), target_shape[:, 1].numpy(), '-', color='green', lw=0.5)
plt.plot(source_dots[:, 0].numpy(), source_dots[:, 1].numpy(), '.', color='blue')
plt.plot(target_dots[:, 0].numpy(), target_dots[:, 1].numpy(), '.', color='green')
plt.axis('equal')
show(figname+"_data"+ext)

growth_points = intermediates['states'][0][3].gd[0]
growth_rot = intermediates['states'][0][3].gd[1]
plt.subplots(figsize=figsize)
plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
plt.plot(growth_points[:, 0].numpy(), growth_points[:, 1].numpy(), '.', color='black')
plt.axis('equal')

show(figname+"_points"+ext)

growth_constants = polynomial(growth_points, abc[0].unsqueeze(1), abc[1].unsqueeze(1), abc[2].unsqueeze(1), abc[3].unsqueeze(1), abc[4].unsqueeze(1), abc[5].unsqueeze(1)).transpose(0, 1). unsqueeze(2)

# x = torch.stack([1.*torch.ones(100), torch.linspace(torch.min(source_shape[:, 1])-5., torch.max(source_shape[:, 1])+5., 100)], dim=1)
# y = polynomial(x, abc[0].unsqueeze(1), abc[1].unsqueeze(1), abc[2].unsqueeze(1), abc[3].unsqueeze(1), abc[4].unsqueeze(1), abc[5].unsqueeze(1)).transpose(0, 1)
# print(x.shape)
# print(y.shape)

# plt.plot(x[:, 1].numpy(), y[:, 1].numpy())
# plt.show()

_, ax = plt.subplots(figsize=figsize)
plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
dm.Utilities.plot_C_ellipses(ax, growth_points, growth_constants, growth_rot, color='blue', scale=100.)

plt.axis('off')
show(figname+"_growth"+ext)

for i, state in enumerate(intermediates['states']):
    plt.subplots(figsize=figsize)
    plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
    plt.plot(target_shape[:, 0].numpy(), target_shape[:, 1].numpy(), '-', color='green', lw=0.5)
    plt.plot(source_dots[:, 0].numpy(), source_dots[:, 1].numpy(), '.', color='blue')
    plt.plot(target_dots[:, 0].numpy(), target_dots[:, 1].numpy(), '.', color='green')

    deformed_shape = state[0].gd
    deformed_dots = state[1].gd
    plt.plot(deformed_shape[:, 0].numpy(), deformed_shape[:, 1].numpy(), '-', color='red')
    plt.plot(deformed_dots[:, 0].numpy(), deformed_dots[:, 1].numpy(), '.', color='red')
    plt.axis('off')
    plt.axis('equal')
    show(figname+"_deformed_{}".format(i)+ext)


for i, state in enumerate(intermediates_growth['states']):
    _, ax = plt.subplots(figsize=figsize)
    plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
    plt.plot(target_shape[:, 0].numpy(), target_shape[:, 1].numpy(), '-', color='green', lw=0.5)
    plt.plot(source_dots[:, 0].numpy(), source_dots[:, 1].numpy(), '.', color='blue')
    plt.plot(target_dots[:, 0].numpy(), target_dots[:, 1].numpy(), '.', color='green')

    deformed_shape = state[0].gd
    deformed_dots = state[1].gd
    deformed_grid = dm.Utilities.vec2grid(state[2].gd, *growth_grid_resolution)
    plt.plot(deformed_shape[:, 0].numpy(), deformed_shape[:, 1].numpy(), '-', color='red')
    plt.plot(deformed_dots[:, 0].numpy(), deformed_dots[:, 1].numpy(), '.', color='red')
    dm.Utilities.plot_grid(ax, deformed_grid[0], deformed_grid[1], color='xkcd:light blue', lw=0.3)
    plt.axis('off')
    plt.axis('equal')
    show(figname+"_deformed_growth_{}".format(i)+ext)


for i, state in enumerate(intermediates['states']):
    implicit1_points = state[3].gd[0]
    implicit1_r = state[3].gd[1]
    _, ax = plt.subplots(figsize=figsize)
    plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
    plt.plot(target_shape[:, 0].numpy(), target_shape[:, 1].numpy(), '-', color='green', lw=0.5)
    dm.Utilities.plot_C_ellipses(ax, implicit1_points, growth_constants, implicit1_r, scale=0.28, color='blue')
    plt.axis('off')
    plt.axis('equal')
    show(figname+"_deformed_growth_constants_{}".format(i)+ext)


