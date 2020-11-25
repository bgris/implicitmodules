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


with open("results.pickle", 'rb') as f:
    results = pickle.load(f)

ext = ".png"
figname = "images/" + results['experience_name']
source_shape = results['source_shape']
target_shape = results['target_shape']
source_dots = results['source_dots']
target_dots = results['target_dots']
deformed_shape = results['deformed_shape']
intermediates = results['intermediates']
intermediates_lddmm = results['intermediates_lddmm']
lddmm_grid_resolution = results['lddmm_grid_resolution']

figsize = (3., 6.)

plt.subplots(figsize=figsize)
plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
plt.plot(target_shape[:, 0].numpy(), target_shape[:, 1].numpy(), '-', color='green', lw=0.5)
plt.plot(source_dots[:, 0].numpy(), source_dots[:, 1].numpy(), 'o', markersize=0.5, color='blue')
plt.plot(target_dots[:, 0].numpy(), target_dots[:, 1].numpy(), 'o', markersize=0.5, color='green')
plt.axis('equal')
show(figname+"_data"+ext)

lddmm_points = intermediates['states'][0][1].gd
plt.subplots(figsize=figsize)
plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
plt.plot(lddmm_points[:, 0].numpy(), lddmm_points[:, 1].numpy(), 'o', markersize=0.5, color='black')
plt.axis('equal')
show(figname+"_points"+ext)


for i, state in enumerate(intermediates['states']):
    plt.subplots(figsize=figsize)
    plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
    plt.plot(target_shape[:, 0].numpy(), target_shape[:, 1].numpy(), '-', color='green', lw=0.5)
    plt.plot(source_dots[:, 0].numpy(), source_dots[:, 1].numpy(), 'o', markersize=0.5, color='blue')
    plt.plot(target_dots[:, 0].numpy(), target_dots[:, 1].numpy(), 'o', markersize=0.5, color='green')

    deformed_shape = state[0].gd
    deformed_dots = state[1].gd
    plt.plot(deformed_shape[:, 0].numpy(), deformed_shape[:, 1].numpy(), '-', color='red')
    plt.plot(deformed_dots[:, 0].numpy(), deformed_dots[:, 1].numpy(), 'o', markersize=0.5, color='red')
    plt.axis('off')
    plt.axis('equal')
    show(figname+"_deformed_{}".format(i)+ext)


for i, state in enumerate(intermediates_lddmm['states']):
    _, ax = plt.subplots(figsize=figsize)
    plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), '-', color='blue', lw=0.5)
    plt.plot(target_shape[:, 0].numpy(), target_shape[:, 1].numpy(), '-', color='green', lw=0.5)
    plt.plot(source_dots[:, 0].numpy(), source_dots[:, 1].numpy(), 'o', markersize=0.5, color='blue')
    plt.plot(target_dots[:, 0].numpy(), target_dots[:, 1].numpy(), 'o', markersize=0.5, color='green')

    deformed_shape = state[0].gd
    deformed_dots = state[1].gd
    deformed_grid = dm.Utilities.vec2grid(state[2].gd, *lddmm_grid_resolution)
    plt.plot(deformed_shape[:, 0].numpy(), deformed_shape[:, 1].numpy(), '-', color='red')
    plt.plot(deformed_dots[:, 0].numpy(), deformed_dots[:, 1].numpy(), 'o', markersize=0.5, color='red')
    dm.Utilities.plot_grid(ax, deformed_grid[0], deformed_grid[1], color='xkcd:light blue', lw=0.3)
    plt.axis('off')
    plt.axis('equal')
    show(figname+"_deformed_lddmm_{}".format(i)+ext)

