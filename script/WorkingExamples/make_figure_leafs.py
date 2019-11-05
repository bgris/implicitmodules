


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

import pickle
#%%

path_results = '/home/gris/Results/ImplicitModules/Leaf/'
exp = 'basipetal_data'


with open(path_results + exp +'.pkl', 'rb') as f:
    img, lx = pickle.load(f)
#%%
    
