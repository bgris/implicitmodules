import torch

import implicitmodules.torch.DeformationModules as dm


class MultiShape:
    def __init__(self, module_list, sigma_background):
        self.__nb_shapes = len(module_list)
        self.__sigma_background = sigma_background
        self.__modules = module_list
        
    def copy(self):
        return Multishape([mod.copy() for mod in self.__module_list], self.__sigma_background)