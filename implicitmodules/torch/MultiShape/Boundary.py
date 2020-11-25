import torch

from implicitmodules.torch.Models import DeformablePoints
from implicitmodules.torch.Utilities import close_shape, area_shape, is_shape_closed


class Boundary(DeformablePoints):
    def __init__(self, points):
        #if not is_shape_closed(points):
        #    points = close_shape(points)
        super().__init__(points)


    def isin_label(self, points):
        return area_shape(points, shape=self.silent_module.manifold.gd)
    
    
    def isin_extract(self, points):
        labels = self.isin_label(points)
        return points[labels==True]
    
    