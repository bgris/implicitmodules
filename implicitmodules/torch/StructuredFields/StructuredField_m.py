import torch

from implicitmodules.torch.Kernels.kernels import gauss_kernel, rel_differences
from implicitmodules.torch.StructuredFields.Abstract import SupportStructuredField

from pykeops.torch import Genred

class StructuredField_m(SupportStructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma
    
    @property
    def sigma(self):
        return self.__sigma
    
    def __call__(self, points, k=0):
        
        P = (self.moments - torch.transpose(self.moments, 1, 2)) / 2
        
        if k==0:
            d = points.shape[1]
            formula = 'MatVecMult( GradMatrix(Exp(-p * SqDist(x, y)), y), a) '
            variables = ['x = Vi(' + str(d) + ')',
                         'y = Vj(' + str(d) + ')',
                         'a = Vj(' + str(d) + ')',
                         'p = Pm(1)']
            cuda_type = "float32"
            my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=cuda_type)
            
            res21 = my_routine(points, self.support, P[:,:,0].contiguous(), torch.tensor([.5 / self.sigma / self.sigma]), backend="auto")
            res22 = my_routine(points, self.support, P[:, :, 1].contiguous(),
                              torch.tensor([.5 / self.sigma / self.sigma]), backend="auto")
            res2 = -torch.cat([res21, res22], dim=1)
        
        
        
        ker_vec = -gauss_kernel(rel_differences(points, self.support), k + 1, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
    
        #  res = torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2),
        #                        P,
        #                        dims=([2, 3, 4], [1, 0, 2]))
    
        res = torch.transpose(torch.tensordot(
            torch.tensordot(torch.eye(2), ker_vec, dims=0),
            P,
            dims=([0, 3, 4], [1, 0, 2])),
            0, 1)
    
        return res
