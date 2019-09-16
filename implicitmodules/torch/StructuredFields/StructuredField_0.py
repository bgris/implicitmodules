import torch

from implicitmodules.torch.Kernels.kernels import gauss_kernel, rel_differences
from implicitmodules.torch.StructuredFields.Abstract import SupportStructuredField
from pykeops.torch import Genred

class StructuredField_0(SupportStructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma
    
    @property
    def sigma(self):
        return self.__sigma
    
    def __call__(self, points, k=0):
        """
        If k=0:  sum_j K(points, support_j) * moments_j
        If k=1   sum_j \partial_points K(points, support_j) (x) moments_j
        """
        #  Keops
        if k==0:
            d = points.shape[1]
            formula = 'Exp(-p * SqDist(x, y)) * b'
            variables = ['x = Vi(' + str(d) + ')',
                         'y = Vj(' + str(d) + ')',
                         'b = Vj(' + str(d) + ')',
                         'p = Pm(1)']
            cuda_type = "float32"
            my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=cuda_type)
            res = my_routine(points, self.support, self.moments, torch.tensor([.5/self.sigma/self.sigma]), backend="auto")
        
        elif k==1:
            
            d = points.shape[1]
            formula = 'GradMatrix(Exp(-p * SqDist(x, y)) * b, x)'
            variables = ['x = Vi(' + str(d) + ')',
                         'y = Vj(' + str(d) + ')',
                         'b = Vj(' + str(d) + ')',
                         'p = Pm(1)']
            cuda_type = "float32"
            my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=cuda_type)
            res2 = my_routine(points, self.support, self.moments, torch.tensor([.5/ self.sigma / self.sigma]), backend="auto").reshape(-1, d, d)
            
        else:
            NotImplementedError
        #  End Keops
        
        ker_vec = gauss_kernel(rel_differences(points, self.support), k, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        res = torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), self.moments,
                              dims=([2, 3], [1, 0]))
        return res
