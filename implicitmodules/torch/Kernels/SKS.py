import numpy as np
import torch

from implicitmodules.torch.Kernels.kernels import gauss_kernel, rel_differences


def eta():
    return torch.tensor([[[1., 0., 0.], [0., 1 / np.sqrt(2), 0.]],
                         [[0., 1 / np.sqrt(2), 0.], [0., 0., 1.]]], dtype=torch.get_default_dtype())


def compute_sks(x, sigma, order):
    N = x.shape[0]
    if order == 0: # not used
        return torch.einsum('ij, kl->ijkl', gauss_kernel(rel_differences(x, x), 0, sigma).view(N, N),
                            torch.eye(2)).permute([0, 2, 1, 3]).contiguous().view(2 * N, 2 * N)
    elif order == 1:
        A = torch.tensordot(-gauss_kernel(rel_differences(x, x), 2, sigma), torch.eye(2), dims=0)
        K = torch.tensordot(torch.transpose(A, 2, 3), eta())
        K = torch.tensordot(K, eta(), dims=([1, 2], [0, 1]))
        return K.view(N, N, 3, 3).contiguous().permute([0, 2, 1, 3]).contiguous().view(3 * N, 3 * N)
    
    
    
    
    
    else:
        raise NotImplementedError
