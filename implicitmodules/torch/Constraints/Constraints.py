import torch

from typing import Iterable


class Constraints:
    def __init__(self, *parameters):
        """
        manifolds is a list of manifolds
        """
                
        
class ConstraintsPointIdentityBase:
    """
    Identity between the 'tan' component of the manifolds two specified manifolds (by an index) 
    """
    def __init__(self, indexes_manifolds0, indexes_manifolds1, dimconstraint): 
        """
        manifolds is a multi-shape compound manifold
        indexes_manifolds is a tuple of 2 indexes
        """
        self.__indexes0 = indexes_manifolds1
        self.__indexes1 = indexes_manifolds0
        if indexes_manifolds0[0] < indexes_manifolds1[0]:
            self.__indexes0 = indexes_manifolds0
            self.__indexes1 = indexes_manifolds1

        self.__dimconstraint = dimconstraint
        #super().__init__()
        
    def __call__(self, manifolds):
        """
        returns a tensor
        """
        if len(self.__indexes0) == 1:
            tan0 = manifolds[self.__indexes0[0]].tan
        else:
            tan0 = manifolds[self.__indexes0[0]][self.__indexes0[1]].tan        
        
        if len(self.__indexes1) == 1:
            tan1 = manifolds[self.__indexes1[0]].tan
        else:
            tan1 = manifolds[self.__indexes1[0]][self.__indexes1[1]].tan   
       
        #print('tan0', tan0)
        #print('tan1', tan1)
        return (tan0 - tan1).view(-1, 1)
    
    @property
    def dimconstraint(self):
        return self.__dimconstraint
    
    def adjoint(self, lam, manifold):
        man = manifold.clone()
        shape = man[self.__indexes0[0]][self.__indexes0[1]].cotan.shape
        man.fill_cotan_zeros()
        man[self.__indexes0[0]][self.__indexes0[1]].fill_cotan(lam.view(shape))
        man[self.__indexes1[0]][self.__indexes1[1]].fill_cotan(-lam.view(shape))
        return man
    
    def matrixAMAs(self, M, manifolds):
        """
        returns A_q M A_q^\ast (corresponds to extracting sub matrices of M and sum/substract : 
        M_11 -M_21 -M_12 + M22)
        """
        
        c0 = sum([sum(man.numel_gd) for man in manifolds[:self.__indexes0[0]]])
        
        if len(self.__indexes0) == 1:
            c1 = 0
            d0 = sum(manifolds[self.__indexes0[0]].numel_gd)
            c2 = 0
        else:
            c1 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes0[0]][:self.__indexes0[1]]])
            c2 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes0[0]][self.__indexes0[1]+1:]])
            d0 = sum(manifolds[self.__indexes0[0]][self.__indexes0[1]].numel_gd)
        
        c3 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes0[0] + 1:self.__indexes1[0]]])
        
        if len(self.__indexes1) == 1:
            c4 = 0
            c5 = 0
            d1 = sum(manifolds[self.__indexes1[0]].numel_gd)
        else:
            c4 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes1[0]][:self.__indexes1[1]]])
            c5 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes1[0]][self.__indexes1[1]+1:]])
            d1 = sum(manifolds[self.__indexes1[0]][self.__indexes1[1]].numel_gd)
        
        c6 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes1[0]:]])
        assert (d0==d1)
        
        ind0 = c0 + c1
        ind1 = c0 + c1 + d0 + c2 + c3 + c4
        return M[ind0:ind0 + d0, ind0:ind0 + d0] -M[ind1:ind1+d1, ind0:ind0 + d0]-M[ind1:ind1+d1, ind0:ind0 + d0] + M[ind1:ind1+d1, ind1:ind1+d1]
        
    
    
class ConstraintsPointIdentityBackground(ConstraintsPointIdentityBase):
    """
    Identity between one specified module (by an index) and the background one on a specified boundary (same index as the module)
    """
    def __init__(self, indexes_module, dimconstraint):  

        self.__dimconstraint = dimconstraint
        self.__index = indexes_module
        #indexes_manifolds0 = [indexes_module[0], indexes_module[1] - 1] 
        #indexes_manifolds1 = [indexes_module[2] - 1 , indexes_module[0]]
        #indexes_manifolds0 = [indexes_module[0],  - 1] 
        #indexes_manifolds1 = [- 1 , indexes_module[0]]
        
        #indexes_manifolds0 = [indexes_module, 0] 
        #indexes_manifolds1 = [len(manifolds.manifolds) - 1 , indexes_module]
        #super().__init__(indexes_manifolds0, indexes_manifolds1, dimconstraint)

        
    def generate_indexes(self, manifolds):
        
        indexes_manifolds0 = [self.__index, len(manifolds.manifolds[self.__index].manifolds) - 1] 
        indexes_manifolds1 = [len(manifolds.manifolds) - 1 , self.__index]
        return [indexes_manifolds0, indexes_manifolds1]
        
    def __call__(self, manifolds):
        """
        returns a tensor
        """
        indexes_manifolds0, indexes_manifolds1 = self.generate_indexes(manifolds)
        
        if len(indexes_manifolds0) == 1:
            tan0 = manifolds[indexes_manifolds0[0]].tan
        else:
            tan0 = manifolds[indexes_manifolds0[0]][indexes_manifolds0[1]].tan        
        
        if len(indexes_manifolds1) == 1:
            tan1 = manifolds[indexes_manifolds1[0]].tan
        else:
            tan1 = manifolds[indexes_manifolds1[0]][indexes_manifolds1[1]].tan   
       
        #print('tan0', tan0)
        #print('tan1', tan1)
        return (tan0 - tan1).view(-1, 1)
    
    @property
    def dimconstraint(self):
        return self.__dimconstraint
    
    def adjoint(self, lam, manifolds):
        indexes_manifolds0, indexes_manifolds1 = self.generate_indexes(manifolds)
        
        man = manifolds.clone()
        shape = man[indexes_manifolds0[0]][indexes_manifolds0[1]].cotan.shape
        man.fill_cotan_zeros()
        man[indexes_manifolds0[0]][indexes_manifolds0[1]].fill_cotan(lam.view(shape))
        man[indexes_manifolds1[0]][indexes_manifolds1[1]].fill_cotan(-lam.view(shape))
        return man
    
    def matrixAMAs(self, M, manifolds):
        """
        returns A_q M A_q^\ast (corresponds to extracting sub matrices of M and sum/substract : 
        M_11 -M_21 -M_12 + M22)
        """
        indexes_manifolds0, indexes_manifolds1 = self.generate_indexes(manifolds)
                
        c0 = sum([sum(man.numel_gd) for man in manifolds[:indexes_manifolds0[0]]])
        
        if len(indexes_manifolds0) == 1:
            c1 = 0
            d0 = sum(manifolds[indexes_manifolds0[0]].numel_gd)
            c2 = 0
        else:
            c1 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds0[0]][:indexes_manifolds0[1]]])
            c2 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds0[0]][indexes_manifolds0[1]+1:]])
            d0 = sum(manifolds[indexes_manifolds0[0]][indexes_manifolds0[1]].numel_gd)
        
        c3 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds0[0] + 1:indexes_manifolds1[0]]])
        
        if len(indexes_manifolds1) == 1:
            c4 = 0
            c5 = 0
            d1 = sum(manifolds[indexes_manifolds1[0]].numel_gd)
        else:
            c4 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds1[0]][:indexes_manifolds1[1]]])
            c5 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds1[0]][indexes_manifolds1[1]+1:]])
            d1 = sum(manifolds[indexes_manifolds1[0]][indexes_manifolds1[1]].numel_gd)
        
        c6 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds1[0]:]])
        assert (d0==d1)
        
        ind0 = c0 + c1
        ind1 = c0 + c1 + d0 + c2 + c3 + c4
        return M[ind0:ind0 + d0, ind0:ind0 + d0] -M[ind0:ind0 + d0, ind1:ind1+d1]-M[ind1:ind1+d1, ind0:ind0 + d0] + M[ind1:ind1+d1, ind1:ind1+d1]

        
        
class ConstraintsPointSlinding:
    """
    Identity between the 'tan' component of the manifolds two specified manifolds (by an index) 
    """
    def __init__(self, indexes_manifolds0, indexes_manifolds1, dimconstraint): 
        """
        manifolds is a multi-shape compound manifold
        indexes_manifolds is a tuple of 2 indexes
        """
        self.__indexes0 = indexes_manifolds1
        self.__indexes1 = indexes_manifolds0
        if indexes_manifolds0[0] < indexes_manifolds1[0]:
            self.__indexes0 = indexes_manifolds0
            self.__indexes1 = indexes_manifolds1

        self.__dimconstraint = dimconstraint
        #super().__init__()
    
    def compute_normals(self, manifolds):
        """
        Computes normals at boundary points
        """
        pts0 = manifolds[self.__indexes0[0]][self.__indexes0[1]].gd
        diff0 = torch.empty_like(pts0)
        diff0[1:-1] = pts0[2:] - pts0[:-2]
        diff0[0] = pts0[1] - pts0[0]
        diff0[-1] = pts0[-1] - pts0[-2]
        no0 = torch.norm(diff0, 1)
        tan0 = diff0/no0
        normals0 = torch.flip(tan0)
        normals0[:,0] = -normals0[:,0]
        
        
        pts1 = manifolds[self.__indexes1[0]][self.__indexes1[1]].gd
        diff1 = torch.empty_like(pts1)
        diff1[1:-1] = pts1[2:] - pts1[:-2]
        diff1[0] = pts1[1] - pts1[0]
        diff1[-1] = pts1[-1] - pts1[-2]
        no1 = torch.norm(diff1, 1)
        tan1 = diff1/no1
        normals1 = torch.flip(tan1)
        normals1[:,0] = -normals1[:,0]
        
        return [normals0, normals1]
    
    
    def __call__(self, manifolds):
        """
        returns a tensor
        """
        normals0, normals1 = self.compute_normals
        
        if len(self.__indexes0) == 1:
            tan0 = manifolds[self.__indexes0[0]].tan
        else:
            tan0 = manifolds[self.__indexes0[0]][self.__indexes0[1]].tan        
        
        if len(self.__indexes1) == 1:
            tan1 = manifolds[self.__indexes1[0]].tan
        else:
            tan1 = manifolds[self.__indexes1[0]][self.__indexes1[1]].tan   
       
        #print('tan0', tan0)
        #print('tan1', tan1)
        return torch.sum((tan0*normals0 - tan1*normals1), 1).view(-1, 1)
    
    @property
    def dimconstraint(self):
        return self.__dimconstraint
    
    def adjoint(self, lam, manifold):
        normals0, normals1 = self.compute_normals
        
        man = manifold.clone()
        shape = man[self.__indexes0[0]][self.__indexes0[1]].cotan.shape
        man.fill_cotan_zeros()
        man[self.__indexes0[0]][self.__indexes0[1]].fill_cotan(lam.view([-1, 1]) * normals0)
        man[self.__indexes1[0]][self.__indexes1[1]].fill_cotan(-lam.view([-1, 1]) * normals1)
        return man
    
    def matrixAMAs(self, M, manifolds):
        """
        returns A_q M A_q^\ast (corresponds to extracting sub matrices of M and sum/substract : 
        M_11 -M_21 -M_12 + M22)
        """
        
        c0 = sum([sum(man.numel_gd) for man in manifolds[:self.__indexes0[0]]])
        
        if len(self.__indexes0) == 1:
            c1 = 0
            d0 = sum(manifolds[self.__indexes0[0]].numel_gd)
            c2 = 0
        else:
            c1 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes0[0]][:self.__indexes0[1]]])
            c2 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes0[0]][self.__indexes0[1]+1:]])
            d0 = sum(manifolds[self.__indexes0[0]][self.__indexes0[1]].numel_gd)
        
        c3 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes0[0] + 1:self.__indexes1[0]]])
        
        if len(self.__indexes1) == 1:
            c4 = 0
            c5 = 0
            d1 = sum(manifolds[self.__indexes1[0]].numel_gd)
        else:
            c4 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes1[0]][:self.__indexes1[1]]])
            c5 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes1[0]][self.__indexes1[1]+1:]])
            d1 = sum(manifolds[self.__indexes1[0]][self.__indexes1[1]].numel_gd)
        
        c6 = sum([sum(man.numel_gd) for man in manifolds[self.__indexes1[0]:]])
        assert (d0==d1)
        
        ind0 = c0 + c1
        ind1 = c0 + c1 + d0 + c2 + c3 + c4
        
        normals0, normals1 = self.compute_normals
        n, d = normals0.shape
        
        M00 = M[ind0:ind0 + d0, ind0:ind0 + d0].reshape(n,d,n,d).transpose(1,2)
        S00 = torch.einsum('ij, ikjl, kl -> ik', normals0, M00, normals0)
        
        M01 = M[ind0:ind0 + d0, ind1:ind1+d1].reshape(n,d,n,d).transpose(1,2)
        S01 = torch.einsum('ij, ikjl, kl -> ik', normals0, M00, normals1)
        
        M10 = M[ind1:ind1+d1, ind0:ind0 + d0].reshape(n,d,n,d).transpose(1,2)
        S10 = torch.einsum('ij, ikjl, kl -> ik', normals1, M00, normals0)
        
        M11 = M[ind1:ind1+d1, ind1:ind1+d1].reshape(n,d,n,d).transpose(1,2)
        S11 = torch.einsum('ij, ikjl, kl -> ik', normals1, M00, normals1)
        
        return S00 -S01-S10 + S11
        
   
        
class CompoundConstraints(ConstraintsPointIdentityBase):
    """
    concatenates constraints
    """
    def __init__(self, constraints):
            self.__constraints = constraints
            
    def __call__(self, manifolds):
        return torch.cat([constraint(manifolds) for constraint in self.__constraints])
       
    @property
    def dimconstraint(self):
        return sum([constraint.dimconstraint for constraint in self.__constraints])
    
    def adjoint(self, lam, manifold):
        man = manifold.clone()
        man.fill_cotan_zeros()
        ind = 0
        for constraint in self.__constraints:
            dim = constraint.dimconstraint
            man.add_cotan(constraint.adjoint(lam[ind:ind + dim], manifold).cotan)
            ind = ind + dim
        return man
        

    def matrixAMAs(self, M, manifolds):   
        #mat = torch.zeros(self.dimconstraint, self.dimconstraint, requires_grad=True)
        mat = self.__constraints[0].matrixAMAs(M, manifolds)
        ind = mat.shape[0]
        
        for i in range(1, len(self.__constraints)):
            dim = self.__constraints[i].dimconstraint
            mat = torch.cat([torch.cat([mat, torch.zeros(ind, dim)], dim=1), torch.cat([torch.zeros(dim, ind),self.__constraints[i].matrixAMAs(M, manifolds)], dim=1)])
            ind = ind + dim
        return mat
        
        
        
        
