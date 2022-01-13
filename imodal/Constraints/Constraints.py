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
        ind = 0
        ind_tot = 0
        for i in range(self.__index):
            ind = ind + manifolds.manifolds[i][-1].gd.shape[0]
            ind_tot = ind_tot + manifolds.manifolds[i][-1].numel_gd[0]
            
        #indexes_manifolds1 = [len(manifolds.manifolds) - 1 , self.__index]
        indexes_manifolds1 = [len(manifolds.manifolds) - 1 , ind, ind + manifolds.manifolds[self.__index][-1].gd.shape[0], ind_tot, ind_tot + manifolds.manifolds[self.__index][-1].numel_gd[0]]
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
            tan1 = manifolds[indexes_manifolds1[0]][0].tan[indexes_manifolds1[1]:indexes_manifolds1[2]]
       
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
        lam_ext = torch.zeros_like(man[indexes_manifolds1[0]][0].cotan).view(-1)
        lam_ext[indexes_manifolds1[3]:indexes_manifolds1[4]] = lam.view(-1)
        man[indexes_manifolds0[0]][indexes_manifolds0[1]].fill_cotan(lam.view(shape))
        man[indexes_manifolds1[0]][0].fill_cotan(-lam_ext.view(man[indexes_manifolds1[0]][0].cotan.shape))
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
            #ERROR
            c4 = 0
            c5 = 0
            d1 = sum(manifolds[indexes_manifolds1[0]].numel_gd)
        else:
            #c4 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds1[0]][:indexes_manifolds1[1]]])
            #c5 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds1[0]][indexes_manifolds1[1]+1:]])
            c4 = indexes_manifolds1[3]
            
            #d1 = sum(manifolds[indexes_manifolds1[0]][indexes_manifolds1[1]].numel_gd)
            d1 = indexes_manifolds1[4] - c4
        
        c6 = sum([sum(man.numel_gd) for man in manifolds[indexes_manifolds1[0]:]])
        #print(d0)
        #print(d1)
        assert (d0==d1)
        
        ind0 = c0 + c1
        ind1 = c0 + c1 + d0 + c2 + c3 + c4
        return M[ind0:ind0 + d0, ind0:ind0 + d0] -M[ind1:ind1+d1, ind0:ind0 + d0]-M[ind1:ind1+d1, ind0:ind0 + d0] + M[ind1:ind1+d1, ind1:ind1+d1]

        
class ConstraintsPointIdentity(ConstraintsPointIdentityBase):
    """
    Identity between two specified modules (by an index) on a specified boundary (by an index)
    """
    
    
        
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
        
        
        
        