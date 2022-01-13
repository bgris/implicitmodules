import torch

from imodal.MultiShape import MultiFields
from imodal.MultiShape import MultishapeCompoundManifold
from imodal.DeformationModules import CompoundModule
from imodal.Manifolds import CompoundManifold

class MultiShapeModules:
    def __init__(self, module_list, sigma_background, backgroundtype=None):
        """
        module_list is a list of compound modules
        
        """
        self.__nb_shapes = len(module_list)
        self.__sigma_background = sigma_background
        self.__modules = [*module_list]
        self.__nb_modules = len(self.__modules)
        self.__manifold = MultishapeCompoundManifold.MultishapeCompoundManifold([mods.manifold for mods in module_list])
        #print([[man.numel_gd for man in mans] for mans in self.__manifold])
        #self.__list_gd_dim = [sum([man.numel_gd for man in mans]) for mans in self.__manifold]
        self.__list_gd_dim = [sum(man.numel_gd) for man in self.__manifold]
        self.__dim_tot_gd = sum(self.__list_gd_dim)
        self.__backgroundtype = backgroundtype
    
    def copy(self):
        return Multishape([mod.copy() for mod in self.__module_list], self.__sigma_background)
    
    def __get_lam(self):
        return self.__lam
    
    def fill_lam(self, lam, copy=False):
        #assert len(l) == self.nb_module
        if copy:
            self.__lam = lam.detach().clone().requires_grad_()
        else:
            self.__lam = lam
        
    lam = property(__get_lam, fill_lam)
    
    @property
    def sigma_background(self):
        return self.__sigma_background
    
    @property
    def modules(self):
        return self.__modules
    
    @property
    def backgroundtype(self):
        return self.__backgroundtype
    
    @property
    def manifold(self):
        return self.__manifold

    def __getitem__(self, itemid):
        if isinstance(itemid, int) or isinstance(itemid, slice):
            return self.__modules[itemid]
        else:
            return self.todict()[itemid]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.__modules):
            raise StopIteration
        else:
            self.current = self.current + 1
            return self.__modules[self.current - 1]

        
    def compute_geodesic_control_from_self(self):
        """
        manifolds is a MultishapeCompoundManifold
        """
        
        for mod, man in zip(self.__modules, self.__manifold):
            mod.compute_geodesic_control(man)
            
    def compute_geodesic_control(self, manifolds):
        """
        manifolds is a MultishapeCompoundManifold
        """
        
        for mod, man in zip(self.__modules, manifolds):
            mod.compute_geodesic_control(man)
            
    
    def field_generator(self):
        return MultiFields.MultiFields([mod.field_generator() for mod in self.__modules])    
    
    
    def compute_geodesic_variables(self, constraints):
        self.compute_geodesic_control_from_self()
        
        a = self.__manifold.infinitesimal_action(self.field_generator())
        constr = constraints(a)
        
        M = self.autoaction()
        mat = constraints.matrixAMAs(M, self.__manifold)
        self.__lam, _ = torch.solve(constr, mat)
                
        man = constraints.adjoint(self.__lam, self.__manifold)
        
        
        
        man.negate_cotan()
        
        
        man.add_cotan(self.__manifold.cotan)

       
        self.compute_geodesic_control(man)
        
        

    def cost(self):
        """Returns the cost."""
        return sum([module.cost() for module in self.__modules])
        
        
    def autoaction(self):
        mat = torch.zeros([self.__dim_tot_gd, self.__dim_tot_gd])
        c = 0
        for i in range(self.__nb_modules - 1):
            dim_gd = self.__list_gd_dim[i]
            A = self.__modules[i].autoaction()
            mat[c:c+dim_gd, c:c+dim_gd] = A
            c = c + dim_gd
        
        # bad fix to avoid extra computation for background module
        if self.__backgroundtype is None:
            i = self.__nb_modules - 1
            dim_gd = self.__list_gd_dim[i]
            A = self.__modules[i].autoaction()
            mat[c:c+dim_gd, c:c+dim_gd] = A
            c = c + dim_gd    
        else:
            i = self.__nb_modules - 1
            A = self.__modules[i][0].autoaction()
            dim_gd = A.shape[0]
            mat[c:c+dim_gd, c:c+dim_gd] = A
            c = c + dim_gd
        
        
        return mat
    
    
    
    
    
    
    
    
    
    