from typing import Iterable
import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Manifolds import CompoundManifold
from implicitmodules.torch.StructuredFields import SumStructuredField


class CompoundModule(DeformationModule, Iterable):
    """ Combination of deformation modules. """

    """ Compound module constructor.

    Parameters
    ----------
    modules : Iterable
        Iterable of deformation modules we want to build the compound module from.
    label :
        Optional identifier
    """
    def __init__(self, modules, label=None):
        assert isinstance(modules, Iterable)
        super().__init__(label)
        self.__modules = [*modules]

    def __str__(self):
        outstr = "Compound Module\n"
        if self.label:
            outstr += "  Label=" + self.label + "\n"
        outstr += "Modules=\n"
        for module in self.__modules:
            outstr += "*"*20
            outstr += str(modules) + "\n"
        outstr += "*"*20
        return outstr

    def to(self, *args, **kwargs):
        [mod.to(*args, **kwargs) for mod in self.__modules]

    @property
    def device(self):
        return self.__modules[0].device

    @property
    def modules(self):
        return self.__modules

    def todict(self):
        return dict(zip(self.label, self.__modules))

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

    @property
    def dim(self):
        return self.__modules[0].dim # Dirty

    @property
    def label(self):
        return [module.label for module in self.__modules]

    def __get_controls(self):
        return [m.controls for m in self.__modules]

    def fill_controls(self, controls):
        #assert len(controls) == len(self.__modules)
        #[module.fill_controls(control) for module, control in zip(self.__modules, controls)]
        #assert len(controls) == self.nb_module
        for i in range(len(controls)):
            self.__modules[i].fill_controls(controls[i])


    controls = property(__get_controls, fill_controls)

    @property
    def dim_cont(self):
        return sum([m.dim_cont for m in self.__modules])

    def fill_controls_zero(self):
        [module.fill_controls_zero() for module in self.__modules]

    @property
    def manifold(self):
        return CompoundManifold([m.manifold for m in self.__modules])

    def __call__(self, points):
        """Applies the generated vector field on given points."""
        return sum([module(points) for module in self.__modules])

    def cost(self):
        """Returns the cost."""
        return sum([module.cost() for module in self.__modules])

    def compute_geodesic_control(self, man):
        """Computes geodesic control from \delta \in H^\ast."""
        [module.compute_geodesic_control(man) for module in self.__modules]

    def field_generator(self):
        return SumStructuredField([m.field_generator() for m in self.__modules])

    def costop_inv(self):
        # blockdiagonal matrix of inverse cost operators of each module
        Z = torch.zeros(self.dim_cont, self.dim_cont)
        n = 0
        for m in self.__modules:
            ni = m.dim_cont
            Z[n:n+ni, n:n+ni] = m.costop_inv().contiguous()
            n = n + ni
        return Z
        
    def autoaction(self):
        # count non silent modules
        #c = 0
        #ind = 0
        #for mod in self.__modules:
        #    if (isinstance(mod, dm.DeformationModules.SilentLandmark.SilentBase)==False):
        #        c = c + 1
        
        if len(self.__modules)==1:
            A =  self.__modules[0].autoaction()
        else:
            #TODO: seems to be an error in gradient 
            dimgd = sum(self.manifold.numel_gd)
            actionmat = torch.zeros(dimgd, self.dim_cont)
            tmp = 0
            controls = self.controls
            for m in range(len(self.__modules)):
                for i in range(self.__modules[m].dim_cont):
                    self.fill_controls_zero()
                    c = torch.eye(self.__modules[m].dim_cont, requires_grad=True)[i].view(self.modules[m].controls.shape)
                    self.modules[m].fill_controls(c)
                    speed = self.manifold.infinitesimal_action(self.__modules[m].field_generator())
                    a = torch.cat([s.view(-1) for s in speed.tan])
                    actionmat[:,tmp+i] = a 
                tmp = tmp + self.__modules[m].dim_cont
            
            self.fill_controls(controls)
            A = torch.mm(actionmat, torch.mm(self.costop_inv(), torch.transpose(actionmat, 0,1)))

        return A
    
