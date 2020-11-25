from collections import Iterable
import torch

from implicitmodules.torch.DeformationModules.Combination import CompoundModule


class Hamiltonian_multishape:
    """Class used to represent the hamiltonian given by a collection of modules."""
    def __init__(self, modules, constraints):
        """
        Instantiate the Hamiltonian related to a set of deformation module.

        Parameters
        ----------
        modules : Iterable or DeformationModules.DeformationModule
            Either an iterable of deformation modules or an unique module.
        """
        self.__modules = modules
        self.__constraints = constraints

    @classmethod
    def from_hamiltonian(cls, class_instance):
        return cls(class_instance.module)

    @property
    def module(self):
        return self.__modules

    @property
    def dim(self):
        return self.__module.dim

    def __call__(self):
        """Computes the hamiltonian.

        Mathematicaly, computes the quantity :math:`\mathcal{H}(q, p, h)`.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the hamiltonian quantity.
        """
        return self._apply_mom() - self.__modules.cost() - self.apply_constr()

    def geodesic_controls(self):
        """
        Computes the geodesic controls of the hamiltonian's module.
        """
        self.__modules.compute_geodesic_variables(self.__constraints)
        print('print constraints')
        print(self.__constraints(self.__modules.manifold.infinitesimal_action(self.__modules.field_generator())))
        print('print constraints done')

    def _apply_mom(self):
        """Apply the moment on the geodesic descriptors."""
        return sum([self.__modules.manifold.manifolds[i].inner_prod_field(self.__modules[i].field_generator()) for i in range(len(self.__modules.modules))])

    def apply_constr(self):
        return torch.tensordot( self.__modules.lam, self.__constraints(self.__modules.manifold.infinitesimal_action(self.__modules.field_generator())))
