import torch

from implicitmodules.torch.Manifolds.Abstract import Manifold
from implicitmodules.torch.StructuredFields import StructuredField_0


class Landmarks(Manifold):
    def __init__(self, dim, nb_pts, gd=None, tan=None, cotan=None, device=None):
        assert (gd is None) or ((gd.shape[0] == nb_pts) and (gd.shape[1] == dim))
        assert (tan is None) or ((tan.shape[0] == nb_pts) and (tan.shape[1] == dim))
        assert (cotan is None) or ((cotan.shape[0] == nb_pts) and (cotan.shape[1] == dim))

        super().__init__(((dim,),), nb_pts, gd, tan, cotan, device=device)

        self.__dim = dim

    @property
    def dim(self):
        return self.__dim

    def inner_prod_field(self, field):
        man = self.infinitesimal_action(field)
        return torch.dot(self.cotan.flatten(), man.tan.flatten())

    def infinitesimal_action(self, field):
        """Applies the vector field generated by the module on the landmark."""
        tan = field(self.gd)
        return Landmarks(self.__dim, self.nb_pts, gd=self.gd, tan=tan, device=self.device)

    def cot_to_vs(self, sigma, backend=None):
        return StructuredField_0(self.gd, self.cotan, sigma, device=self.device, backend=backend)

