from typing import Iterable

import torch

from implicitmodules.torch.Manifolds.Abstract import Manifold
from implicitmodules.torch.StructuredFields import StructuredField_m, StructuredField_0
from implicitmodules.torch.StructuredFields.Abstract import SumStructuredField


class Stiefel(Manifold):
    def __init__(self, dim, nb_pts, gd=None, tan=None, cotan=None, device=None):
        assert (gd is None) or ((gd[0].shape[0] == nb_pts) and (gd[0].shape[1] == dim) and\
                                (gd[1].shape[0] == nb_pts) and (gd[1].shape[1] == dim) and\
                                (gd[1].shape[2] == dim))
        assert (tan is None) or ((tan[0].shape[0] == nb_pts) and (tan[0].shape[1] == dim) and\
                                 (tan[1].shape[0] == nb_pts) and (tan[1].shape[1] == dim) and\
                                 (tan[1].shape[2] == dim))
        assert (cotan is None) or ((cotan[0].shape[0] == nb_pts) and (cotan[0].shape[1] == dim) and\
                                   (cotan[1].shape[0] == nb_pts) and (cotan[1].shape[1] == dim) and\
                                   (cotan[1].shape[2] == dim))

        super().__init__(((dim,), (dim, dim)), nb_pts, gd, tan, cotan, device=device)

        self.__dim = dim

    @property
    def dim(self):
        return self.__dim

    def inner_prod_field(self, field):
        man = self.infinitesimal_action(field)
        return torch.dot(self.cotan[0].flatten(), man.tan[0].flatten()) + \
            torch.einsum('nij, nij->', self.cotan[1], man.tan[1])

    def infinitesimal_action(self, field):
        """Applies the vector field generated by the module on the landmark."""
        vx = field(self.gd[0])
        d_vx = field(self.gd[0], k=1)

        S = 0.5 * (d_vx - torch.transpose(d_vx, 1, 2))
        vr = torch.bmm(S, self.gd[1])

        return Stiefel(self.__dim, self.nb_pts, gd=self.gd, tan=(vx, vr))

    def cot_to_vs(self, sigma, backend=None):
        v0 = StructuredField_0(self.gd[0], self.cotan[0], sigma, backend=backend)
        R = torch.einsum('nik, njk->nij', self.cotan[1], self.gd[1])

        vm = StructuredField_m(self.gd[0], R, sigma, backend=backend)

        return SumStructuredField([v0, vm])

