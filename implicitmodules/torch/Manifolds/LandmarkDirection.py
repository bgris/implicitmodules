import torch

from implicitmodules.torch.Manifolds.Abstract import Manifold
from implicitmodules.torch.StructuredFields import StructuredField_0, StructuredField_0_u, SumStructuredField


class LandmarksDirection(Manifold):
    def __init__(self, dim, nb_pts, transport, gd=None, tan=None, cotan=None, device=None):
        assert (gd is None) or ((gd[0].shape[0] == nb_pts) and (gd[0].shape[1] == dim) and\
                                (gd[1].shape[0] == nb_pts) and (gd[1].shape[1] == dim))
        assert (tan is None) or ((tan[0].shape[0] == nb_pts) and (tan[0].shape[1] == dim) and\
                                 (tan[1].shape[0] == nb_pts) and (tan[1].shape[1] == dim))
        assert (cotan is None) or ((cotan[0].shape[0] == nb_pts) and (cotan[0].shape[1] == dim) and\
                                   (cotan[1].shape[0] == nb_pts) and (cotan[1].shape[1] == dim))
        super().__init__(((dim,), (dim,)), nb_pts, gd, tan, cotan)

        self.to_(device=device)
        self.__dim = dim

        if transport != 'vector' and transport != 'orthogonal':
            raise RuntimeError("LandmarksDirection.__init__(): transport mode {transport} not recognised!".format(transport=transport))

        self.__transport = transport

    @property
    def dim(self):
        return self.__dim

    @property
    def transport(self):
        return self.__transport

    def inner_prod_field(self, field):
        man = self.infinitesimal_action(field)
        return torch.dot(self.cotan[0].flatten(), man.tan[0].flatten()) +\
            torch.dot(self.cotan[1].flatten(), man.tan[1].flatten())

    def infinitesimal_action(self, field):
        """Applies the vector field generated by the module on the landmark."""
        tan_landmarks = field(self.gd[0])
        if self.__transport == 'vector':
            tan_directions = torch.bmm(field(self.gd[0], k=1), self.gd[1].unsqueeze(2))
        else:
            tan_directions = -torch.bmm(field(self.gd[0], k=1).transpose(1, 2), self.gd[1].unsqueeze(2))

        return LandmarksDirection(self.__dim, self.nb_pts, transport=self.__transport, gd=self.gd, tan=(tan_landmarks, tan_directions), device=self.device)

    def cot_to_vs(self, sigma, backend=None):
        v0 = StructuredField_0(self.gd[0], self.cotan[0], sigma, device=self.device, backend=backend)

        if self.__transport == 'vector':
            vu = StructuredField_0_u(self.gd[0], self.cotan[1], self.gd[1], sigma)
        else:
            vu = StructuredField_0_u(self.gd[0], -self.gd[1], self.cotan[1], sigma)

        return SumStructuredField([v0, vu])

