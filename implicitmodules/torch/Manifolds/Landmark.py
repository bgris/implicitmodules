import torch

from implicitmodules.torch.Manifolds.Abstract import Manifold
from implicitmodules.torch.StructuredFields import StructuredField_0


class Landmarks(Manifold):
    def __init__(self, dim, nb_pts, gd=None, tan=None, cotan=None):
        assert (gd is None) or (gd.shape[0] == dim * nb_pts)
        assert (tan is None) or (tan.shape[0] == dim * nb_pts)
        assert (cotan is None) or (cotan.shape[0] == dim * nb_pts)
        super().__init__()

        self.__nb_pts = nb_pts
        self.__dim = dim
        self.__numel_gd = nb_pts * dim

        self.__gd = torch.zeros(nb_pts, dim, requires_grad=True).view(-1)
        if isinstance(gd, torch.Tensor):
            self.fill_gd(gd.requires_grad_(), copy=False)

        self.__tan = torch.zeros(nb_pts, dim, requires_grad=True).view(-1)
        if isinstance(tan, torch.Tensor):
            self.fill_tan(tan.requires_grad_(), copy=False)

        self.__cotan = torch.zeros(nb_pts, dim, requires_grad=True).view(-1)
        if isinstance(cotan, torch.Tensor):
            self.fill_cotan(cotan.requires_grad_(), copy=False)

    def copy(self):
        out = Landmarks(self.__dim, self.__nb_pts)
        out.fill(self, copy=True)
        return out

    def move_to(self, device):
        with torch.autograd.no_grad():
            self.__gd = self.__gd.to(device).requires_grad_()
            self.__tan = self.__tan.to(device).requires_grad_()
            self.__cotan = self.__cotan.to(device).requires_grad_()

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def dim(self):
        return self.__dim

    @property
    def numel_gd(self):
        return self.__numel_gd

    @property
    def len_gd(self):
        return 1

    @property
    def dim_gd(self):
        return (self.__numel_gd,)

    def unroll_gd(self):
        return [self.__gd]

    def unroll_tan(self):
        return [self.__tan]

    def unroll_cotan(self):
        return [self.__cotan]

    def roll_gd(self, l):
        return l.pop(0)

    def roll_tan(self, l):
        return l.pop(0)

    def roll_cotan(self, l):
        return l.pop(0)

    def __get_gd(self):
        return self.__gd

    def __get_tan(self):
        return self.__tan

    def __get_cotan(self):
        return self.__cotan

    def fill(self, manifold, copy=False):
        assert isinstance(manifold, Landmarks)
        self.fill_gd(manifold.gd, copy=copy)
        self.fill_tan(manifold.tan, copy=copy)
        self.fill_cotan(manifold.cotan, copy=copy)

    def fill_gd(self, gd, copy=False):
        assert gd.shape[0] == self.__numel_gd
        if copy:
            self.__gd = gd.detach().clone().requires_grad_()
        else:
            self.__gd = gd

    def fill_tan(self, tan, copy=False):
        assert tan.shape[0] == self.__numel_gd
        if copy:
            self.__tan = tan.detach().clone().requires_grad_()
        else:
            self.__tan = tan

    def fill_cotan(self, cotan, copy=False):
        assert cotan.shape[0] == self.__numel_gd
        if copy:
            self.__cotan = cotan.detach().clone().requires_grad_()
        else:
            self.__cotan = cotan

    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def muladd_gd(self, gd, scale):
        if isinstance(gd, torch.Tensor):
            self.__gd = self.__gd + scale * gd
        else:
            self.__gd = self.__gd + scale * gd.__gd

    def muladd_tan(self, tan, scale):
        if isinstance(tan, torch.Tensor):
            self.__tan = self.__tan + scale * tan
        else:
            self.__tan = self.__tan + scale * tan.__tan

    def muladd_cotan(self, cotan, scale):
        if isinstance(cotan, torch.Tensor):
            self.__cotan = self.__cotan + scale * cotan
        else:
            self.__cotan = self.__cotan + scale * cotan.__cotan

    def negate_gd(self):
        self.__gd = -self.__gd

    def negate_tan(self):
        self.__tan = -self.__tan

    def negate_cotan(self):
        self.__cotan = -self.__cotan

    def inner_prod_field(self, field):
        man = self.infinitesimal_action(field)
        return torch.dot(self.__cotan.view(-1), man.tan.view(-1))

    def infinitesimal_action(self, field):
        """Applies the vector field generated by the module on the landmark."""
        tan = field(self.__gd.view(-1, self.__dim)).view(-1)
        return Landmarks(self.__dim, self.__nb_pts, gd=self.__gd, tan=tan)

    def cot_to_vs(self, sigma):
        return StructuredField_0(self.__gd.view(-1, self.__dim),
                                 self.__cotan.view(-1, self.__dim), sigma)

