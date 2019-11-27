import torch

from implicitmodules.torch.Manifolds.Abstract import Manifold
from implicitmodules.torch.StructuredFields import StructuredField_0


class Landmarks(Manifold):
    def __init__(self, dim, nb_pts, gd=None, tan=None, cotan=None, device=None):
        assert (gd is None) or (gd.shape == torch.Size([nb_pts, dim]))
        assert (tan is None) or (tan.shape == torch.Size([nb_pts, dim]))
        assert (cotan is None) or (cotan.shape == torch.Size([nb_pts, dim]))
        super().__init__()

        self.__device = self.__find_device(gd, tan, cotan, device)

        self.__nb_pts = nb_pts
        self.__dim = dim
        self.__numel_gd = nb_pts * dim
        self.__point_shape = torch.Size([self.__nb_pts, self.__dim])

        self.__gd = torch.zeros(self.__nb_pts, self.__dim, requires_grad=True, device=self.__device)
        if isinstance(gd, torch.Tensor):
            self.fill_gd(gd, copy=False)

        self.__tan = torch.zeros(self.__nb_pts, self.__dim, requires_grad=True, device=self.__device)
        if isinstance(tan, torch.Tensor):
            self.fill_tan(tan, copy=False)

        self.__cotan = torch.zeros(self.__nb_pts, self.__dim, requires_grad=True, device=self.__device)
        if isinstance(cotan, torch.Tensor):
            self.fill_cotan(cotan, copy=False)

    def __find_device(self, gd, tan, cotan, device):
        if device is None:
            # Device is not specified, we need to get it from the tensors
            cur_device = None
            if gd is not None:
                cur_device = gd.device
            elif tan is not None:
                cur_device = tan.device
            elif cotan is not None:
                cur_device = cotan.device
            else:
                return None

            # We now compare the device with the other tensors and see if it corresponds
            if ((gd is not None) and gd.device != cur_device):
                raise RuntimeError("Landmarks.__init__(): gd is not on device" + str(device))
            if ((tan is not None) and tan.device != cur_device):
                raise RuntimeError("Landmarks.__init__(): tan is not on device" + str(device))
            if ((cotan is not None) and cotan.device != cur_device):
                raise RuntimeError("Landmarks.__init__(): cotan is not on device" + str(device))

            return cur_device

        else:
            if gd is not None:
                gd.to(device=device)
            if tan is not None:
                tan.to(device=device)
            if cotan is not None:
                cotan.to(device=device)

            return device

    def to(self, device):
        self.__device = device
        self.__gd.to(device)
        self.__tan.to(device)
        self.__cotan.to(device)

    @property
    def dtype(self):
        return self.gd.dtype

    @property
    def device(self):
        return self.__device

    def copy(self, requires_grad=True):
        out = Landmarks(self.__dim, self.__nb_pts)
        out.fill(self, copy=True, requires_grad=requires_grad)
        return out

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def dim(self):
        return self.__dim

    @property
    def numel_gd(self):
        return (self.__numel_gd,)

    @property
    def shape_gd(self):
        return (self.__gd.shape,)

    @property
    def len_gd(self):
        return 1

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

    def fill(self, manifold, copy=False, requires_grad=True):
        assert isinstance(manifold, Landmarks)
        self.fill_gd(manifold.gd, copy=copy, requires_grad=requires_grad)
        self.fill_tan(manifold.tan, copy=copy, requires_grad=requires_grad)
        self.fill_cotan(manifold.cotan, copy=copy, requires_grad=requires_grad)

    def fill_gd(self, gd, copy=False, requires_grad=True):
        assert gd.shape == self.__point_shape
        if copy:
            self.__gd = gd.detach().clone().requires_grad_(requires_grad)
        else:
            self.__gd = gd

    def fill_tan(self, tan, copy=False, requires_grad=True):
        assert tan.shape == self.__point_shape
        if copy:
            self.__tan = tan.detach().clone().requires_grad_(requires_grad)
        else:
            self.__tan = tan

    def fill_cotan(self, cotan, copy=False, requires_grad=True):
        assert cotan.shape == self.__point_shape
        if copy:
            self.__cotan = cotan.detach().clone().requires_grad_(requires_grad)
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
        return torch.dot(self.__cotan.flatten(), man.tan.flatten())

    def infinitesimal_action(self, field):
        """Applies the vector field generated by the module on the landmark."""
        tan = field(self.__gd)
        return Landmarks(self.__dim, self.__nb_pts, gd=self.__gd, tan=tan, device=self.device)

    def cot_to_vs(self, sigma, backend=None):
        return StructuredField_0(self.__gd, self.__cotan, sigma, device=self.device, backend=backend)

