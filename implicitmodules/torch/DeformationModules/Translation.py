import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_0


class Translations(DeformationModule):
    """Module generating sum of translations."""
    
    def __init__(self, manifold, sigma):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__dim_controls = self.__manifold.dim * self.__manifold.nb_pts
        self.__controls = torch.zeros(self.__dim_controls, requires_grad=True)
    
    @classmethod
    def build_from_points(cls, dim, nb_pts, sigma, gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma)

    def move_to(self, device):
        self.__manifold.move_to(device)

        self.__controls = self.__controls.to(device)
    
    @property
    def manifold(self):
        return self.__manifold
    
    @property
    def sigma(self):
        return self.__sigma
    
    @property
    def dim_controls(self):
        return self.__dim_controls
    
    def __get_controls(self):
        return self.__controls
    
    def fill_controls(self, controls):
        self.__controls = controls
    
    controls = property(__get_controls, fill_controls)
    
    def fill_controls_zero(self):
        self.__controls = torch.zeros(self.__dim_controls)
    
    def __call__(self, points):
        """Applies the generated vector field on given points."""
        K_q = K_xy(points, self.__manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        return torch.mm(K_q, self.__controls.view(-1, self.__manifold.dim))
    
    def cost(self):
        """Returns the cost."""
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        m = torch.mm(K_q, self.__controls.view(-1, self.__manifold.dim))
        return 0.5 * torch.dot(m.view(-1), self.__controls.view(-1))
    
    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        controls, _ = torch.solve(vs(self.manifold.gd.view(-1, self.manifold.dim)), K_q)
        self.__controls = controls.contiguous().view(-1)
    
    def field_generator(self):
        return StructuredField_0(self.__manifold.gd.view(-1, self.__manifold.dim),
                                 self.__controls.view(-1, self.__manifold.dim), self.__sigma)
    
    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)
