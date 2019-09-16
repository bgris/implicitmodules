import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_0


class ImplicitModule0(DeformationModule):
    """Module generating sum of translations."""

    def __init__(self, manifold, sigma, nu, coeff=1.):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__dim_controls = self.__manifold.dim * self.__manifold.nb_pts
        self.__controls = torch.zeros(self.__dim_controls)

    @classmethod
    def build_from_points(cls, dim, nb_pts, sigma, nu, coeff=1., gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, nu, coeff)

    @property
    def manifold(self):
        return self.__manifold

    @property
    def sigma(self):
        return self.__sigma

    @property
    def nu(self):
        return self.__nu

    @property
    def coeff(self):
        return self.__coeff

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
        K_q = K_xy(points, self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        return torch.mm(K_q, self.__controls.view(-1, self.__manifold.dim))

    def cost(self):
        """Returns the cost."""
        
        # SKS order 0:
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        m = torch.mm(K_q + self.__nu * torch.eye(self.__manifold.nb_pts), self.__controls.view(-1, self.__manifold.dim))
        return 0.5 * self.__coeff * torch.dot(m.view(-1), self.__controls.view(-1))

    def compute_geodesic_control(self, man):
        r"""Computes geodesic control from \delta \in H^\ast."""

        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma) + self.__nu * torch.eye(
            self.__manifold.nb_pts)
        controls, _ = torch.solve(vs(self.manifold.gd.view(-1, self.manifold.dim)), K_q)
        self.__controls = controls.contiguous().view(-1) / self.__coeff

        #  Keops
        from pykeops.torch import KernelSolve
        formula = 'Exp(-p * SqDist(x, y)) * b'
        aliases = ['x = Vi(' + str(self.__manifold.dim) + ')',
                   'y = Vj(' + str(self.__manifold.dim) + ')',
                   'b = Vj(' + str(self.__manifold.dim) + ')',
                   'p = Pm(1)']
        Kinv = KernelSolve(formula, aliases, "b", axis=1)
        x = self.manifold.gd.view(-1, self.__manifold.dim)
        b = vs(self.manifold.gd.view(-1, self.manifold.dim))
        g = torch.tensor([.5 / self.sigma / self.sigma])

        d = controls.contiguous().view(-1) / self.__coeff
        c = Kinv(x, x, b, g, alpha=self.__nu).view(-1) / self.__coeff

        if not torch.allclose(c, d, atol=1e-6, rtol=1e-4):
            raise RuntimeError('ElasticOrder0 module [KeOps] KernelSolve bad conditionned')
        # end Keops

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd.view(-1, self.__manifold.dim),
                                 self.__controls.view(-1, self.__manifold.dim), self.__sigma)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)

