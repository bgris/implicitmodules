import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.DeformationModules.Translation import Translations
from implicitmodules.torch.Manifolds import Landmarks

from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.StructuredFields import StructuredField_0

class ConstrainedTranslations(DeformationModule):
    """Module generating a local field via a sum of translations."""
    
    def __init__(self, manifold, support_generator, vector_generator, sigma, coeff=1):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__supportgen = support_generator
        self.__vectorgen = vector_generator
        self.__sigma = sigma
        self.__dim_controls = 1
        self.__controls = torch.zeros(self.__dim_controls, requires_grad=True)
        self.__coeff = coeff
        a = torch.sqrt(torch.tensor(3.))
        self.__direc_scaling_pts = torch.tensor([[1., 0.], [-0.5 , 0.5* a],  [-0.5, -0.5* a]], requires_grad=True)
        self.__direc_scaling_vec = torch.tensor([[1., 0.], [-0.5 , 0.5* a],  [-0.5, -0.5* a]], requires_grad=True)

    
    @classmethod
    def build_from_points(cls, dim, nb_pts, sigma, gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma)
    
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
        gd = self.__manifold.gd.view(-1, 2)
        pts = self.__supportgen(gd)
        #cont = self.__controls * self.__vectorgen(gd)
        
        #pts = self.__manifold.gd.view(-1, 2)
        cont = self.__controls * self.__vectorgen(gd)
        
        
        #manifold_Landmark = Landmarks(self.__manifold.dim, self.__manifold.dim + 1, gd=self.__supportgen(gd).view(-1))
        #Trans = Translations(manifold_Landmark, self.__sigma)
        #Trans.fill_controls(self.__controls * self.__vectorgen(gd))
        K_q = K_xy(points, pts, self.__sigma)
        return torch.mm(K_q, cont)
        
    
    def cost(self):
        """Returns the cost."""
        gd = self.__manifold.gd.view(-1, 2)
        pts = self.__supportgen(gd)
        #cont = self.__controls * self.__vectorgen(gd)
        
        
        #pts = self.__manifold.gd.view(-1, 2)
        cont = self.__controls * self.__vectorgen(gd)
        
        
        
        K_q = K_xx(pts, self.__sigma)
        m = torch.mm(K_q, cont)
        #manifold_Landmark = Landmarks(self.__manifold.dim, self.__manifold.dim + 1, gd=self.__supportgen(gd).view(-1))
        #Trans = Translations(manifold_Landmark, self.__sigma)
        #Trans.fill_controls(self.__controls * self.__vectorgen(gd))
        return  0.5 * torch.dot(m.view(-1), cont.view(-1))
        #return self.__coeff * Trans.cost()
    
    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        #self.__controls = torch.tensor(1., dtype=self.__manifold.gd.dtype, requires_grad=True)
        #cost_1 = self.cost()
        #v = self.field_generator()
        gd = self.__manifold.gd.view(-1, 2)
        pts = self.__supportgen(gd)
        v = StructuredField_0(pts,
                                 self.__vectorgen(gd), self.__sigma)
        apply = man.inner_prod_field(v)
        self.fill_controls(2 * apply.contiguous())
        #gd = self.__manifold.gd.view(-1, 2)
        #self.__controls =torch.sum(self.__supportgen(gd)**2)
    
    def field_generator(self):
        gd = self.__manifold.gd.view(-1, 2)
        pts = self.__supportgen(gd)
        #manifold_Landmark = Landmarks(self.__manifold.dim, self.__manifold.dim + 1, gd=self.__supportgen(gd).view(-1))
        #Trans = Translations(manifold_Landmark, self.__sigma)
        #Trans.fill_controls(self.__controls * self.__vectorgen(gd))

        #return Trans.field_generator()
        #return StructuredField_0(self.__supportgen(gd),
        #                        self.__controls *self.__vectorgen(gd), self.__sigma)
        return StructuredField_0(pts,
                                 self.__controls * self.__vectorgen(gd), self.__sigma)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)
