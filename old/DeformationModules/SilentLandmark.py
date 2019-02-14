import numpy as np

from old import GeometricalDescriptors
import old.StructuredFields.ZeroFields
from old.DeformationModules.DeformationModules import DeformationModule


class SilentLandmark(DeformationModule):#tested
    """
    Silent deformation module with GD that are points
    """

    
    def __init__(self, N_pts, dim):
       self.N_pts = N_pts
       self.dim = dim
       self.GD = old.GeometricalDescriptors.GD_landmark.GD_landmark(N_pts, dim)
       #self.Mom = np.empty([self.N_pts, self.dim])
       self.cost = 0.
       self.Cont = np.empty([0])
   
    def copy(self):
        return SilentLandmark(self.N_pts, self.dim)
    
    
    def add_cot(self, GD):
        self.GD.add_cot(GD.Cot)
        
    def copy_full(self):
        Mod = SilentLandmark(self.N_pts, self.dim)
        Mod.GD = self.GD.copy_full()
        #Mod.Mom = self.Mom.copy()
        return Mod
    
    def update(self):
        pass
    
    def fill_GD(self, GD):
        self.GD = GD.copy_full()
        
    def GeodesicControls_curr(self, GDCot):
        pass
    
    def GeodesicControls(self, GD, GDCot):
        pass
    
    def field_generator_curr(self):
        return old.StructuredFields.ZeroFields.ZeroField(self.dim)

    def field_generator(self, GD, Cont):
        return old.StructuredFields.ZeroFields.ZeroField(self.dim)
    
    
    def Cost_curr(self):
        pass
        
    def Cost(self, GD, Cont):
        return 0.
       

    def DerCost_curr(self):#tested
        out = self.GD.copy()
        out.Cot['0'] = [(np.zeros([self.N_pts, self.dim]), np.zeros([self.N_pts, self.dim]) )]
        return out

      
    def DerCost(self, GD, Mom):#tested
        out = GD.copy()
        out.Cot['0'] = [(np.zeros([self.N_pts, self.dim]), np.zeros([self.N_pts, self.dim]) )]
        return out


    
    def cot_to_innerprod_curr(self, GDCot, j):#tested
        """
         Transforms the GD (with Cot filled) GDCot into vsr and computes
         the inner product (or derivative wrt self.GD) with the
         field generated by self.
         The derivative is a returned as a GD with Cot filled (mom=0)
        """
        
        
        if j==0:
            out = 0.
        if j==1:
            out = self.GD.copy()
            out.Cot['0'] = [ (np.zeros([self.N_pts,self.dim]), np.zeros([self.N_pts,self.dim]) )]
            
        return out