import numpy as np
from scipy.linalg import solve

import src.DeformationModules.Abstract as ab
import src.GeometricalDescriptors.Landmark as GeoDescr

import src.StructuredFields.StructuredFields as stru_fie
from src.Kernels import ScalarGaussian as ker

class ElasticOrderO(ab.DeformationModule):
    """
     Elastic module of order 0
    """
    
    
    def __init__(self, sigma, N_pts, dim, coeff, nu):
        """
        sigma is the scale of the rkhs of generated vector fields
        N_pts is the number of landmarks
        dim is the dimension of the ambient space
        """
        self.sig = sigma
        self.N_pts = N_pts
        self.dim =dim
        self.coeff = coeff
        self.nu = nu
        self.GD = GeoDescr.GD_landmark(N_pts, dim)
        self.SKS = np.zeros([self.N_pts*self.dim,self.N_pts*self.dim])
        self.Mom = np.zeros([self.N_pts, self.dim])
        self.Cont = np.zeros([self.N_pts, self.dim])
        self.cost = 0.

    def copy(self):
        return ElasticOrderO(self.sig, self.N_pts, self.dim, self.coeff, self.nu)
    
    def copy_full(self):
        Mod = ElasticOrderO(self.sig, self.N_pts, self.dim, self.coeff, self.nu)
        Mod.GD = self.GD.copy_full()
        Mod.SKS = self.SKS.copy()
        Mod.Mom = self.Mom.copy()
        Mod.Cont = self.Cont.copy()
        Mod.cost = self.cost
        return Mod

    
    
    def fill_GD(self, GD):
        self.GD = GD.copy_full()
        self.SKS = np.zeros([self.N_pts*self.dim,self.N_pts*self.dim])


    def Compute_SKS_curr(self):
        """
        Supposes that values of GD have been filled
        """
        try:
            x = self.GD.get_points()
            self.SKS = ker.my_K(x, x, self.sig, 0)
            self.SKS += self.nu * np.eye(self.N_pts * self.dim)
        except:
            raise NameError('Need to fill landmark points before computing SKS')
    
    #def Compute_SKS(self, x):
    #    return ker.my_K(x, x, self.sig, 0) + self.nu * np.eye(self.N_pts * self.dim)


    def update(self):
        """
        Computes SKS so that it is done only once.
        Supposes that values of GD have been filled
        """
        
        self.Compute_SKS_curr()


    def GeodesicControls_curr(self, GDCot):
        """
        Supposes that SKS has been computed and values of GD filled
        Supposes that GDCot has Cot filled
        """
        vs = GDCot.Cot_to_Vs(self.sig)
        vm = vs.Apply(self.GD.get_points(), 0)
        #print(self.sig)
        self.Cont = solve(self.coeff * self.SKS,
                     vm.flatten(),sym_pos = True).reshape(self.N_pts, self.dim)
        self.Mom = self.Cont.copy()


    def field_generator_curr(self):
        return self.field_generator(self.GD, self.Cont)
    
    
    def field_generator(self, GD, Cont):
        param = [GD.get_points(), Cont]
        v = stru_fie.StructuredField_0(self.sig, self.dim)
        v.fill_fieldparam(param)
        return v
    
    def Cost_curr(self):
        SKS = self.SKS
        p = self.Cont.flatten()
        self.cost = self.coeff * np.dot(p, np.dot(SKS, p))/2
        
    def Cost(self, GD, Cont):
        x = GD.get_points()
        SKS = self.Compute_SKS(x)
        p = Cont.flatten()
        return self.coeff * np.dot(p, np.dot(SKS, p))/2
       
    def DerCost_curr(self):
        vs  = self.field_generator_curr()
        out =  self.p_Ximv_curr(vs, 1)
        
        return out
    
    
      
    """
    def DerCost(self, GD, Mom):#tested
    vs  = self.field_generator(GD, Mom)
    der = vs.p_Ximv(vs, 1)
    out = self.GD.copy()
    out.Cot['0'] = [( self.coeff * der['0'][0][1], np.zeros([self.N_pts, self.dim]) )]
    return out
    """

    
    def cot_to_innerprod_curr(self, GDCot, j):#
        """
         Transforms the GD (with Cot filled) GDCot into vsr and computes
         the inner product (or derivative wrt self.GD) with the
         field generated by self.
         The derivative is a returned as a GD with cotan filled (tan=0)
        """
        
        vsr = GDCot.Cot_to_Vs(self.sig)
        out = self.p_Ximv_curr(vsr, j)
        
        """
        if j==0:
            out = innerprod
        if j==1:
            out = self.GD.copy()
            out.Cot['0'] = [ (innerprod['0'][0][1], np.zeros([self.N_pts,self.dim]) )]
        """   
        return out
 

    def p_Ximv_curr(self, vs, j):
        """
        Put in Module because it uses the link between GD and support 
        of vector fields      
        """
        GD_cont = self.GD.copy_full()
        GD_cont.cotan = self.Cont.copy()
        
        if j==0:
            out = 0.
            x = GD_cont.GD.copy()
            p = GD_cont.cotan.copy()
            vx = vs.Apply(x, j)
            out += np.sum(np.asarray([np.dot(p[i],vx[i]) 
                        for i in range(x.shape[0])]))

        elif j==1:
            out = self.GD.copy_full()
            out.fill_zero_tan()
            x = GD_cont.GD.copy()
            p = GD_cont.cotan.copy()
            vx = vs.Apply(x, j)
            der = np.asarray([np.dot(p[i],vx[i]) for i in range(x.shape[0])])
            out.cotan = der.copy()
            
        return out
            
      
        


          
            




































            