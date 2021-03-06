import numpy as np

from implicitmodules.numpy.Manifolds.Abstract import Manifold
from implicitmodules.numpy.StructuredFields import Sum


class CompoundManifold(Manifold):
    def __init__(self, GD_list):  # tested 0
        self.GD_list = GD_list
        self.N_GDs = len(self.GD_list)
        self.dim = GD_list[0].dim
    
    def copy(self):  # tested
        GD_list = []
        for i in range(self.N_GDs):
            GD_list.append(self.GD_list[i].copy())
        GD = CompoundManifold(GD_list)
        # GD.indi_0 = self.indi_0.copy()
        # GD.indi_xR = self.indi_xR.copy()
        return GD
    
    def copy_full(self):  # tested
        GD_list = []
        for i in range(self.N_GDs):
            GD_list.append(self.GD_list[i].copy_full())
        GD_comb = CompoundManifold(GD_list)
        # GD_comb.indi_0 = self.indi_0.copy()
        # GD_comb.indi_xR = self.indi_xR.copy()
        # GD_comb.fill_cot_from_GD()
        return GD_comb
    
    def fill_zero(self):
        for i in range(self.N_GDs):
            self.GD_list[i].fill_zero()
        # self.fill_cot_from_GD()
    
    def fill_zero_GD(self):
        for i in range(self.N_GDs):
            self.GD_list[i].fill_zero_GD()
    
    def fill_zero_tan(self):
        for i in range(self.N_GDs):
            self.GD_list[i].fill_zero_tan()
    
    def fill_zero_cotan(self):
        for i in range(self.N_GDs):
            self.GD_list[i].fill_zero_cotan()
    
    def fill_cot_from_param(self, param):
        for i in range(self.N_GDs):
            self.GD_list[i].fill_cot_from_param(param[i])
    
    def Cot_to_Vs(self, sig):  # tested
        """
        Supposes that Cot has been filled
        """
        v_list = [self.GD_list[i].Cot_to_Vs(sig) for i in range(self.N_GDs)]
        return Sum.sum_structured_fields(v_list)

    def infinitesimal_action(self, v):
        """
        xi_m ()
        
        """
        return CompoundManifold([GDi.infinitesimal_action(v) for GDi in self.GD_list])
    
    def dCotDotV(self, vs):  #
        """
        Supposes that Cot has been filled
        (p infinitesimal_action(m,v)) wrt m
        """
        return CompoundManifold([GDi.dCotDotV(vs) for GDi in self.GD_list])
    
    def inner_prod_v(self, v):  # tested0
        return sum([GD_i.inner_prod_v(v) for GD_i in self.GD_list])
    
    def add_GD(self, GDCot):
        for i in range(self.N_GDs):
            self.GD_list[i].add_GD(GDCot.GD_list[i])
    
    def add_tan(self, GDCot):
        for i in range(self.N_GDs):
            self.GD_list[i].add_tan(GDCot.GD_list[i])
    
    def add_cotan(self, GDCot):
        for i in range(self.N_GDs):
            self.GD_list[i].add_cotan(GDCot.GD_list[i])
    
    def mult_GD_scal(self, s):
        for i in range(self.N_GDs):
            self.GD_list[i].mult_GD_scal(s)
    
    def mult_tan_scal(self, s):
        for i in range(self.N_GDs):
            self.GD_list[i].mult_tan_scal(s)
    
    def mult_cotan_scal(self, s):
        for i in range(self.N_GDs):
            self.GD_list[i].mult_cotan_scal(s)
    
    def add_speedGD(self, GDCot):
        for i in range(self.N_GDs):
            self.GD_list[i].add_speedGD(GDCot.GD_list[i])
    
    def add_tantocotan(self, GDCot):
        for i in range(self.N_GDs):
            self.GD_list[i].add_tantocotan(GDCot.GD_list[i])
    
    def add_cotantotan(self, GDCot):
        for i in range(self.N_GDs):
            self.GD_list[i].add_cotantotan(GDCot.GD_list[i])
    
    def add_cotantoGD(self, GDCot):
        for i in range(self.N_GDs):
            self.GD_list[i].add_cotantoGD(GDCot.GD_list[i])
    
    def exchange_tan_cotan(self):
        for i in range(self.N_GDs):
            self.GD_list[i].exchange_tan_cotan()
    
    def get_GDinVector(self):
        vec_list = [self.GD_list[i].get_GDinVector() for i in range(self.N_GDs)]
        return np.concatenate(vec_list)
    
    def get_cotaninVector(self):
        vec_list = [self.GD_list[i].get_cotaninVector() for i in range(self.N_GDs)]
        return np.concatenate(vec_list)
    
    def get_taninVector(self):
        vec_list = [self.GD_list[i].get_taninVector() for i in range(self.N_GDs)]
        return np.concatenate(vec_list)
    
    def fill_from_vec(self, PX, PMom):
        countX = 0
        countMom = 0
        for i in range(self.N_GDs):
            dimX = self.GD_list[i].dimGD
            dimMom = self.GD_list[i].dimMom
            self.GD_list[i].fill_from_vec(PX[countX: countX + dimX], PMom[countMom: countMom + dimMom])
            countX += dimX
            countMom += dimMom
