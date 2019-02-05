# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:33:00 2019

@author: gris
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:02:37 2019

@author: gris
"""


import numpy as np

import src.GeometricalDescriptors.Abstract as ab
from utilities import pairing_structures as npair
import src.StructuredFields.StructuredField_0 as stru_fie0
import src.StructuredFields.StructuredField_m as stru_fiem
import src.StructuredFields.Sum as stru_fie_sum


class GD_xR_C(ab.GeometricalDescriptors):
    def __init__(self, N_pts, dim, dimCont):  # 
        """
         This GD contains in addition of points and rotation matrices, the 
         matrices C. There infinitesimal action is null: they are not transported
         by the flow.
        """
        self.Cot = {'x,R': []}
        self.N_pts = N_pts
        self.dimCont = dimCont
        self.dim = dim
        self.GDshape = [self.N_pts, self.dim]
        self.Rshape = [self.N_pts, self.dim, self.dim]
        self.Cshape = [self.N_pts, self.dim, self.dimCont]
        self.GD = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape), np.zeros(self.Cshape))
        self.tan = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape), np.zeros(self.Cshape))
        self.cotan = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape), np.zeros(self.Cshape))
        
        self.dimGD = self.N_pts * self.dim + self.N_pts * self.dim * self.dim + self.N_pts * self.dim * self.dimCont
        self.dimMom = self.N_pts * self.dim + self.N_pts * self.dim * self.dim + self.N_pts * self.dim * self.dimCont
    
    def copy(self):  # 
        return GD_xR_C(self.N_pts, self.dim, self.dimCont)
    
    def copy_full(self):  # 
        GD = GD_xR_C(self.N_pts, self.dim, self.dimCont)
        x, R, C = self.GD
        dx, dR, dC = self.tan
        cotx, cotR, cotC = self.cotan
        GD.GD = (x.copy(), R.copy(), C.copy())
        GD.tan = (dx.copy(), dR.copy(), dC.copy())
        GD.cotan = (cotx.copy(), cotR.copy(), cotC.copy())
        return GD
    
    def fill_zero(self):
        self.GD = (np.zeros(self.GDshape), np.zeros(self.Rshape), np.zeros(self.Cshape))
        self.tan = (np.zeros(self.GDshape), np.zeros(self.Rshape), np.zeros(self.Cshape))
        self.cotan = (np.zeros(self.GDshape), np.zeros(self.Rshape), np.zeros(self.Cshape))

    def fill_zero_GD(self):        
        self.GD = (np.zeros(self.GDshape), np.zeros(self.Rshape), np.zeros(self.Cshape))

    def fill_zero_tan(self):        
        self.tan = (np.zeros(self.GDshape), np.zeros(self.Rshape), np.zeros(self.Cshape))

    def fill_zero_cotan(self):        
        self.cotan = (np.zeros(self.GDshape), np.zeros(self.Rshape), np.zeros(self.Cshape))
        

    def updatefromCot(self):
        pass
    
    def get_points(self):  # 
        return self.GD[0]
    
    def get_R(self):  # 
        return self.GD[1]
        
    def get_C(self):  # 
        return self.GD[2]
    
    def get_mom(self):  # 
        cotx, cotR, cotC = self.cotan
        return (cotx.copy(), cotR.copy(), cotC.copy())
    
    def fill_cot_from_param(self, param):  # 
        self.GD = (param[0][0].copy(), param[0][1].copy(), param[0][2].copy())
        self.cotan = (param[1][0].copy(), param[1][1].copy(), param[1][2].copy())
    
    
    def Cot_to_Vs(self, sig):  # 
        x = self.GD[0].copy()
        R = self.GD[1].copy() 
        px = self.cotan[0].copy()
        pR = self.cotan[1].copy()


        v0 = stru_fie0.StructuredField_0(sig, self.N_pts, self.dim)
        v0.fill_fieldparam((x, px))
        
        vm = stru_fiem.StructuredField_m(sig, self.N_pts, self.dim)
        P = np.asarray([np.dot(pR[i], R[i].transpose())
                                           for i in range(x.shape[0])])
        vm.fill_fieldparam((x, P))
        
        return stru_fie_sum.Summed_field([v0, vm])
    

    def Ximv(self, v):  #
        pts = self.get_points()
        R = self.get_R()
        vx = v.Apply(pts, 0)
        dvx = v.Apply(pts, 1)
        S = (dvx - np.swapaxes(dvx, 1, 2)) / 2
        vR = np.asarray([np.dot(S[i], R[i]) for i in range(pts.shape[0])])
        out = self.copy_full()
        out.fill_zero_cotan()
        out.tan = (vx.copy(), vR.copy(), np.zeros(self.Cshape))
        
        return out
    
    def dCotDotV(self, vs):  #  ,
        """
        Supposes that Cot has been filled
        """
        x = self.get_points()
        R = self.get_R()
        px, pR, pC = self.get_mom()
        
        dvx = vs.Apply(x, 1)
        ddvx = vs.Apply(x, 2)
        
        skew_dvx = (dvx - np.swapaxes(dvx, 1, 2)) / 2
        skew_ddvx = (ddvx - np.swapaxes(ddvx, 1, 2)) / 2
        
        dx = np.asarray([np.dot(px[i], dvx[i]) + np.tensordot(pR[i],
                                                              np.swapaxes(
                                                                  np.tensordot(R[i], skew_ddvx[i], axes=([0], [1])),
                                                                  0, 1))
                         for i in range(x.shape[0])])
        
        dR = np.asarray([np.dot(-skew_dvx[i], pR[i])
                         for i in range(x.shape[0])])
        
        GD = self.copy()
        GD.fill_zero_tan()
        GD.cotan = (dx.copy(), dR.copy(), np.zeros(self.Cshape))
        return GD
    
    def inner_prod_v(self, v):  # 
        vGD = self.Ximv(v)
        vx, vR, vC = vGD.tan
        px, pR, pC = self.get_mom()
        out = np.dot(px.flatten(), vx.flatten())
        out += np.sum([np.tensordot(pR[i], vR[i]) for i in range(vR.shape[0])])
        return out

    
    def add_GD(self, GDCot):
        x, R, C = self.GD
        xGD, RGD, CGD = GDCot.GD
        self.GD = (x + xGD, R + RGD, C+CGD)
        
            
    def add_tan(self, GDCot):
        dx, dR, dC = self.tan
        dxGD, dRGD, dCGD = GDCot.tan
        self.tan = (dx + dxGD, dR + dRGD, dC + dCGD)
            
    def add_cotan(self, GDCot):
        cotx, cotR, cotC = self.cotan
        cotxGD, cotRGD, cotCGD = GDCot.cotan
        self.cotan = (cotx + cotxGD, cotR + cotRGD, cotC + cotCGD)
        
    
    def mult_GD_scal(self, s):
        x, R, C = self.GD
        self.GD = (s * x, s * R, s * C)

    def mult_tan_scal(self, s):
        dx, dR, dC = self.tan
        self.tan = (s * dx, s * dR, s * dC)

    def mult_cotan_scal(self, s):
        cotx, cotR, cotC = self.cotan
        self.cotan = (s * cotx, s * cotR, s * cotC)

    def add_speedGD(self, GDCot):
        x, R, C = self.GD
        dx, dR, dC = GDCot.tan
        self.GD = (x + dx, R + dR, C + dC)
        
    def add_tantocotan(self, GDCot):
        dxGD, dRGD, dCGD = GDCot.tan
        cotx, cotR, cotC = self.cotan
        self.cotan = (cotx + dxGD, cotR + dRGD, cotC + dCGD)
        
    def add_cotantotan(self, GDCot):
        dx, dR, dC = self.tan
        cotxGD, cotRGD, cotCGD = GDCot.cotan
        self.tan = (dx + cotxGD, dR + cotRGD, dC + cotCGD)

    def add_cotantoGD(self, GDCot):
        x, R, C = self.GD
        cotxGD, cotRGD, cotCGD = GDCot.cotan
        self.GD = (x + cotxGD, R + cotRGD, C + cotCGD)

    def exchange_tan_cotan(self):
        (dx, dR, dC) = self.tan
        (cotx, cotR, cotC) = self.cotan
        self.tan = (cotx.copy(), cotR.copy(), cotC.copy())
        self.cotan = (dx.copy(), dR.copy(), dC.copy())

    def get_GDinVector(self):
        x, R, C = self.GD
        return np.concatenate([x.flatten(), R.flatten(), C.flatten()])

    def get_cotaninVector(self):
        cotx, cotR, cotC = self.cotan
        return np.concatenate([cotx.flatten(), cotR.flatten(), cotC.flatten()])

    def get_taninVector(self):
        tanx, tanR, tanC = self.tan
        return np.concatenate([tanx.flatten(), tanR.flatten(), tanC.flatten()])

    def fill_from_vec(self, PX, PMom):
        x = PX[:self.N_pts * self.dim]
        x = x.reshape([self.N_pts, self.dim])
        R = PX[self.N_pts * self.dim:self.N_pts * self.dim + self.N_pts * self.dim * self.dim]
        R = R.reshape([self.N_pts, self.dim, self.dim])
        C = PX[self.N_pts * self.dim + self.N_pts * self.dim * self.dim:]
        C = C.reshape([self.N_pts, self.dim, self.dimCont])
        
        cotx = PMom[:self.N_pts * self.dim]
        cotx = cotx.reshape([self.N_pts, self.dim])
        cotR = PMom[self.N_pts * self.dim:self.N_pts * self.dim + self.N_pts * self.dim * self.dim]
        cotR = cotR.reshape([self.N_pts, self.dim, self.dim])
        cotC = PMom[self.N_pts * self.dim + self.N_pts * self.dim * self.dim:]
        cotC = cotC.reshape([self.N_pts, self.dim, self.dimCont])
        
        param = ((x, R, C), (cotx, cotR, cotC))
        self.fill_cot_from_param(param)