#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:14:19 2018

@author: barbaragris
"""

import numpy as np
from implicitmodules.src import kernels as ker


def my_VsToV(Par, z, j): # generic vector field (tested)
    """ This is the main function to compute the derivative of order 0 to 2 
    for vector fields generated by simple dirac of order 0 (landmarks) and 
    first order dirivatives of dirac: 'p' points corresponds to the symmetric
    of the derivatives and needs a couple (x,p) where x is (N,2) and p is (N,2,2) 
    and 'm' points correspont to skew symmetric part of the derivative and 
    are given by a pair (x,p) where x is (N,2) and p is (N,2,2). You can mixed 
    different kind of points ie '0', 'm' and 'p' points (no nneed to have all
    tyoes present) should be transmited in a dictionary as 
    Par={'0':[(x,p)], 'p':[(x,p]), 'm':[(x,p)], 'sig':sig} 
    z is the location where the values are computed
    
    The output is a list ordered according the the input '0', 'p' and 'm'
    """
    Nz = z.shape[0]
    sig = Par['sig']
    lsize = ((Nz,2),(Nz,2,2),(Nz,2,2,2))
    djv = np.zeros(lsize[j])
    
    if '0' in Par:
        for (x,p) in Par['0']:
            ker_vec = ker.my_vker(ker.my_xmy(z,x),j,sig)
            my_shape = (Nz, x.shape[0]) + tuple(ker_vec.shape[1:])
            ker_vec = ker_vec.reshape(my_shape)
            djv += np.tensordot(np.swapaxes(np.tensordot(np.eye(2),ker_vec, axes=0),0,2),
                p, axes = ([2,3],[1,0]))
        
    if 'p' in Par:
        for (x,P) in Par['p']:
            P = (P + np.swapaxes(P,1,2))/2
            ker_vec = -ker.my_vker(ker.my_xmy(z,x),j+1,sig)
            my_shape = (Nz, x.shape[0]) + tuple(ker_vec.shape[1:])
            ker_vec = ker_vec.reshape(my_shape)
            djv += np.tensordot(np.swapaxes(np.tensordot(np.eye(2),ker_vec, axes=0),0,2),
                P, axes = ([2,3,4],[1,0,2]))
    
    if 'm' in Par:
        for (x,P) in Par['m']:
            P = (P - np.swapaxes(P,1,2))/2
            ker_vec = -ker.my_vker(ker.my_xmy(z,x),j+1,sig)
            my_shape = (Nz, x.shape[0]) + tuple(ker_vec.shape[1:])
            ker_vec = ker_vec.reshape(my_shape)
            djv += np.tensordot(np.swapaxes(np.tensordot(np.eye(2),ker_vec, axes=0),0,2),
                P, axes = ([2,3,4],[1,0,2]))
    return djv


# Takes Cot (couples of GD=m and momenta=\eta) and returns
# an element of the dual of V_sig corresponding
    # to \xi^\ast_m (\eta)
def my_CotToVs(Cot, sig):
    Vs = {'0': [], 'p': [], 'm': []}
    [Vs['0'].append(s) for s in Cot['0']]
    
    if 'x,R' in Cot:
        [Vs['0'].append((s[0][0], s[1][0])) for s in Cot['x,R']]
        for ((x, R), (p, P)) in Cot['x,R']:
            Vs['m'].append((x, np.asarray([np.dot(P[i], R[i].transpose())
                                           for i in range(x.shape[0])])))
        # [Vs['m'].append((s[0][0],[np.dot(s[0][1].transpose(),s[1][1])))
    
    Vs['sig'] = sig
    return Vs
 