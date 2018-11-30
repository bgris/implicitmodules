 
##  Exp basipetal  (Sandbox to test backward dynamics)
##
##
import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import scipy.optimize
import os
import numpy as np
import matplotlib.pyplot as plt

from implicitmodules.src.visualisation import my_close
import implicitmodules.src.rotation as rot
import implicitmodules.src.useful_functions as utils
import implicitmodules.src.shooting as shoot
import pickle


plt.show()
plt.ion()

def my_plot3(Mod0, Mod1, Cot, xst, fig, nx, ny, name, i):
    x0 = Mod0['0']
    plt.axis('equal')
    x0 = my_close(x0)
    
    (x1,R) = Mod1['x,R']
    plt.plot(x1[:,0],x1[:,1],'b.')
    plt.plot(x0[:,0],x0[:,1],'r.')
    
    xs = Cot['0'][1][0] #[nx*ny:,:]
    xs = my_close(xs)
    plt.plot(xs[:,0], xs[:,1],'g')
    
    xst = my_close(xst)
    plt.plot(xst[:,0], xst[:,1],'m')
    
    plt.ylim(-40,80)
    plt.show()  
    plt.savefig(name + '{:02d}'.format(i) +'.png')
    return

    
def my_plot4(Mod0, Mod1, Cot, xst, fig, nx, ny, name, i):            
    x0 = Mod0['0']
    plt.axis('equal')
    x0 = my_close(x0)
    
    
    xs = Cot['0'][1][0][0:nx*ny,:]
    xsx = xs[:,0].reshape((nx,ny))
    xsy = xs[:,1].reshape((nx,ny))
    plt.plot(xsx, xsy,color = 'lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(),color = 'lightblue')
    (x1,R) = Mod1['x,R']
    plt.plot(x1[:,0],x1[:,1],'b.')
    plt.plot(x0[:,0],x0[:,1],'r.')
    
    xs = Cot['0'][1][0][nx*ny:,:]
    xs = my_close(xs)
    plt.plot(xs[:,0], xs[:,1],'g')
    
    xst = my_close(xst)
    plt.plot(xst[:,0], xst[:,1],'m')
    
    
    plt.ylim(-10,120)
    plt.show()  
    plt.savefig(name + '{:02d}'.format(i) +'.png')
    return

def my_exp(*args):
    (model, target, Dx, Dy, height_model, height_target) = args[0]
    (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = args[1]

    height = height_model
    heightt = height_target
    # Load model and proper scaling and shift
    with open('../data/' + model, 'rb') as f:
        img, lx = pickle.load(f)
    
    nlx = np.asarray(lx).astype(np.float64)
    nlx[:,1] = - nlx[:,1]
    (lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))
    
    scale = height/(lmax-lmin)
    nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0])) + Dx
    nlx[:,1]  = scale*(nlx[:,1]-lmin) + Dy
    
    
    nx, ny = (5,11) # create a grid for visualisation purpose
    u = height/38.
    (a,b,c,d) = (-10.*u + Dx, 10.*u + Dx, -3.*u + Dy, 40.*u + Dy)
    [xx, xy] = np.meshgrid(np.linspace(a,b,nx), np.linspace(c,d,ny))
    (nx,ny) = xx.shape
    nxs = np.asarray([xx.flatten(), xy.flatten()]).transpose()
    
    
    # Load target and proper scaling
    with open('../data/' + target, 'rb') as f:
        img, lxt = pickle.load(f)
    
    nlxt = np.asarray(lxt).astype(np.float64)
    nlxt[:,1] = -nlxt[:,1]
    (lmint, lmaxt) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
    scale = heightt/(lmaxt-lmint)
    
    
    nlxt[:,1]  = scale*(nlxt[:,1] - lmint)
    nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))
    
    x0 = nlx[nlx[:,2]==0,0:2]
    x1 = nlx[nlx[:,2]==1,0:2]
    xs = nlx[nlx[:,2]==2,0:2]
    xst = nlxt[nlxt[:,2]==2,0:2]
    
    th = 0*np.pi
    p0 = np.zeros(x0.shape)
    Mod0 ={'0': x0, 'sig':sig0, 'coeff':coeffs[0]}
    
    th = th*np.ones(x1.shape[0])
    R = np.asarray([rot.my_R(cth) for cth in th])
    
    for  i in range(x1.shape[0]):
        R[i] = rot.my_R(th[i])
        
    Mod1 = {'x,R':(x1,R), 'sig':sig1, 'coeff' :coeffs[1], 'nu' : nu}
    
    ps = np.zeros(xs.shape)
    
    C = np.zeros((x1.shape[0],2,1))
    K = 10
    
    L = height
    a, b = -2/L**3, 3/L**2
    C[:,1,0] = K*(a*(L-x1[:,1]+Dy)**3 + b*(L-x1[:,1]+Dy)**2)
    C[:,0,0] = 1.*C[:,1,0]
    
    Mod1['C'] = C
    
    (p1,pR) = (np.zeros((x1.shape)), np.zeros((x1.shape[0],2,2)))
    
    Cot ={ '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,pR))]}
    
    N = 7
    X = utils.my_X(x0, xs, x1, R)
    nX = (x0.shape[0], xs.shape[0], x1.shape[0])
    args = (X, nX, sig0, sig1, coeffs[0], coeffs[1], C, nu, xst, lam_var, sig_var, N)
    P0 = utils.my_P(p0, ps, p1, pR)
    
    res= scipy.optimize.minimize(shoot.my_fun,
            P0,
            args = args,
            method='L-BFGS-B',
            jac=shoot.my_jac,
            bounds=None,
            tol=None,
            callback=None,
            options={ 'disp': True, 'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03, 'eps': 1e-08, 'maxfun': 100, 'maxiter': 5, 'iprint': -1, 'maxls': 20}
            )
    
    args, nxs , name
    
    fig = plt.figure(1)
    plt.clf()
    P0 = res['x']
    tp0, tps, (tp1, tpR) = utils.my_splitP(P0,nX)
    axs = np.concatenate((nxs,xs), axis = 0)
    nps = np.zeros(nxs.shape)
    aps = np.concatenate((nps,tps), axis = 0)
    aX = utils.my_X(x0, axs, x1, R)
    anX = (x0.shape[0], axs.shape[0], x1.shape[0])
    aargs = (aX, anX, sig0, sig1, coeffs[0], coeffs[1], C, nu, xst, lam_var, sig_var, N)
    
    aP0 = utils.my_P(tp0, aps, tp1, tpR)
    Traj = shoot.my_fun_Traj(aP0,*aargs)
    
    for i in range(N+1):
        plt.clf()
        (tMod0, tMod1, tCot) = Traj[2*i]
        my_plot4(tMod0,tMod1,tCot,xst, fig,nx,ny, name, i)
    
    filepng = name + "*.png"
    os.system("convert " + filepng + " -set delay 0 -reverse " + filepng + " -loop 0 " + name + ".gif")
    return

##
# Exemple with a shift 
# (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expf_', [5., 0.05], 0.001, 10., 30., 10., 10.)
# (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expa_', [1., 10.], 0.001, 10., 30. , 10., 10.)
# (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expb_', [1., 10.], 0.001,
# 10., 30.,  1., 10.)
# (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expe_', [5., 0.05], 0.001, 10., 30., 1., 10.)
# (name, coeffs, nu, sin0, sig1, lam_var, sig_var) = ('basi_expd_', [5., 0.05],0.001, 30., 30., 1., 10.)
# (name, coeffs, nu, sin0, sig1, lam_var, sig_var) = ('basi_expc_', [1., 0.05],0.001, 300., 30., 10., 30.)
# Small to Big centrer

#  Big to small (un centred)
(model, target, Dx, Dy, height_model, height_target) =('basi1b.pkl','basi1t.pkl', 0., 0., 100., 38.)
(name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('../results/basi_exp_XLtoS_uncentred_', [1., 0.05], 0.001, 200., 30., 10., 30.)

args = ((model, target, Dx, Dy, height_model, height_target),
        (name, coeffs, nu, sig0, sig1, lam_var, sig_var))
my_exp(*args)

#  Big to small (un centred, right shift)
(model, target, Dx, Dy, height_model, height_target) =('basi1b.pkl', 'basi1t.pkl', 10., 0., 100., 38.)
(name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('../results/basi_exp_XLtoS_uncentredb_', [1., 0.05], 0.001, 200., 30., 10., 30.)

args = ((model, target, Dx, Dy, height_model, height_target),
        (name, coeffs, nu, sig0, sig1, lam_var, sig_var))
my_exp(*args)

# small to big (centred)
(model, target, Dx, Dy, height_model, height_target) = ('basi1b.pkl','basi1t.pkl', 0., 30., 38., 100.)
(name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('../results/basi_exp_XLtoS_centred_', [1., 0.05], 0.001, 200., 30., 10., 30.)

args = ((model, target, Dx, Dy, height_model, height_target),
        (name, coeffs, nu, sig0, sig1, lam_var, sig_var))
my_exp(*args)

#  small to big (un centred)
(model, target, Dx, Dy, height_model, height_target) = ('basi1b.pkl','basi1t.pkl', 0., 0., 38., 100.)
(name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('../results/basi_exp_uncentred_', [1., 0.05], 0.001, 200., 30., 10., 30.)

args = ((model, target, Dx, Dy, height_model, height_target),
        (name, coeffs, nu, sig0, sig1, lam_var, sig_var))
my_exp(*args)

#  small to big (un centred, right shift)
(model,target, Dx, Dy, height_model, height_target) = ('basi1b.pkl','basi1t.pkl', 20., 0., 38., 100.)
(name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('../results/basi_exp_uncentredb_', [1., 0.05], 0.001, 200., 30., 10., 30.)

args = ((model, target, Dx, Dy, height_model, height_target),
        (name, coeffs, nu, sig0, sig1, lam_var, sig_var))
my_exp(*args)
