import sys
sys.path.append('/home/trouve/Implicit/implicitmodules')

import pickle

import src.ExpIpam.utils as ipam


    
################################################################################
#########



# %%  Basi experiments
flag = 'basi'
(source, target) = ('leafbasi.pkl', 'leafbasit.pkl')
(height_source, height_target, Dx, Dy) = (38., 100., 0., 0.)
dir_res = '/home/trouve/Dropbox/Talks/Pics/ResImp/ResIpam/'
dir_Mysh = '/home/trouve/Dropbox/Talks/2019_04_IPAM/tex/Mysh/'

#%%
# lddmm
name_exp = 'leaf_ld2_p'
coeffs = [0.01, 0.01, 100]
nu = 0.001
(sig00, sig0, sig1) = (1000., 15., 30.)
(lam_var, sig_var) = (10., 30.)
attach_var = (lam_var, sig_var)
maxiter = 30
max_fun = 100
N = 10
Np = 15

#%%
# parametric
name_exp = 'leaf_ba2_p'
coeffs = [0.01, 100, 0.001]
nu = 0.001
(sig00, sig0, sig1) = (1000., 15., 30.)
(lam_var, sig_var) = (10., 30.)
attach_var = (lam_var, sig_var)
maxiter = 30
max_fun = 100
N = 10
Np = 15

#%%  Computation

(P0, outvar) = ipam.exp_ipam_init(flag = flag, 
           source = source, target = target, name_exp = name_exp, 
           dir_res = dir_res,  
           Dx = Dx, Dy = Dy, height_source = height_source,
           height_target = height_target, sig00 = sig00, sig0 = sig0, sig1 = sig1, 
           coeffs = coeffs, nu = nu, lam_var = lam_var, sig_var = sig_var)
          # outvar = (Module, xs, xst, opti.fun, opti.jac)
          
P1 = ipam.exp_ipam_optim((P0, outvar), 
                         attach_var, maxiter = maxiter, maxfun = 200, N=N)

# Save the result in case
filepkl = dir_res + name_exp + ".pkl"
with open(filepkl, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([P0, P1, outvar, attach_var, maxiter, max_fun, N], f)

# plottings and savings

invar = (Np, flag, name_exp, height_source, height_target, Dx, Dy)
(Module, xs, xst, fun, jac) = outvar
outvarplot = (Module, P1, xs, xst, dir_res, dir_Mysh)
ipam.exp_ipam_plot(invar, outvarplot)

#%%

from_past = True
if (from_past):
    with open(filepkl, 'rb') as f:  # Python 3: open(..., 'rb')
     P0, P1, outvar, attach_var, maxiter, max_fun, N = pickle.load(f)
else:
    P0 = P1.copy()
         
P1 = ipam.exp_ipam_optim((P0, outvar), attach_var, maxiter = 1, maxfun = 200, N=10)

# Save  again the result in case
filepkl = dir_res + name_exp + ".pkl"
with open(filepkl, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([P0, P1, outvar, attach_var, maxiter, max_fun, N], f)
    
# plottings and savings
N = 15
invar = (N, flag, name_exp, height_source, height_target, Dx, Dy)
(Module, xs, xst, fun, jac) = outvar
outvarplot = (Module, P1, xs, xst, dir_res, dir_Mysh)
ipam.exp_ipam_plot(invar, outvarplot)
 