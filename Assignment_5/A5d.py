# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 23:59:39 2022

@author: liamf
"""

import numpy as np
from A5a import get_spectrum
from A5b import CMBmodel_param_grad
from A5c import MCMC_chain
import tqdm

def gaussian(x,mean,sig):
    return np.exp(-0.5*(x-mean)**2/sig**2)

def importance_sample(chain,wt):
    npar=len(chain[0,:])
    tot=np.zeros(npar)
    totsqr=np.zeros(npar)
    for i in range(npar):
        tot[i]=np.sum(wt*chain[:,i])
        totsqr[i]=np.sum(wt*chain[:,i]**2)
    mean=tot/np.sum(wt)
    meansqr=totsqr/np.sum(wt)
    var=meansqr-mean**2
    return mean,np.sqrt(var)

if __name__=="__main__":
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3])
    
    planck_chain=np.loadtxt('planck_chain.txt',delimiter=',')
    chisq,chain=(planck_chain[:,0],planck_chain[:,1:])
    
    tau=0.0540
    tau_err=0.0074
    wt_fun=gaussian(chain[:,3],tau,tau_err)
    m,m_err=importance_sample(chain,wt_fun)
    impsample=np.empty([len(m),2])
    impsample[:,0]=m
    impsample[:,1]=m_err
    resid=spec-get_spectrum(m)[:len(spec)]
    chi_sq=np.sum((resid/errs)**2)
    print('Importance sampled chisq:',chi_sq)
    np.savetxt('planck_params_impsample.txt',impsample,delimiter=',')
    
    A=CMBmodel_param_grad(m,spec)
    N_inv=np.linalg.inv(np.diag(errs**2))
    cov_new=np.linalg.inv(A.T@N_inv@A)
    
    m0=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
    m0=np.loadtxt('planck_fit_params.txt',delimiter=',')[:,0]
    y_data=spec
    nstep=20000
    print(m)
    chain,chisq=MCMC_chain(y_data,errs,m0,cov_new,nstep=nstep,scale=0.5)
    chain_data=np.empty([len(nstep),len(m0)])
    chain_data[:,0]=chisq
    chain_data[:,1:]=chain
    np.savetxt('planck_chain_tauprior.txt',chain_data,delimiter=',')
    

    
# [6.76129133e+01 2.22717993e-02 1.19099062e-01 5.58662850e-02
#  2.09997597e-09 9.69836774e-01]
    
    