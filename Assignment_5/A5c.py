# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:17:41 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from tqdm import tqdm
from A5a import get_spectrum
from A5b import CMBmodel_param_grad


def get_step_scale(m0,hess_inv,y_data):
    ndata=len(y_data)
    grad=CMBmodel_param_grad(m0)[:ndata]
    model=get_spectrum(m0)[:ndata]
    resid=y_data-model
    return -hess_inv@grad.T@resid

def model_chisq(m,errs,y_data):
    model=get_spectrum(m)
    resid=y_data-model[:len(y_data)]
    return np.sum((resid/errs)**2),resid

def MCMC_chain(y_data,errs,m0,hess_inv,nstep=10000,T=1):
    nparam=len(m0)
    chain=np.zeros([nstep,nparam])
    chisq=np.zeros(nstep)
    chain[0,:]=m0
    cur_chisq,resid=model_chisq(m0,errs,y_data)
    chisq[0]=cur_chisq
    step_scale=get_step_scale(m0,hess_inv,y_data)
    m=m0
    for i in tqdm(range(1,nstep)):
        dm=np.multiply(np.random.randn(nparam),step_scale)
        trial_m=np.matrix(m+dm)
        new_chisq,resid=model_chisq(trial_m.T,errs,y_data)
        accept_prob=np.exp(-0.5*(new_chisq-cur_chisq)/T)
        if np.random.rand(1)<accept_prob:
            m=trial_m
            cur_chisq,resid=(new_chisq,resid)
        chain[i,:]=np.ravel(m)
        chisq[i]=cur_chisq
        print(cur_chisq)
    return chain,chisq

def chain_eval(chain,chisq,T=1):
    dchi=chisq-np.min(chisq)
    wt=np.exp(-0.5*dchi*(1-1/T))
    npar=chain.shape[1]
    tot=np.zeros(npar)
    totsqr=np.zeros(npar)
    for i in range(npar):
        tot[i]=np.sum(wt*chain[:,i])
        totsqr[i]=np.sum(wt*chain[:,i]**2)
    mean=tot/np.sum(wt)
    meansqr=totsqr/np.sum(wt)
    var=meansqr-mean**2
    return mean,np.sqrt(var),wt

if __name__=="__main__":
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3])
    # planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
    # errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])
    hess_inv=np.loadtxt('inverse_hessian.txt',delimiter=',')
    
    m0=[69,0.022,0.12,0.06,2.1e-9,0.95]
    y_data=spec
    chain,chisq=MCMC_chain(y_data,errs,m0,hess_inv)
    chain_data=np.hstack(chisq,chain)
    np.savetxt('planck_chain.txt',chain_data,delimiter=',')
    
