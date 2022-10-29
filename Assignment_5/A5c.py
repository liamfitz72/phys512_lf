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

def get_step(trial_step):
    if len(trial_step.shape)==1:
        return np.random.randn(len(trial_step))*trial_step
    else:
        L=np.linalg.cholesky(trial_step)
        return L@np.random.randn(trial_step.shape[0])

def model_chisq(m,errs,y_data):
    model=get_spectrum(m)
    resid=y_data-model[:len(y_data)]
    return np.sum((resid/errs)**2),resid

def MCMC_chain(y_data,errs,m0,trial_step,nstep=10000,T=1):
    nparam=len(m0)
    chain=np.zeros([nstep,nparam])
    chisq=np.zeros(nstep)
    chain[0,:]=m0
    cur_chisq,resid=model_chisq(m0,errs,y_data)
    chisq[0]=cur_chisq
    m=m0
    for i in tqdm(range(1,nstep)):
        dm=get_step(trial_step)
        trial_m=np.matrix(m+dm)
        new_chisq,resid=model_chisq(trial_m.T,errs,y_data)
        accept_prob=np.exp(-0.5*(new_chisq-cur_chisq)/T)
        if np.random.rand(1)<accept_prob:
            m=trial_m
            cur_chisq,resid=(new_chisq,resid)
        chain[i,:]=np.ravel(m)
        chisq[i]=cur_chisq
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
    cov=np.loadtxt('cov_matrix.txt',delimiter=',')
      
    # m0=np.loadtxt('planck_fit_params.txt',delimiter=',')[:,0]
    m0=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
    y_data=spec
    nstep=20000
    chain,chisq=MCMC_chain(y_data,errs,m0,cov,nstep=nstep)
    chain_data=np.empty([len(nstep),len(m0)])
    chain_data[:,0]=chisq
    chain_data[:,1:]=chain
    np.savetxt('planck_chain.txt',chain_data,delimiter=',')
