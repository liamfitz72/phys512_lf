# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 17:38:18 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from tqdm import tqdm
plt.rcParams['figure.dpi'] = 200

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

if __name__=="__main__":
    pars=np.asarray( [69,0.022,0.12,0.06,2.1e-9,0.95])
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3]);
    model=get_spectrum(pars)
    model=model[:len(spec)]
    resid=spec-model
    chisq=np.sum( (resid/errs)**2)
    print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.")
    
    #read in a binned version of the Planck PS for plotting purposes
    planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
    errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3]);
    plt.scatter(ell,model,s=1)
    plt.errorbar(ell[:,0],spec[:,1],errs,fmt='.')
    plt.clf()
    
    