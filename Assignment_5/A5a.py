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

def get_hessian(m,n_data):
    l=len(m)
    hess=np.empty([n_data,l,l])
    m_c=m.copy()
    for i in tqdm(range(l)):
        for j in range(l):
            dm_i=m[i]/100
            dm_j=m[j]/100
            if i==j:
                y0=get_spectrum(m_c)
                m_c[i]+=dm_i
                y1=get_spectrum(m_c)
                m_c[i]-=2*dm_i
                y_1=get_spectrum(m_c)
                hess[:,i,i]=(y1-2*y0+y_1)[:n_data]/dm_i**2
            else:
                m_c[i]+=dm_i
                m_c[j]+=dm_j
                y11=get_spectrum(m_c)
                m_c[j]-=2*dm_j
                y1_1=get_spectrum(m_c)
                m_c[i]-=2*dm_i
                m_c[j]+=2*dm_j
                y_11=get_spectrum(m_c)
                m_c[j]-=2*dm_j
                y_1_1=get_spectrum(m_c)
                hess[:,i,j]=(y11-y1_1-y_11+y_1_1)[:n_data]/(4*dm_i*dm_j)
    return hess

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
    plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
    plt.clf()
    
    #plot residuals to find interval without structure to calculate noise estimation
    plt.scatter(ell,resid,s=1)
    plt.legend(['Residuals'])
    plt.savefig('A5_plot1.png',bbox_inches='tight')
    plt.clf()
    plt.scatter(ell[1400:1900],resid[1400:1900],s=1)
    noise_est=np.std(resid[1400:1900])
    plt.legend(['Residuals from 1400-1900'])
    plt.savefig('A5_plot2.png',bbox_inches='tight')
    print('noise estimate is:',noise_est)
    print(get_hessian(pars,len(spec)))
    
    