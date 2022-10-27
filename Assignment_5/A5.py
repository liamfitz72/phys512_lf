# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 17:38:18 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from tqdm import tqdm

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


pars=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
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
# planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
# errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3]);
# plt.plot(ell,model)
# plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')


def CMBmodel_param_grad(m):  # Get parameter gradient of model numerically
    dx=10^(-5)
    grad=[]
    for i in range(len(m)):
        m_c=m.copy()
        m_c[i]+=dx    # m_i -> m_i + dm_i
        y2=get_spectrum(m)
        m_c[i]-=2*dx  # m_i -> m_i - dm_i
        y1=get_spectrum(m)
        ddm_i=(y2-y1)/(2*dx)  # Central difference
        grad.append(ddm_i)
    return np.matrix(grad).T  # Transpose fits with Newton method function

def CMBmodel_newton_min(m0,y_data,n):
    m=m0.copy()
    for i in tqdm(range(n)):
        pred=get_spectrum(m)[:len(y_data)]
        grad=CMBmodel_param_grad(m)[:len(y_data)]
        resid=y_data-pred
        resid=np.matrix(resid).T
        grad=np.matrix(grad)
        lhs=grad.T@grad
        rhs=grad.T@resid
        dm=np.linalg.pinv(lhs)*rhs
        for j in range(len(m)):
            m[j]=m[j]+float(dm[j])  # Keep m as list type to pass through calc_lorentz
    return m

m0=[69,0.022,0.12,0.06,2.1e-9,0.95]
y_data=spec
n=5
m=CMBmodel_newton_min(m0,y_data,n)
model_newton=get_spectrum(m)
model_newton=model_newton[:len(spec)]
resid=spec-model_newton
chisq=np.sum( (resid/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(m)," degrees of freedom.")