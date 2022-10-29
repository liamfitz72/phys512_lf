# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:52:30 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
from tqdm import tqdm
from A5a import get_spectrum


def CMBmodel_param_grad(m,y_data):  # Get parameter gradient of model numerically
    n_data=len(y_data)
    n_m=len(m)
    A=np.empty([n_data,n_m])
    for i in range(n_m):
        dm_i=m[i]/10**8  # Orders of magnitudes differences among parameters, set dx proportional
        m_c=m.copy()
        m_c[i]=m[i]+dm_i
        y2=get_spectrum(m_c)[:n_data]
        m_c[i]=m[i]-dm_i
        y1=get_spectrum(m_c)[:n_data]
        A[:,i]=(y2-y1)/(2*dm_i)  # Central difference
    return A  # Transpose fits with Newton method function

def CMBmodel_newton_min(m0,y_data,errs,n):
    m=m0.copy()
    ndata=len(y_data)
    N=np.diag(errs**2)
    N_inv=np.linalg.inv(N)
    for i in tqdm(range(n)):
        pred=get_spectrum(m)[:ndata]
        grad=CMBmodel_param_grad(m,y_data)[:ndata]
        resid=y_data-pred
        lhs=grad.T@N_inv@grad
        rhs=grad.T@N_inv@resid
        cov=np.linalg.inv(lhs)
        dm=cov@rhs   # Newton method matrix equation from notes
        m=m+dm
    m_err=np.sqrt(np.diag(cov))
    return m,m_err,cov

if __name__=="__main__":
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3])
    planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
    errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])
    
    m0=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
    y_data=spec
    n=3
    m,m_err,curv=CMBmodel_newton_min(m0,y_data,errs,n)
    
    model_newton=get_spectrum(m)
    model_newton=model_newton[:len(y_data)]
    resid=y_data-model_newton
    chisq=np.sum((resid/errs)**2)
    print("chisq is ",chisq," for ",len(resid)-len(m)," degrees of freedom.")
    
    noise_est=np.std(resid[1400:1900]) # Noise estimate from residuals
    param_results=np.matrix([m,m_err]).T
    np.savetxt('planck_fit_params.txt',param_results, delimiter=',')
    np.savetxt('cov_matrix.txt',curv,delimiter=',')
    
    
    