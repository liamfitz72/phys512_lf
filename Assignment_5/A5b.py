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

def CMBmodel_param_grad(m):  # Get parameter gradient of model numerically
    grad=[]
    num=len(m)
    for i in range(num):
        dm_i=m[i]/1000  # Order of magnitudes difference among parameters, set dx proportional
        dm=np.zeros(num)
        dm[i]=dm_i
        y2=get_spectrum(m+dm)
        y1=get_spectrum(m-dm)
        ddm=(y2-y1)/(2*dm_i)  # Central difference
        grad.append(ddm)
    return np.matrix(grad).T  # Transpose fits with Newton method function

def CMBmodel_newton_min(m0,y_data,n):
    m=m0.copy()
    ndata=len(y_data)
    for i in tqdm(range(n)):
        pred=get_spectrum(m)[:ndata]
        grad=CMBmodel_param_grad(m)[:ndata]
        resid=np.matrix(y_data-pred).T
        grad=np.matrix(grad)
        lhs=grad.T@grad
        rhs=grad.T@resid
        hess_inv=np.linalg.pinv(lhs)
        dm=hess_inv@rhs   # Newton method matrix equation from notes
        for j in range(len(m)):
            m[j]=m[j]+float(dm[j])  # Keep m as list type to pass through calc_lorentz
    return m,hess_inv

def get_m_err(m,noise_est):
    N_inv=noise_est**(-2)  
    A=CMBmodel_param_grad(m)  # Calculate gradient to get covariance matrix
    cov=np.multiply(N_inv,np.linalg.inv(A.T@A))    # N=const*identity
    m_err=np.sqrt(np.diag(cov))
    return m_err

if __name__=="__main__":
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3])
    planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
    errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])
    
    m0=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
    y_data=spec
    n=10
    m,hess_inv=CMBmodel_newton_min(m0,y_data,n)
    model_newton=get_spectrum(m)
    model_newton=model_newton[:len(spec)]
    resid=spec-model_newton
    chisq=np.sum((resid/errs)**2)
    print("chisq is ",chisq," for ",len(resid)-len(m)," degrees of freedom.")
    
    noise_est=np.std(resid[1400:1900]) # Noise estimate from residuals
    m_err=get_m_err(m,noise_est)
    param_results=np.matrix([m,m_err]).T
    np.savetxt('planck_fit_params.txt',param_results, delimiter=',')
    np.savetxt('inverse_hessian.txt',hess_inv,delimiter=',')
    
    
    