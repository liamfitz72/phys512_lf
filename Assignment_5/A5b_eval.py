# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:52:21 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from A5a import get_spectrum

planck_fit_params=np.loadtxt('planck_fit_params.txt',delimiter=',')
m=planck_fit_params[:,0]
m_err=planck_fit_params[:,1]

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])
model_newton=get_spectrum(m)
model_newton=model_newton[:len(spec)]
resid=spec-model_newton


plt.plot(ell,model_newton)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.legend(['Newton method fit'])
plt.savefig('A5_plot3.png',bbox_inches='tight')
plt.clf()

plt.scatter(ell,resid,s=2)
plt.scatter
plt.legend(['Newton method fit residuals'])
plt.savefig('A5_plot4.png',bbox_inches='tight')