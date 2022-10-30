# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:51:29 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from A5a import get_spectrum

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

newton_fit=np.loadtxt('planck_fit_params.txt',delimiter=',')
m=newton_fit[:,0]
m_err=newton_fit[:,1]
newton_model=get_spectrum(m)[:len(ell)]
resid=spec-newton_model

plt.errorbar(ell,spec,yerr=errs,fmt='none',elinewidth=0.75)
plt.plot(ell,newton_model,color='orange')
plt.xlabel('Multipole moment')
plt.ylabel('Power spectrum')
plt.legend(['CMB Data','Newton method least squares fit'])
plt.savefig('A5_plot1.png',bbox_inches='tight')
plt.clf()

plt.errorbar(ell,resid,yerr=errs,fmt='none',elinewidth=0.75)
plt.xlabel('Multipole moment')
plt.legend(['Newton method fit residuals'])
plt.savefig('A5_plot2.png',bbox_inches='tight')