# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:15:27 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from A5c_eval import chain_params,plot_and_pspec
from A5a import get_spectrum

import corner

if __name__=="__main__":
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3])
    
    data=np.loadtxt('planck_chain_tauprior.txt',delimiter=',')
    chisq,chain=(data[:,0],data[:,1:])
    
    m,m_err=chain_params(chain)
    newMCMC=np.empty([len(m),2])
    np.savetxt('planck_params_newMCMC',newMCMC,delimiter=',')
    resid=spec-get_spectrum(m)[:len(spec)]
    chi_sq=np.sum((resid/errs)**2)
    print('New MCMC chisq:',chi_sq)
    
    steps=np.linspace(0,19999,20000)
    
    param_names=[r'$H_0$',
                 r'$\Omega_b h^2$',
                 r'$\Omega_c h^2$',
                 r'$\tau$',
                 r'$A_s$',
                 r'$n_s$',
                 ]
    
    plot_and_pspec(chain,param_names,'A5d')
    matplotlib.rcParams.update({'font.size': 14})
    corner.corner(chain,labels=param_names,fontsize=15)
    plt.savefig('A5_plot6.png',bbox_inches='tight')
    matplotlib.rcParams.update({'font.size': 10})
    
    