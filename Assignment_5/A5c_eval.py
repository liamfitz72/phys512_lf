# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:19:35 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
from A5a import get_spectrum
plt.rcParams['figure.dpi'] = 200

def chain_params(chain):
    m=np.mean(chain,axis=0)
    m_err=np.std(chain,axis=0)
    return m,m_err

def plot_and_pspec(chain,param_names,save_name,save_plots=True):
    for i in range(len(chain[0,:])):
        steps=np.linspace(0,len(chain[:,0]),len(chain[:,0]))
        plt.plot(steps,chain[:,i])
        plt.legend([param_names[i]])
        if save_plots:
            plt.savefig(save_name+'_chain_param'+str(i)+'.png',bbox_inches='tight')
        plt.clf()
        fft=np.fft.fft(chain[:,i])
        fftfreq=np.fft.fftfreq(len(chain[:,i]))
        plt.scatter(fftfreq,fft,s=5)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend([param_names[i]+' power spectrum'])
        if save_plots:
            plt.savefig(save_name+'_chain_param'+str(i)+'_pspec.png',bbox_inches='tight')
        plt.clf()

if __name__=='__main__':
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3])

    planck_chain=np.loadtxt('planck_chain.txt',delimiter=',')
    chisq,chain=(planck_chain[:,0],planck_chain[:,1:])
    
    steps=np.linspace(0,19999,20000)
    plt.plot(steps,chisq,linewidth=1)
    plt.xlabel('Steps')
    plt.legend([r'MCMC chain $\chi^2$ values'])
    plt.savefig('A5_plot3.png',bbox_inches='tight')
    plt.clf()
    
    plt.plot(steps[7500:],chisq[7500:],linewidth=1)
    plt.xlabel('Steps')
    plt.legend([r'MCMC chain $\chi^2$ values (step 7500 onwards)'])
    plt.savefig('A5_plot4.png',bbox_inches='tight')
    plt.clf()
    
    param_names=[r'$H_0$',
                 r'$\Omega_b h^2$',
                 r'$\Omega_c h^2$',
                 r'$\tau$',
                 r'$A_s$',
                 r'$n_s$',
                 ]
    
    plot_and_pspec(chain,param_names,'A5c')
    
    corner.corner(chain,labels=param_names,fontsize=15)
    plt.savefig('A5_plot5.png',bbox_inches='tight')
    plt.clf()
    
    m,m_err=chain_params(chain)
    MCMC_params=np.empty([len(m),2])
    MCMC_params[:,0]=m
    MCMC_params[:,1]=m_err
    resid=spec-get_spectrum(m)[:len(spec)]
    chi_sq=np.sum((resid/errs)**2)
    print('MCMC fit chisq:',chi_sq)
    np.savetxt('planck_params_MMCMC.txt',MCMC_params,delimiter=',')
    
    H0_pow_neg2_err=2*m_err[0]/m[0]**3
    baryon_density=(100/m[0])**2*m[1]
    baryon_density_err=np.sqrt((H0_pow_neg2_err/m[0]**(-2))**2+(m[1]/m_err[1])**2)*baryon_density/100**2
    print('Baryon density is:',baryon_density,'+/-',baryon_density_err)
    
    dark_matter_density=(100/m[0])**2*m[2]
    dark_matter_density_err=np.sqrt((H0_pow_neg2_err/m[0]**(-2))**2+(m[2]/m_err[2])**2)*dark_matter_density/100**2
    print('Dark matter density is:',dark_matter_density,'+/-',dark_matter_density_err)
    
    
    dark_energy=1-baryon_density-dark_matter_density
    dark_energy_err=np.sqrt(baryon_density_err**2+dark_matter_density_err**2)
    
    print('Dark energy density is:',dark_energy,'+/-',dark_energy_err)
    
    
    
    
    
    
    
    
