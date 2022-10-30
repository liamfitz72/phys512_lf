# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:19:35 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
plt.rcParams['figure.dpi'] = 200


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

param_names=[r'Hubble constant $H_0$',
             r'Baryon density $\Omega_b h^2$',
             r'Dark matter density $\Omega_c h^2$',
             r'Optical depth $\tau$',
             r'Primordial amplitude $A_s$',
             r'Primordial tilt $n_s$',
             ]

plt.plot(steps,chain[:,0])
plt.legend([param_names[0]])
plt.savefig('A5_plot5.png',bbox_inches='tight')
plt.clf()
fft=np.fft.fft(chain[:,0][7500:])
fftfreq=np.fft.fftfreq(len(chain[:,0][7500:]))
plt.scatter(fftfreq,fft,s=2)
plt.xscale('log')
plt.yscale('log')
plt.clf()

plt.plot(steps,chain[:,1])
plt.legend([param_names[1]])
plt.savefig('A5_plot6.png',bbox_inches='tight')
plt.clf()
fft=np.fft.fft(chain[:,1][7500:])
fftfreq=np.fft.fftfreq(len(chain[:,1][7500:]))
plt.scatter(fftfreq,fft,s=2)
plt.xscale('log')
plt.yscale('log')
plt.clf()

plt.plot(steps,chain[:,2])
plt.legend([param_names[2]])
plt.savefig('A5_plot7.png',bbox_inches='tight')
plt.clf()
fft=np.fft.fft(chain[:,2][7500:])
fftfreq=np.fft.fftfreq(len(chain[:,2][7500:]))
plt.scatter(fftfreq,fft,s=2)
plt.xscale('log')
plt.yscale('log')
plt.clf()

plt.plot(steps,chain[:,3])
plt.legend([param_names[3]])
plt.savefig('A5_plot8.png',bbox_inches='tight')
plt.clf()
fft=np.fft.fft(chain[:,3][7500:])
fftfreq=np.fft.fftfreq(len(chain[:,3][7500:]))
plt.scatter(fftfreq,fft,s=2)
plt.xscale('log')
plt.yscale('log')
plt.clf()

plt.plot(steps,chain[:,4])
plt.legend([param_names[4]])
plt.savefig('A5_plot9.png',bbox_inches='tight')
plt.clf()
fft=np.fft.fft(chain[:,4][7500:])
fftfreq=np.fft.fftfreq(len(chain[:,4][7500:]))
plt.scatter(fftfreq,fft,s=2)
plt.xscale('log')
plt.yscale('log')
plt.clf()

plt.plot(steps,chain[:,5])
plt.legend([param_names[5]])
plt.savefig('A5_plot10.png',bbox_inches='tight')
plt.clf()
fft=np.fft.fft(chain[:,5][7500:])
fftfreq=np.fft.fftfreq(len(chain[:,5][7500:]))
plt.scatter(fftfreq,fft,s=2)
plt.xscale('log')
plt.yscale('log')
# plt.clf()

# corner.corner(chain[7500:],labels=param_names,fontsize=15)
# plt.savefig('A5_plot11.png',bbox_inches='tight')
# plt.clf()


# H0=np.mean(chain[:,0][7500:])
# H0_err=np.std(chain[:,0][7500:])
# baryon_h2=np.mean(chain[:,1][7500:])
# baryon_h2_err=np.std(chain[:,1][7500:])
# print('Baryon density parameter is:',baryon_h2,'+/-',baryon_h2_err)
# dark_matter_h2=np.mean(chain[:,2][7500:])
# dark_matter_h2_err=np.std(chain[:,2][7500:])
# print('Dark matter density parameter is:',dark_matter_h2,'+/-',dark_matter_h2_err)

# H0_pow_neg2_err=2*H0_err/H0**3
# baryon_density=(100/H0)**2*baryon_h2
# baryon_density_err=np.sqrt((H0_pow_neg2_err/H0**(-2))**2+(baryon_h2/baryon_h2_err)**2)*baryon_density/100**2
# print('Baryon density is:',baryon_density,'+/-',baryon_density_err)

# dark_matter_density=(100/H0)**2*dark_matter_h2
# dark_matter_density_err=np.sqrt((H0_pow_neg2_err/H0**(-2))**2+(dark_matter_h2/dark_matter_h2_err)**2)*dark_matter_density/100**2
# print('Dark matter density is:',dark_matter_density,'+/-',dark_matter_density_err)


# dark_energy=1-baryon_density-dark_matter_density
# dark_energy_err=np.sqrt(baryon_density_err**2+dark_matter_density_err**2)

# print('Dark energy density is:',dark_energy,'+/-',dark_energy_err)








