# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:58:03 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
plt.rcParams['figure.dpi']=200

def relax_iterate(rho,niter,plot=False):
    n=len(rho[:,0])
    V=np.zeros([n,n])
    for i in tqdm(range(niter)):
        Vavg=1/4*(np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1))
        Vnew=rho+Vavg
        V=Vnew
        V=V+(1-V.max())  # offset potential to keep V[0,0]=1
        if i%10==0:
            plt.clf()
            plt.imshow(np.fft.fftshift(V),cmap='Reds')  # fftshift to plot negative indices
            plt.colorbar()
            plt.title('Iteration '+str(i)+'/'+str(niter))
            plt.pause(0.2)
    return V

n=100
rho=np.zeros([n,n])
rho[0,0]=1
niter=n*20

fig1=plt.figure(1)
V=relax_iterate(rho,niter,True)

print('After '+str(niter)+' iterations',
      '\nV[1,0] =',V[1,0],
      '\nV[2,0] =',V[2,0],
      '\nV[5,0] =',V[5,0])

plt.savefig('A7Q2_plot1.png',bbox_inches='tight')

rho_inferred=-scipy.ndimage.filters.laplace(V)
fig2=plt.figure(2)
plt.imshow(np.fft.fftshift(rho_inferred),cmap='Blues')

V_inferred=scipy.signal.convolve2d(V,rho_inferred,'same')
fi3=plt.figure(3)
plt.imshow(np.fft.fftshift(V_inferred),cmap='Reds')

V_new=V-V_inferred
fig4=plt.figure(4)
plt.imshow(np.fft.fftshift(V_new),cmap='Reds')

plt.colorbar()