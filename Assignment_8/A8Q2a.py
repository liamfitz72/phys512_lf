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

n=100
V=np.zeros([n,n])
rho=np.zeros([n,n])
rho[0,0]=1
niter=n*20

for i in tqdm(range(niter)):
    Vavg=1/4*(np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1))
    Vnew=rho+Vavg
    V=Vnew
    V=V+(1-V.max())  # offset potential to keep V[0,0]=1
    if i%10==0:
        plt.clf()
        plt.imshow(np.fft.fftshift(V))  # fftshift to plot negative indices
        plt.colorbar()
        plt.pause(0.01)

print('After '+str(niter)+' iterations',
      '\nV[1,0] =',V[1,0],
      '\nV[2,0] =',V[2,0],
      '\nV[5,0] =',V[5,0])

plt.imshow(np.fft.fftshift(V))
plt.colorbar()
plt.savefig('A8Q2a_Vrelax.png',bbox_inches='tight')
np.savetxt('greenfun.txt',V)

# rho_inferred=-scipy.ndimage.filters.laplace(V)
# fig2=plt.figure(2)
# plt.imshow(np.fft.fftshift(rho_inferred),cmap='Blues')

# V_inferred=scipy.signal.convolve2d(V,rho_inferred,'same')
# fi3=plt.figure(3)
# plt.imshow(np.fft.fftshift(V_inferred),cmap='Reds')

# V_new=V-V_inferred
# fig4=plt.figure(4)
# plt.imshow(np.fft.fftshift(V_new),cmap='Reds')
# plt.colorbar()