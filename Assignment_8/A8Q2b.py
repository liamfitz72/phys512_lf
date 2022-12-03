# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:50:06 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

gf=np.loadtxt('greenfun.txt')
gf=np.fft.fftshift(gf)
n=len(gf[:,0])

def avg_neighbors(mat):
    avg=1/4*(np.roll(mat,-1,0)+np.roll(mat,1,0)+np.roll(mat,-1,1)+np.roll(mat,1,1))
    return avg

class grid:
    def __init__(self,bc,mask):
        self.bc=bc
        self.mask=mask
    def rhs_bc(self):
        rhs_bc=avg_neighbors(self.bc)
        rhs_bc[self.mask]=0
        return rhs_bc
    def __matmul__(self,x):
        x[self.mask]=0  # We only care about interior, apply mask
        avg=avg_neighbors(x)
        avg[self.mask]=0 # Boundaries become non-zero after avg'ing
        return x-avg
    
def conjgrad(A,b,xinit,niter,plot=False):
    r=b-A@xinit
    p=r.copy()
    rr=np.sum(r**2)
    x=xinit
    for i in range(niter):
        Ap=A@p
        pAp=np.sum(p*Ap)  # Can't use @ for dot product when not vectors
        alpha=rr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rr_new=np.sum(r**2) # Same for this dot product
        beta=rr_new/rr
        p=r+beta*p
        rr=rr_new
        if plot:
            plt.clf()
            plt.imshow(x)
            plt.colorbar()
            plt.pause(0.1)
    return x


n=101
mask=np.zeros([n,n],dtype='bool')
bc=np.zeros([n,n])
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True
mask[2*n//5:3*n//5,2*n//5:3*n//5]=True
bc[2*n//5:3*n//5,2*n//5:3*n//5]=1
# bc[2*n//5,n//4:(3*n)//4]=1.0
# bc[3*n//5,n//4:(3*n)//4]=-1.0

# mask[2*n//5,n//4:(3*n)//4]=True
# mask[3*n//5,n//4:(3*n)//4]=True




A=grid(bc,mask)
b=A.rhs_bc()
x=conjgrad(A,b,0*b,niter=3*n)
V=x.copy()
V[A.mask]=A.bc[A.mask]
rho=V-avg_neighbors(V)

plt.imshow(rho)
plt.colorbar()
