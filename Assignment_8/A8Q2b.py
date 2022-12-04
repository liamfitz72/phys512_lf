# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:50:06 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


def avg_neighbors(mat):
    up=np.roll(mat,1,0)
    down=np.roll(mat,-1,0)
    right=np.roll(mat,1,1)
    left=np.roll(mat,-1,1)
    avg=1/4*(up+down+right+left)
    return avg
    
def mat_mult(x,mask):
    x[mask]=0  # We only care about interior, apply mask
    avg=avg_neighbors(x)
    avg[mask]=0  # Boundaries become non-zero after avg'ing
    return x-avg

def conjgrad(b,xinit,mask,niter,plot=False):  # CG method from class
    r=b-mat_mult(xinit,mask)
    p=r.copy()
    rr=np.sum(r**2)
    x=xinit
    for i in range(niter):
        Ap=mat_mult(p,mask)
        pAp=np.sum(p*Ap)  # Can't use @ for dot product when not vectors
        alpha=rr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rr_new=np.sum(r**2) # Same for this dot product
        beta=rr_new/rr
        p=r+beta*p
        rr=rr_new
        b=mat_mult(x,mask) # Calculate and plot rho
        if plot:
            if i%5==0:
                plt.clf()
                plt.imshow(b)
                plt.colorbar(label='Charge density')
                plt.title('Iteration '+str(i))
                plt.pause(0.01)
    return x,b

n=101
mask=np.zeros([n,n],dtype='bool') # Mask for edges and inside box
mask[0,:]=True  
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True
mask[2*n//5:3*n//5,2*n//5:3*n//5]=True
bc=np.zeros([n,n])  # Potential bc's on edges of box
bc[2*n//5,2*n//5:3*n//5]=1
bc[3*n//5-1,2*n//5:3*n//5]=1
bc[2*n//5:3*n//5,2*n//5]=1
bc[2*n//5:3*n//5,3*n//5-1]=1

bc_rhs=avg_neighbors(bc)  # Move boundary conditions to RHS
bc_rhs[mask]=0  
b=bc_rhs
V,rho=conjgrad(b,0*b,mask,niter=n)
V[mask]=bc[mask] # Add back BC's after zeroing with mask

plt.imshow(bc+mask)
plt.savefig('A8Q2b_bc&mask.png',bbox_inches='tight')
plt.clf()

plt.imshow(V)
plt.colorbar(label='Potential')
plt.savefig('A8Q2b_Vout.png',bbox_inches='tight')
plt.clf()

plt.imshow(rho)
plt.colorbar(label='Charge density')
plt.savefig('A8Q2b_chargedens.png',bbox_inches='tight')
plt.clf()

plt.plot(rho[2*n//5-1,2*n//5:3*n//5])
plt.legend(['Charge density on side'])
plt.savefig('A8Q2b_sidechargedens')
plt.clf()


# Part C)
mask=np.zeros([n,n],dtype='bool') # Mask just for edges
mask[0,:]=True  
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True

Vall=conjgrad(rho,0*rho,mask,niter=n)[0]
plt.imshow(Vall)
plt.colorbar(label='Potential')
plt.savefig('A8Q2b_Vall.png',bbox_inches='tight')

Ex=Vall-np.roll(Vall,-1,1)
Ey=Vall-np.roll(Vall,-1,0)

plt.clf()
plt.quiver(Ex,Ey)
plt.savefig('A8Q2b_Efield.png',bbox_inches='tight')

Vin=Vall[2*n//5:3*n//5,2*n//5:3*n//5]
plt.clf()
plt.imshow(Vin)
plt.colorbar()
plt.savefig('A8Q2b_Vconst.png',bbox_inches='tight')

# plt.imshow(np.fft.fftshift(np.fft.ifftn(np.fft.fftn(gf)*np.fft.fftn(rho))).real)
# plt.colorbar()