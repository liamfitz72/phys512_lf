# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:39:54 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt

# Part A) #

def calc_lorentz(m,t):  # Same function from class, different derivatives
    a,t_0,w=m
    y=a/(1+(t-t_0)**2/w**2)
    grad=np.zeros([len(t),len(m)])
    grad[:,0]=1/(1+(t-t_0)**2/w**2)
    grad[:,1]=a/(1+(t-t_0)**2/w**2)**2*2*(t-t_0)/w**2     # Analytic derivatives
    grad[:,2]=a/(1+(t-t_0)**2/w**2)**2*2*(t-t_0)**2/w**3
    return y,grad

def newton_lorentz(t,y,m0,n):  # Same code for Newtons method from class
    m=m0.copy()
    for i in range(n):
        pred,grad=calc_lorentz(m,t)
        r=y-pred
        r=np.matrix(r).T
        grad=np.matrix(grad)
        lhs=grad.T@grad
        rhs=grad.T@r
        dm=np.linalg.inv(lhs)*rhs
        for j in range(len(m)):
            m[j]=m[j]+float(dm[j])  # Keep m as list type to pass through calc_lorentz
    return m

stuff=np.load('sidebands.npz')
t=stuff['time']
y=stuff['signal']

n=20
m0=[1,0.0002,0.00002]   # Initial guess
m=newton_lorentz(t,y,m0,n)
pred,A=calc_lorentz(m,t)

plt.scatter(t,y,s=0.001)
plt.plot(t,pred,color='orange')
plt.xlabel('t')
plt.xticks([0.0000,0.0001,0.0002,0.0003,0.0004])
plt.legend(['Data',"Least squares fit using \nNewton's Method"])
plt.savefig('A4_plot1.png')
plt.clf()

plt.scatter(t,y-pred,s=1)
plt.xlabel('t')
plt.xticks([0.0000,0.0001,0.0002,0.0003,0.0004])
plt.legend(['Residuals of least squares fit'])
plt.savefig('A4_plot2.png')
plt.clf()


# Part B) #

noise_est=np.mean(np.abs(y-pred))   # Noise estimate from residuals
N_inv=noise_est**(-2)*np.identity(len(m))  # Match shape of A.T@A, can put N to front of
cov=np.linalg.inv(N_inv@A.T@A)             # equation since N=const*identity (avoid memory error)
m_err=np.sqrt(np.diag(cov))

print('Parameters are:''\na =',m[0],'+/-',m_err[0],
      '\nt_0 =',m[1],'+/-',m_err[1],
      '\nw =',m[2],'+/-',m_err[2])



