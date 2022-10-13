# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:39:54 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt

# Part C) #

def lorentz(m,t):
    return m[0]/(1+(t-m[1])**2/m[2]**2)

def param_grad(fun,m,t):  # Get parameter gradient of function numerically
    dx=10**-5
    derivs=[]
    for i in range(len(m)):
        m_c=m.copy()
        m_c[i]+=dx    # m_i -> m_i + dm_i
        y1=fun(m_c,t)
        m_c[i]-=2*dx  # m_i -> m_i - dm_i
        y_1=fun(m_c,t) 
        d=(y1-y_1)/(2*dx)  # Central difference
        derivs.append(d)
    return np.matrix(derivs).T  # Transpose fit with Newton method function

def calc_lorentz(m,t):
    y=lorentz(m,t)
    grad=param_grad(lorentz,m,t)
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
m0=[1,0.0002,0.00002]
m=newton_lorentz(t,y,m0,n)
pred,A=calc_lorentz(m,t)

plt.scatter(t,y-pred,s=1)
plt.xlabel('t')
plt.xticks([0.0000,0.0001,0.0002,0.0003,0.0004])
plt.legend(['Residuals of least squares fit \n(numerical differentiation)'])
plt.savefig('A4_plot3.png')
plt.clf()

noise_est=np.mean(np.abs(y-pred))   # Noise estimate from residuals
N=noise_est**(-2)*np.identity(len(m))  # Match shape of A.T@A, can put N to front of
cov=np.linalg.inv(N@A.T@A)             # equation since N=const*identity (avoid memory error)
m_err=np.sqrt(np.diag(cov))

print('Parameters are:''\na =',m[0],'+/-',m_err[0],
      '\nt_0 =',m[1],'+/-',m_err[1],
      '\nw =',m[2],'+/-',m_err[2])
