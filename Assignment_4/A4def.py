# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:39:54 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

# Part D) and E)

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

def lorentz(m,t):  # New function to fit
    lorentz_left=m[3]/(1+(t-m[1]+m[5])**2/m[2]**2)
    lorentz_main=m[0]/(1+(t-m[1])**2/m[2]**2)
    lorentz_right=m[4]/(1+(t-m[1]-m[5])**2/m[2]**2)
    return lorentz_left+lorentz_main+lorentz_right

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
m0=[1,0.0002,0.00002,0.4,0.4,0.00005]
m=newton_lorentz(t,y,m0,n)
pred,A=calc_lorentz(m,t)

plt.scatter(t,y-pred,s=1)
plt.xlabel('t')
plt.xticks([0.0000,0.0001,0.0002,0.0003,0.0004])
plt.legend(['Residuals of least squares fit'])
plt.savefig('A4_plot4.png')
plt.clf()

noise_est=np.mean(np.abs(y-pred))   # Noise estimate from residuals
N_inv=noise_est**(-2)*np.identity(len(m))  # Match shape of A.T@A, can put N to front of
cov=np.linalg.inv(N_inv@A.T@A)             # equation since N=const*identity (avoid memory error)
m_err=np.sqrt(np.diag(cov))

print('Parameters are:''\na =',m[0],'+/-',m_err[0],
      '\nt_0 =',m[1],'+/-',m_err[1],
      '\nw =',m[2],'+/-',m_err[2],
      '\nb =',m[3],'+/-',m_err[3],
      '\nc =',m[4],'+/-',m_err[4],
      '\ndt =',m[5],'+/-',m_err[5],
      )


# Part F) 

def chi_sq(y,m,t,noise_est): 
    pred=calc_lorentz(m,t)[0]
    diff=np.matrix(y-pred).T
    chi_sq=1/noise_est**2*diff.T@diff  # Can pull N_inv out in front as a constant
    return chi_sq

num=25  # Generate 25 model realizations
list_chi_sq=[]
for i in range(num):
    n_rand=np.random.normal(loc=0.0,scale=noise_est,size=len(t))  # Random noise
    n_rand=np.matrix(n_rand).T  
    m_err_rlz=cov@N_inv@A.T@n_rand  # Can pull N_inv out in front
    m_rlz=np.empty(len(m))
    for j in range(len(m)):
        m_rlz[j]=m[j]+float(m_err_rlz[j])  # realized random m
    list_chi_sq.append(chi_sq(y,m_rlz,t,noise_est))
    pred_rlz=calc_lorentz(m_rlz,t)[0]
    
    plt.scatter(t,y-pred_rlz,s=1)  # Plot different realizations to compare
    plt.legend([i])
    if i==0:
        plt.savefig('A4_plot5.png')
    if i==24:
        plt.savefig('A4_plot6.png')
    plt.show()
    plt.cla()
plt.clf()

mean,std=(np.mean(list_chi_sq),np.std(list_chi_sq))
print('\nMean and standard deviation of chi squared for 25 model realizations:\n',
      mean,'+/-',std)

print('Chi squared for actual model fit:\n',float(chi_sq(y,m,t,noise_est)),'\n')


