# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:39:54 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
from tqdm import tqdm

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

def chi_sq(y,m,t,noise_est):
    errs=[noise_est]*len(t)
    pred=calc_lorentz(m,t)[0]
    chisq=np.sum(np.power(pred-y,2)/np.power(errs,2))
    return chisq

stuff=np.load('sidebands.npz')
t=stuff['time']
y=stuff['signal']

n=20
m0=[1,0.0002,0.00002,0.4,0.4,0.00005]
m=newton_lorentz(t,y,m0,n)
pred,A=calc_lorentz(m,t)

plt.scatter(t,y,s=0.001)
plt.plot(t,pred,color='orange')
plt.xlabel('t')
plt.xticks([0.0000,0.0001,0.0002,0.0003,0.0004])
plt.legend(['Data',"Least squares fit using \nNewton's Method"])
plt.savefig('A4_plot14.png')
plt.clf()

plt.scatter(t,y-pred,s=1)
plt.xlabel('t')
plt.xticks([0.0000,0.0001,0.0002,0.0003,0.0004])
plt.legend(['Residuals of least squares fit'])
plt.savefig('A4_plot4.png')
plt.clf()

noise_est=np.std(y[:10000])   # Noise estimate from residuals
N_inv=noise_est**(-2)*np.identity(len(m))  # Match shape of A.T@A, can put N to front of
cov=np.linalg.inv(N_inv@A.T@A)             # equation since N=const*identity (avoid memory error)
m_err=np.sqrt(np.diag(cov))

print('Newton method fit parameters are:''\na =',m[0],'+/-',m_err[0],
      '\nt_0 =',m[1],'+/-',m_err[1],
      '\nw =',m[2],'+/-',m_err[2],
      '\nb =',m[3],'+/-',m_err[3],
      '\nc =',m[4],'+/-',m_err[4],
      '\ndt =',m[5],'+/-',m_err[5],
      '\nChi squared is:',
      'Chi-sq = ',float(chi_sq(y,m,t,noise_est))) 


# Part F) 


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
    plt.xticks([0.0000,0.0001,0.0002,0.0003,0.0004])
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

print('Chi squared for Newton method fit fit:\n',float(chi_sq(y,m,t,noise_est)),'\n')


# Part G)

def lorentz_chisq(m,data,noise_est): 
    t=data['time']
    y=data['signal']
    errs=[noise_est]*len(t)
    
    pred=calc_lorentz(m,t)[0]
    chisq=np.sum(np.power(pred-y,2)/np.power(errs,2))
    return chisq

def MCMC_chain(fun_chisq,data,start_params,noise_est,scale,nstep=10000,T=1):
    nparam=len(start_params)
    chain=np.zeros([nstep,nparam])
    chisq=np.zeros(nstep)
    chain[0,:]=start_params
    cur_chisq=fun_chisq(start_params,data,noise_est)
    chisq[0]=cur_chisq
    params=start_params
    for i in tqdm(range(1,nstep)):
        trial_params=params+np.random.randn(len(params))*scale
        new_chisq=fun_chisq(trial_params,data,noise_est)
        accept_prob=np.exp(-0.5*(new_chisq-cur_chisq)/T)
        if np.random.rand(1)<accept_prob:
            params=trial_params
            cur_chisq=new_chisq
        chain[i,:]=params
        chisq[i]=cur_chisq
    return chain,chisq

def chain_eval(chain,chisq,T=1):
    dchi=chisq-np.min(chisq)
    wt=np.exp(-0.5*dchi*(1-1/T))
    npar=chain.shape[1]
    tot=np.zeros(npar)
    totsqr=np.zeros(npar)
    for i in range(npar):
        tot[i]=np.sum(wt*chain[:,i])
        totsqr[i]=np.sum(wt*chain[:,i]**2)
    mean=tot/np.sum(wt)
    meansqr=totsqr/np.sum(wt)
    var=meansqr-mean**2
    return mean,np.sqrt(var),wt
    
data=np.load('sidebands.npz')
t=data['time']
y=data['signal']
m0=[1.4,0.0002,0.00002,0.1,0.1,0.00005]
noise_est=np.std(y[:10000])
scale=5*m_err

nstep=20001
chain,chisq=MCMC_chain(lorentz_chisq,data,m0,noise_est,scale,nstep=nstep)

x=np.linspace(0,20000,nstep)
param_names=[r'$a$',r'$t_0$',r'$w$',r'$b$',r'$c$',r'$dt$']
for i in range(len(m0)):
    plt.figure()
    plt.scatter(x,chain[:,i],s=2)
    plt.legend([param_names[i]])
    plt.savefig('A4_plot'+str(7+i)+'.png',bbox_inches='tight')
    plt.clf()

m,m_err=chain_eval(chain[10000:],chisq[10000:])[:2] # Start from where converged
pred,A=calc_lorentz(m,t)
plt.scatter(t,y-pred,s=2)
plt.legend(['20000 step MCMC fit residuals'])
plt.xlabel('t')
plt.xticks([0.0000,0.0001,0.0002,0.0003,0.0004])
plt.savefig('A4_plot13',bbox_inches='tight')

print('MCMC fit parameters are:''\na =',m[0],'+/-',m_err[0],
      '\nt_0 =',m[1],'+/-',m_err[1],
      '\nw =',m[2],'+/-',m_err[2],
      '\nb =',m[3],'+/-',m_err[3],
      '\nc =',m[4],'+/-',m_err[4],
      '\ndt =',m[5],'+/-',m_err[5],
      '\nChi squared is:',
      'Chi-sq = ',chi_sq(y,m,t,noise_est))



