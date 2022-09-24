# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:13:49 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#Code from class to get legendre coeffs
def legendre_coeffs(order):
    x=np.linspace(-1,1,order+1)
    mat=np.polynomial.legendre.legvander(x,order)
    minv=np.linalg.inv(mat)
    coeffs=minv[0,:]
    return coeffs


#Code from class to legendre integrate 
def legendre_integrate(fun,a,b,order,npt_targ,coeffs):
    ninterval=int((npt_targ-1)/order)
    npt_use=ninterval*order+1
    x=np.linspace(a,b,npt_use)
    y=fun(x)
    dx=(b-a)/(npt_use-1)
    tot=0
    for i in range(ninterval):
        i1=i*order
        i2=(i+1)*order+1
        pts=y[i1:i2]
        tot=tot+np.sum(coeffs*pts)
    ans=tot*dx*order
    return ans

def integrand(u,z,R):
    return (z+R*u)/(z**2+R**2+2*R*z*u)**(3/2)

def E_field_legendre(z,R,order,npt_targ):
    E_i=[]
    coeffs=legendre_coeffs(order)
    for i in range(len(z)):  #Loop over all z_i value, integrating for each
        def fun(u):  
            return integrand(u,z[i],R)  #Define function which is integrand for each z_i value
        E_i.append(R**2/2*legendre_integrate(fun,-1,1,order,npt_targ,coeffs))
    return E_i

def E_field_quad(z,R):
    E_i=[]
    for i in range(len(z)):
        def fun(u):
            return integrand(u,z[i],R)
        E_i.append(R**2/2*integrate.quad(fun,-1,1)[0])
    return E_i



R=1
a=0
b=2*R
z=np.linspace(a,b,1001)

order=10
npt_targ=1001
plt.plot(z,E_field_legendre(z,R,order,npt_targ))
plt.plot(z,E_field_quad(z,R))
plt.xlabel('z')
plt.ylabel('E(z)')
plt.legend(['Legendre integration (order=10, npt_targ=1001)','scipy integrate.quad'])
plt.savefig('A2Q1_plot1.png')
plt.clf()

order=10
npt_targ=10001
plt.plot(z,E_field_legendre(z,R,order,npt_targ))
plt.plot(z,E_field_quad(z,R))
plt.xlabel('z')
plt.ylabel('E(z)')
plt.legend(['Legendre integration (order=10, npt_targ=10001)','scipy integrate.quad'])
plt.savefig('A2Q1_plot2.png')
plt.clf()

order=10
npt_targ=10001
z1=np.linspace(a,0.99*R,501)  #Splitting up domain to remove singularity
z2=np.linspace(1.01*R,b,501)
E1=E_field_legendre(z1,R,order,npt_targ)
E2=E_field_legendre(z2,R,order,npt_targ)
E=E_field_legendre(z,R,order,npt_targ)
plt.plot(z1,E1)
plt.plot(z2,E2)
plt.plot(z,E_field_quad(z,R),zorder=-10)
plt.xlabel('z')
plt.ylabel('E(z)')
plt.legend(['Legendre integration z<0.99*R','Legendre integration z>1.01*R','scipy.integrate.quad'])
plt.savefig('A2Q1_plot3.png')


