# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:39:25 2022

@author: liamf
"""

import numpy as np
from scipy import interpolate as interp
from matplotlib import pyplot as plt

ord=10

def lorentz(x):
    return 1/(1+x**2)

def poly_interp(fun, x_fine, x):
    p_sum=0
    for i in range(len(x)):
        x_use=np.append(x[:i],x[i+1:])
        x0=x[i]
        mynorm=np.prod(x0-x_use)
        p0=1.0
        for xi in x_use:
            p0=p0*(xi-x_fine)
        p0=p0/mynorm*fun(x[i])
        p_sum+=p0
    return p_sum

def spline_interp(fun, x_fine, x):
    y=fun(x)
    spl=interp.splrep(x,y)
    y_fine=interp.splev(x_fine,spl)
    return y_fine

def rational_interp(fun, x_fine, x, n):
    m=ord-n
    y=fun(x)
    
    pcols=[x**k for k in range(n+1)]
    pmat=np.vstack(pcols)

    qcols=[-x**k*y for k in range(1,m+1)]
    qmat=np.vstack(qcols)
    mat=np.hstack([pmat.T,qmat.T])
    coeffs=np.linalg.inv(mat)@y
    
    p=0
    for i in range(n+1):
        p=p+coeffs[i]*x_fine**i
    qq=1
    for i in range(m):
        qq=qq+coeffs[n+1+i]*x_fine**(i+1)
    y_pred=p/qq
    return y_pred
 

#Cosine interpolation   
x=np.linspace(-np.pi/2,np.pi/2,ord+1)
x_fine=np.linspace(x[0],x[-1],1001)

plt.scatter(x, np.cos(x))
plt.plot(x_fine, poly_interp(np.cos,x_fine,x), color='orange')
plt.savefig('A1Q4_cos_polyfit.png')
plt.clf()

err=np.abs(np.cos(x_fine)-poly_interp(np.cos,x_fine,x))
plt.plot(x_fine, err)
plt.savefig('A1Q4_cos_polyfiterr.png')
print('Max poly error on cos is', np.max(err))
plt.clf()

plt.scatter(x, np.cos(x))
plt.plot(x_fine, spline_interp(np.cos,x_fine,x), color='orange')
plt.savefig('A1Q4_cos_splinefit.png')
plt.clf()

err=np.abs(np.cos(x_fine)-spline_interp(np.cos,x_fine,x))
plt.plot(x_fine, err)
plt.savefig('A1Q4_cos_splinefiterr.png')
print('Max spline error on cos is', np.max(err))
plt.clf()

n=4
plt.scatter(x, np.cos(x))
plt.plot(x_fine, rational_interp(np.cos,x_fine,x,n), color='orange')
plt.savefig('A1Q4_cos_ratfit.png')
plt.clf()

err=np.abs(np.cos(x_fine)-rational_interp(np.cos,x_fine,x,n))
plt.plot(x_fine, err)
plt.savefig('A1Q4_cos_ratfiterr.png')
print('Max ratfit error on cos is', np.max(err))
plt.clf()


#Lorentz interpolation
ord=5
x=np.linspace(-1,1,ord+1)
x_fine=np.linspace(x[0],x[-1],1001)

plt.scatter(x, lorentz(x))
plt.plot(x_fine, np.abs(poly_interp(lorentz,x_fine,x)), color='orange')
plt.savefig('A1Q4_lorentz_polyfit.png')

plt.clf()
err=np.abs(lorentz(x_fine)-np.abs(poly_interp(lorentz,x_fine,x)))
plt.plot(x_fine, err)
plt.savefig('A1Q4_lorentz_polyfiterr.png')
print('Max poly error on lorentz is', np.max(err))

plt.clf()
plt.scatter(x, lorentz(x))
plt.plot(x_fine, spline_interp(lorentz,x_fine,x), color='orange')
plt.savefig('A1Q4_lorentz_splinefit.png')

plt.clf()
err=np.abs(lorentz(x_fine)-spline_interp(lorentz,x_fine,x))
plt.plot(x_fine, err)
plt.savefig('A1Q4_lorentz_splinefiterr.png')
print('Max lorentz error on lorentz is', np.max(err))

plt.clf()
n=3 
plt.scatter(x, lorentz(x))
plt.plot(x_fine, rational_interp(lorentz,x_fine,x,n), color='orange')
plt.savefig('A1Q4_lorentz_ratfit.png')

plt.clf()
err=np.abs(lorentz(x_fine)-rational_interp(lorentz,x_fine,x,n))
plt.plot(x_fine, err)
plt.savefig('A1Q4_lorentz_ratfiterr.png')
print('Max ratfit error on lorentz is', np.max(err))