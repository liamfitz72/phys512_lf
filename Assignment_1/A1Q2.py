# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:00:22 2022

@author: liamf
"""

import numpy as np
from matplotlib import pyplot as plt

eps=10**(-16)  #Machine precision error

def ndiff(fun, x, full=False):
    dx=10**-5
    
    y2=fun(x+2*dx)
    y1=fun(x+dx)
    y_1=fun(x-dx)
    y_2=fun(x-2*dx)
    d3=(y2-2*y1+2*y_1-y_2)/(2*dx**3) #3rd num. deriv. with dx rough estimate
    
    dx=np.abs(eps*fun(x)/d3)**(1/3) #Estimate for optimal dx using 3rd deriv.
    d=(fun(x+dx)-fun(x-dx))/(2*dx)
    err=eps*fun(x)/dx+d3*dx**2
    
    if full==True:
        return d, dx, err
    return d

x=np.linspace(-10,10,1001)
def fun(x):
    return x**3

deriv=ndiff(fun, x, True)

plt.plot(x, fun(x))
plt.plot(x, deriv[0])
#plt.legend([r'$f(x)=x^3$', r'Numerical deriv. of f(x)'])
#plt.title('Numerical derivative computation for x as an array')
#plt.savefig('A1Q2_plot2.png')