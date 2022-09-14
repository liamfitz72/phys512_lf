# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 22:01:20 2022

@author: liamf
"""

import numpy as np
from matplotlib import pyplot as plt

logdelta=np.linspace(-8,-1,1001)
delta=10**logdelta #Create points space apart exponentially

fun=np.exp
x0=1
eps=10**(-16) #Machine precision error

y0_1=fun(x0-delta)
y1_1=fun(x0+delta)
d_1=(y1_1-y0_1)/(2*delta)  #Derivative operator from +/-delta

y0_2=fun(x0-2*delta)
y1_2=fun(x0+2*delta)
d_2=(y1_2-y0_2)/(4*delta)  #Derivative operator from +/-2*delta

d=4/3*d_1-1/3*d_2  #Combined derivative operator cancels delta^2 terms

delta_est=10**(-3.2) #Estimate of delta with minimum error
approx_err=fun(x0)*delta_est**4  #Approximate error arising from delta

plt.loglog(delta,np.abs(d-np.exp(x0)), label='Numerical Error')
plt.scatter(delta_est, approx_err, color='red', label='Minimum Error Estimation')
plt.xlabel(r'Step size ($\delta$)')
plt.ylabel('Error')
plt.title(r'Error as a function of step size for $f(x)=e^x$ at x=1')
plt.legend()
plt.savefig('A1Q1_plot1.png')