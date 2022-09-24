# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:43:59 2022

@author: liamf
"""

import numpy as np

#Modyfying the adaptive integrate function from class, extra used to 'store' repeat function calls
def integrate_adaptive(fun,a,b,tol,extra=None): 
    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    if extra==None:  #Has no previous function calls stored
        y=fun(x)
    else:
        y=[extra[0],fun(x[1]),extra[1],fun(x[2]),extra[2]] #Mid and endpoints re-used from previous
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    if myerr<tol:
        return i2
    else:
        mid=(a+b)/2
        #Store mid and endpoints for next iteration
        int1=integrate_adaptive(fun,a,mid,tol/2,extra=[y[0],y[1],y[2]]) 
        int2=integrate_adaptive(fun,mid,b,tol/2,extra=[y[2],y[3],y[4]])
        return int1+int2
    
fun=np.cos
print('Integral of cos(x) from 0 to pi/2:', integrate_adaptive(fun,0,np.pi/2,0.01))