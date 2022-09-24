# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:43:59 2022

@author: liamf
"""

import numpy as np

#Adding saved call count to adaptive integrate
def integrate_adaptive(fun,a,b,tol,extra=None):
    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    if extra==None: 
        calls_saved=0 #Initialize calls saved
        y=fun(x)
    else:
        calls_saved=extra[-1]  #Saved calls count carried through in extra variable
        y=[extra[0],fun(x[1]),extra[1],fun(x[3]),extra[2]]
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    if myerr<tol:
        #print(calls_saved,'function calls saved on interval', [a,b])  #<-- use to see final integral intervals
        return np.array([i2, calls_saved])
    else:
        mid=(a+b)/2
        calls_saved+=6  #Each integration 'split' reuses 6 function calls of previous integration
        #Store updated saved call count into next iteration
        int1=integrate_adaptive(fun,a,mid,tol/2,extra=[y[0],y[1],y[2],calls_saved])  
        int2=integrate_adaptive(fun,mid,b,tol/2,extra=[y[2],y[3],y[4],calls_saved])
        return int1+int2
    
fun=np.sin
a=0
b=np.pi/2
prec=8
tol=10**(-prec)
x=np.linspace(a,b,1001)
integral, calls_saved=integrate_adaptive(fun,a,b,tol)
print('Integral of sin(x) from 0 to pi/2:', integral, 'with', calls_saved, 'function calls saved')

def fun(x):
   return 5*x**4
a=0
b=1
prec=8
tol=10**(-prec)
x=np.linspace(a,b,1001)
integral, calls_saved=integrate_adaptive(fun,a,b,tol)
print('Integral of 5x^4 from 0 to 1:', integral, 'with', calls_saved, 'function calls saved')

fun=np.log
a=1
b=np.exp(1)
prec=8
tol=10**(-prec)
x=np.linspace(a,b,1001)
integral, calls_saved=integrate_adaptive(fun,a,b,tol)
print('Integral of ln(x) from 1 to e:', integral, 'with', calls_saved, 'function calls saved')
