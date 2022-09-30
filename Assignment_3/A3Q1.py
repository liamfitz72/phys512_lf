# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:01:19 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt

#RK4 step function as shown in slides
def rk4_step(fun,x,y,h):            #Start count of fun calls:
    k1=h*fun(x,y)                   #+1
    k2=h*fun(x+h/2,y+k1/2)          #+1
    k3=h*fun(x+h/2,y+k2/2)          #+1
    k4=h*fun(x+h,y+k3)              #+1
    return y+(k1+2*k2+2*k3+k4)/6    #Total 4 fun calls

def fun(x,y):   #RHS of ODE
    return y/(1+x**2)

def sol(x, C0):     #General solution to ODE
    return C0*np.exp(np.arctan(x))

nsteps=200
npt=nsteps+1
x=np.linspace(-20,20,npt)
y=np.zeros(npt)
y[0]=1  #Initial condition
for i in range(nsteps):    #Iterate stepper over each interval
    h=x[i+1]-x[i]
    y[i+1]=rk4_step(fun,x[i],y[i],h)

#Plotting error vs true solution
true=sol(x,1/sol(x[0],1))
err=np.abs(true-y)
plt.plot(x,err)  
plt.xlabel('x')
plt.ylabel('Error')
plt.legend(['RK4 error'])
plt.savefig('A3Q1_plot1.png',bbox_inches='tight',dpi=200)

def rk4_stepd(fun,x,y,h):           #Start count of fun calls:
    y1=rk4_step(fun,x,y,h)          #+4
    y2=rk4_step(fun,x,y,h/2)        #+4
    y2=rk4_step(fun,x+h/2,y2,h/2)   #+4
    return (16*y2-y1)/15            #Total 12 fun calls
                                    # => use 1/3 the intervals for same call count
plt.clf()                       
y[0]=1
#Solving ODE using improved stepper
for i in range(nsteps):
    h=x[i+1]-x[i]
    y[i+1]=rk4_stepd(fun,x[i],y[i],h)
#Plotting error of new method
err=np.abs(true-y)
plt.plot(x,err)
plt.xlabel('x')
plt.ylabel('Error')
plt.legend(['New RK4 method error'])
plt.savefig('A3Q1_plot2.png',bbox_inches='tight',dpi=200)


plt.clf()
#Solving ODE using old and new stepper, using same # of function calls counts over inteverval
nsteps1=600
npt1=nsteps1+1
x1=np.linspace(-20,20,npt1)
y1=np.zeros(npt1)
y1[0]=1
for i in range(nsteps1):
    h=x1[i+1]-x1[i]
    y1[i+1]=rk4_step(fun,x1[i],y1[i],h)
nsteps2=int(nsteps1/3)
npt2=nsteps2+1
x2=np.linspace(-20,20,npt2)
y2=np.zeros(npt2)
y2[0]=1
for j in range(nsteps2):
    h=x2[j+1]-x2[j]
    y2[j+1]=rk4_stepd(fun,x2[j],y2[j],h)
    
#Plotting RK4 vs improved RK4 error
true1=sol(x1,1/sol(x1[0],1))
err1=np.abs(true1-y1)
true2=sol(x2,1/sol(x2[0],1))
err2=np.abs(true2-y2)
plt.plot(x1,err1)
plt.plot(x2,err2)
plt.legend(['RK4 Error ('+str(nsteps1)+' steps)','Improved RK4 Error ('+str(nsteps2)+' steps)'])
plt.xlabel('x')
plt.ylabel('y(x)')
plt.savefig('A3Q1_plot3.png',bbox_inches='tight',dpi=200)
print('Improved RK4 is ~',int(np.max(err1)/np.max(err2)),
      'times more accurate than the original method on the interval', [x[0],x[-1]])





