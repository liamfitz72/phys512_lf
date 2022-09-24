# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:40:09 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt

#Fits Cheb. poly. with degree=deg given #points=pts over x_range
def log2_chebfit(x_range,deg,pts):
    x_pts=np.linspace(x_range[0],x_range[-1],pts)
    
    #Need to fit x in [-1,1]
    x_range=x_range*2/(x_range[-1]-x_range[0]) #Scaling interval length to 2
    x_range=x_range-x_range[0]-1  #Shifting interval to start at -1
    x_pts_new=np.linspace(x_range[0],x_range[-1],pts)
    
    log2_pts=np.log2(x_pts)
    cheb_fit=np.polynomial.chebyshev.chebfit(x_pts_new,log2_pts,deg)
    return cheb_fit

#Evaulates cheb. poly. fit over x_range at a given x (not necessarily part of x_range)
def log2_chebval(x,x_range,deg,pts):
    cheb_fit=log2_chebfit(x_range,deg,pts)
    
    #Rescaling and shifting
    x_new=x*2/(x_range[-1]-x_range[0])
    x_range_new=x_range*2/(x_range[-1]-x_range[0])
    x_new=x_new-x_range_new[0]-1
    
    ans=np.polynomial.chebyshev.chebval(x_new,cheb_fit)
    err=np.abs(ans-np.log2(x))
    return ans, err, np.max(err)

def mylog2(x):
    deg=7
    pts=10
    x_range=np.linspace(0.5,1,1001)
    
    mant_x,exp_x=np.frexp(x)  #Evaluate log2(x) after splitting into mantissa + exp.
    log2_x=log2_chebval(mant_x,x_range,deg,pts)[0]+exp_x
    
    mant_e,exp_e=np.frexp(np.exp(1)) #Evaluate log2(e) ''                        ''
    log2_e=log2_chebval(mant_e,x_range,deg,pts)[0]+exp_e
    return log2_x/log2_e


#log_2(x) from [0.5,1] fit and graph
a=0.5
b=1
deg=7
pts=10
x=np.linspace(a,b,1001)
y,err,max_err=log2_chebval(x,x,deg,pts)
x_pts=np.linspace(a,b,pts)
y_pts=np.log2(x_pts)
plt.plot(x,y,c='orange')
plt.scatter(x_pts,y_pts,zorder=10)
plt.xlabel(r'$x$')
plt.ylabel(r'$log_2(x)$')
plt.legend(['Sampled points: '+str(pts),'Chebyshev polnomial fit deg. '+str(deg)+'\nw/ max error: '+str(max_err)])
plt.savefig('A2Q3_plot1.png')

#Evaluate mylog2(x) and graph against np.log(x)
plt.clf()
a=0.5
b=10
deg=7
pts=10
x=np.linspace(a,b,1001)
y=mylog2(x)
plt.plot(x,y)
plt.plot(x,np.log(x))
plt.legend(['mylog2(x)','numpy.log(x)'])
plt.savefig('A2Q3_plot2.png')

#Error from previous graph
plt.clf()
a=-3
b=3
deg=7
pts=10
x=10**np.linspace(a,b,1001)
y=mylog2(x)-np.log(x)
plt.plot(x,y)
plt.xscale('log')
plt.legend(['Difference between mylog2(x) and numpy.log(x)'])
plt.savefig('A2Q3_plot3.png')
