# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:26:07 2022

@author: liamf
"""

import numpy as np
from scipy import interpolate as interp
from matplotlib import pyplot as plt
import random

data=np.loadtxt('lakeshore.txt')

def lakeshore(V, data):
    t=data[:,0] #Temperature
    v=data[:,1] #Voltage

    spl=interp.splrep(t,v)
    t_fine=np.linspace(t[0],t[-1],1001)  #Creating spline
    V_fine=interp.splev(t_fine, spl)
    t_interp=list()
    t_err=list()
    for p in range(len(V_fine)): #Loop through V's individually to avoid comparing arrays
        i=np.abs(V_fine-V[p]).argmin() #Finding index of spline value closest to V
        t_interp.append(t_fine[i])  #Interpolated T given V
    
        #Calculating error
        t_diff=list()  #Initialize list of T differences
        for k in range(0,25): #25 samples to calculate standard deviation
            n=np.size(t)
        
            r1=-1  #Initial values so while loop runs
            r2=0
            while V[p]>v[r1] or V[p]<v[r2]:  #Ensure V is in random interval
                r1=random.randint(0,(n-1)-4)  #Leave Room for at least 4 points
                r2=random.randint(r1+4,(n-1)) #Start so that interval has at least 4 points
            
            sub_spl=interp.splrep(t[r1:r2],v[r1:r2]) #Create sub spline
            sub_t_fine=np.linspace(t[r1],t[r2],1001)
            sub_V_fine=interp.splev(sub_t_fine, sub_spl)
        
            j=(np.abs(sub_V_fine-V[p])).argmin() #Find index of sub spline value closest to V
        
            t_diff.append(np.abs(sub_t_fine[j]-t_fine[i]))
        t_err.append(np.std(t_diff))
    return t_interp, t_err

v_fine=np.linspace(data[0,1], data[-1,1], 1001)
plt.plot(v_fine, lakeshore(v_fine, data)[1])
plt.xlabel('Voltage')
plt.ylabel('Temperature error')
plt.title('Error in interpolated temperature as a function of voltage')
plt.savefig('A1Q3_plot.png')