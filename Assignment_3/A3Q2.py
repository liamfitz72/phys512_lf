# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:18:34 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#Defining time units and half lives


year=1
day=year/365
hour=day/24
minute=hour/60
second=minute/60
microsecond=10**(-6)*second

half_lives=[4.468*10**9*year,
            24.10*day,
            6.70*hour,
            245500*year,
            75380*year,
            1600*year,
            3.8235*day,
            3.10*minute,
            26.8*minute,
            19.9*minute,
            164.3*microsecond,
            22.3*year,
            5.015*year,
            138.376*day]

elements=['U-238',
          'Th-234',
          'Pa-234',
          'U-234',
          'Th-230',
          'Ra-226',
          'Rn-222',
          'Po-218',
          'Pb-214',
          'Bi-214',
          'Po-214',
          'Pb-210',
          'Bi-210',
          'Po-210',
          'Pb-206']

decay_rates=np.log(2)/half_lives
n=len(decay_rates)+1

#Generating decy chain functions on RHS of equations
def fun(t,y):
    dydt=np.zeros(n)
    dydt[0]=-decay_rates[0]*y[0]  #First isotope is only decaying
    for i in range(1,n-1):
        #Middle isotopes decay and are produced by previous isotopes
        dydt[i]=-decay_rates[i]*y[i]+decay_rates[i-1]*y[i-1]
    dydt[n-1]=decay_rates[n-2]*y[n-2]  #Last isotope is only being produced
    return dydt

#loglog plotting each isotope over time span of 1 U-238 half-life
t_0=year    #For better plot, to avoid [log(0),log(0)]
t_f=half_lives[0]
y0=[1]+[0]*(n-1)  #Initial conditions vector
sol=integrate.solve_ivp(fun,(year,t_f),y0,method='Radau', max_step=t_f/20)  #Radau implicit method for stiff equation
legend=[]
for i in range(len(sol['y'])):  #Plot and label each isotope
    plt.plot(sol['t'],sol['y'][i])
    legend.append(elements[i])
plt.xscale('log')
plt.yscale('log')
plt.legend(legend, ncol=3)
plt.xlabel(r'$t$ (years)')
plt.ylabel(r'$N_i(t)\,/\,N_{U_{238}}(t=0)$')
plt.savefig('A3Q2_plot1.png',bbox_inches='tight',dpi=200)


#Evaluating and plotting ratio of Pb-206 to U-238 over long time frame
plt.clf()
t_f=2*half_lives[0]
sol=integrate.solve_ivp(fun,(t_0,t_f),y0,method='Radau', max_step=t_f/20)
t=sol['t']
true=np.exp(decay_rates[0]*t)-1
plt.loglog(t[2:], (true-sol['y'][-1]/sol['y'][0])[2:])
plt.xlabel('t (years)')
plt.legend(['Error in ratio Pb-206 to U-238 over 2 half lives of U-238'])
plt.savefig('A3Q2_plot2.png',bbox_inches='tight',dpi=200)


#Evauluating and plotting ratio of Th-230 to U-234 over 1 million years
plt.clf()
t_f=1*10**6*year
sol=integrate.solve_ivp(fun,(0,t_f),y0,method='Radau', max_step=t_f/20)
plt.plot(sol['t'],sol['y'][4]/sol['y'][3])
plt.xlabel('t (days)')
plt.legend(['Ratio Th-230 to U-234 over 1 million years'])
plt.savefig('A3Q2_plot3.png',bbox_inches='tight',dpi=200)



    



