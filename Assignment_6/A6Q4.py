# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:00:50 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

def dft_sum(k,k0,N):
    a=1-np.exp(-2J*np.pi*(k-k0))
    b=1-np.exp(-2J*np.pi*(k-k0)/N)
    return a/b

k0=0.1
N=1000
x=np.linspace(0,N,N)
y=np.exp(2J*np.pi*k0*x)
ft=np.fft.fft(y)
k=np.fft.fftfreq(len(y),np.abs(x[1]-x[0]))
# plt.plot(k,np.abs(ft))
# plt.plot(k,np.abs(dft_sum(k,k0,N)))
# resid=np.abs(ft)-np.abs(dft_sum(k,k0,N))
# plt.clf()
# plt.plot(k,resid)
# plt.clf()

# D)

def window_fun(x,N):
    return 0.5-0.5*np.cos(2*np.pi*x/N)

ft_wndw=np.fft.fft(y*window_fun(x,N))
norm=np.sqrt(np.mean(window_fun(x,N)**2))
plt.plot(k,np.abs(ft))
plt.plot(k,np.abs(ft_wndw/norm))
plt.clf()

print(np.abs(np.fft.fft(window_fun(x,N)).real))

ft_wndw_new=np.empty(N)
for i in range(N):
    if i==N-1:
        ft_wndw_new[i]=-ft[i-1]/4+ft[i]/2+-ft[0]/4
    else:
        ft_wndw_new[i]=-ft[i-1]/4+ft[i]/2+-ft[i+1]/4


plt.plot(k,np.abs(ft))
norm_new=np.sqrt(np.mean(ft_wndw_new**2))
plt.plot(k,np.abs(ft_wndw_new/norm))
# plt.clf()
# resid=np.abs(ft)-np.abs(ft_wndw_new/norm)
# plt.plot(k,resid)