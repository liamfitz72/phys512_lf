# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:00:50 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

def dft_sum(k,k0,N):
    a=1-np.exp(-2J*np.pi*N*(k-k0))
    b=1-np.exp(-2J*np.pi*(k-k0))
    return a/b

k0=0.1
N=1001
x=np.linspace(0,N-1,N)
y=np.exp(2J*np.pi*k0*x)
k=np.fft.fftfreq(N,np.abs(x[1]-x[0]))
ft=np.abs(np.fft.fft(y))
dft=np.abs(dft_sum(k,k0,N))
dftshift=np.fft.fftshift(dft)
kshift=np.fft.fftshift(k)
ftshift=np.fft.fftshift(ft)

plt.plot(kshift,ftshift) # fftshift to plot negative freqs
plt.plot(kshift,dftshift)
plt.legend(['FFT','DFT analytic sum'])
plt.savefig('A6Q4_plot1.png',bbox_inches='tight')
plt.clf()
resid=np.abs(dft_sum(k,k0,N))-np.abs(ft)
plt.plot(kshift,np.fft.fftshift(resid))
plt.legend(['Residuals from FFT \nand DFT sum'])
plt.savefig('A6Q4_plot2.png',bbox_inches='tight')
plt.clf()

# D)

def window_fun(x,N):
    return 0.5-0.5*np.cos(2*np.pi*x/N)

ft_wndw=np.abs(np.fft.fft(y*window_fun(x,N)))
norm=np.sqrt(np.mean(window_fun(x,N)**2))
ft_wndw=ft_wndw/norm
ft_wndw_shift=np.fft.fftshift(ft_wndw)
plt.plot(kshift,ftshift)
plt.plot(kshift,ft_wndw_shift)
plt.xlim([0,0.2])
plt.legend(['FFT of sine wave','Windowed FFT'])
plt.savefig('A6Q4_plot3.png',bbox_inches='tight')
plt.clf()

print(np.fft.fft(window_fun(x,N)).real)

ft_wndw_new=np.empty(N)
for i in range(N):
    if i==N-1:  # For wrap around, use special case for last array element 
        ft_wndw_new[i]=-ft[i-1]/4+ft[i]/2+-ft[0]/4
    else:
        ft_wndw_new[i]=-ft[i-1]/4+ft[i]/2+-ft[i+1]/4

plt.plot(kshift,ftshift)
norm_new=np.sqrt(np.mean(ft_wndw_new**2))
ft_wndw_new=np.abs(ft_wndw_new)/norm
plt.plot(kshift,np.fft.fftshift(ft_wndw_new))
plt.xlim([0,0.2])
plt.legend(['FFT of sine wave','Windowed FFT \n (new calculation)'])
plt.savefig('A6Q4_plot4.png',bbox_inches='tight')


