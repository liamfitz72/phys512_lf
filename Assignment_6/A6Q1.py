# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:29:17 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt

def convolve(f,g):
    f_ft=np.fft.fft(f)
    g_ft=np.fft.fft(g)
    conv=np.fft.ifft(f_ft*g_ft)
    return np.fft.fftshift(conv)

def conv_shift(x,n):
    size=len(x)
    delta_shift=np.zeros(size)
    delta_shift[int(size/2)+n]=1
    return convolve(x,delta_shift)

def gaussian(x):
    return np.exp(-x**2/2)
    

if __name__=="__main__":
    N=2000
    x=np.linspace(-5,5,N+1)
    y=gaussian(x)
    y_shift=conv_shift(y,N//2)
    plt.plot(x,y)
    plt.plot(x,y_shift)
    plt.legend(['Gaussian','Gaussian with half \narray length shift'])

