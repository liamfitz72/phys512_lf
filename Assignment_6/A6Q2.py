# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:44:23 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from A6Q1 import conv_shift,gaussian
plt.rcParams['figure.dpi'] = 200

def correlation(f,g):
    f_ft=np.fft.fft(f)
    g_ft=np.fft.fft(g)
    corr=np.fft.ifft(f_ft*np.conj(g_ft))
    return np.fft.fftshift(corr)

if __name__=="__main__":
    x=np.linspace(-5,5,1001)
    y=gaussian(x)
    corr=correlation(y,y)
    plt.plot(x,corr)
    plt.legend(['Cross correlation of \nGaussian with itself'])
    plt.clf()
    
    y_shift=conv_shift(y,100)
    shift_corr=correlation(y,y_shift)
    plt.plot(x,corr)
    plt.plot(x,shift_corr)
    plt.legend(['Cross correlation from a)','Shifted Gaussian \ncorrelation'])

