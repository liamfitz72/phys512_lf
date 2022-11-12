# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:21:04 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from A6Q1 import convolve,conv_shift,gaussian

def padded_conv(f,g,npad):
    pad=np.zeros(npad)
    fpad=np.append(pad,f)
    fpad=np.append(fpad,pad)
    gpad=np.append(pad,g)
    gpad=np.append(gpad,pad)
    conv_pad=convolve(fpad,gpad)
    return conv_pad[npad:-npad]

if __name__=="__main__":
    x=np.linspace(-5,5,1001)
    y=gaussian(x)
    yshift=conv_shift(y,200)
    window=np.cos(np.pi*x/(x[-1]-x[0]))**2
    conv=convolve(y,yshift)
    conv_pad=padded_conv(y,yshift,200)
    plt.plot(x,conv)
    plt.plot(x,conv_pad)
    plt.legend(['FFT convolution of \nshifted Gaussian',
                'Same FFT convolution with \nzero padded arrays'])
