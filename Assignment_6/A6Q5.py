# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:53:50 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib as mpl
import scipy
import json
import h5py
import sys
from A6Q2 import correlation
plt.rcParams['figure.dpi'] = 200
mpl.rcParams['lines.linewidth'] = 0.75
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5


data_dir="C:/Users/liamf/phys512_lf/Assignment_6/Data/"
sys.path.append(data_dir)
import readligo
event_names=['GW150914','GW151226','LVT151012','GW170104']

def read_template(data_dir,filename):
    dataFile=h5py.File(data_dir+filename,'r')
    template=dataFile['template']
    tp=template[0]
    tx=template[1]
    return tp,tx

def read_events(data_dir,event_names):
    json_file="BBH_events_v3.json"
    events=json.load(open(data_dir+json_file,'r'))
    data_dict={}
    for eventname in event_names:
        event=events[eventname]
        fn_H1=event['fn_H1']
        fn_L1=event['fn_L1']
        # fband = event['fband']              
        strain_Ha,time_Ha,chan_dict_Ha=readligo.loaddata(data_dir+fn_H1,'H1')
        strain_Li,time_Li,chan_dict_Li=readligo.loaddata(data_dir+fn_L1,'L1')
        time=time_Ha
        template=read_template(data_dir,event['fn_template'])
        
        params={}
        params['time']=time_Ha
        params['dt']=time[1]-time[0]
        params['strain_Ha']=strain_Ha
        params['strain_Li']=strain_Li
        params['fs']=event['fs']
        params['tevent']=event['tevent']
        params['template']=template
        data_dict[eventname]=params
    return data_dict

def smooth(array,smooth_factor):
    n=len(array)
    x=np.arange(n)
    x[n//2:]=x[n//2:]-n
    sig=smooth_factor
    krnl=np.exp(-0.5*(x/sig)**2)
    krnl=krnl/krnl.sum()
    array_ft=np.fft.fft(array)
    krnl_ft=np.fft.fft(krnl)
    array_smooth=np.fft.ifft(array_ft*krnl_ft)
    return array_smooth

def noise_model(psd):
    psd_smooth=smooth(smooth_spectlines(psd,10),10)
    N_inv=1/psd_smooth
    return N_inv,psd_smooth

data=read_events(data_dir,event_names)
time=data['GW150914']['time']
strain_Ha=data['GW150914']['strain_Ha']
strain_Li=data['GW150914']['strain_Li']
fs=data['GW150914']['fs']
dt=data['GW150914']['dt']
tp,tx=data['GW150914']['template']

psd_Ha,freq=mlab.psd(strain_Ha,Fs=fs,NFFT=4*fs)
psd_Ha_smooth=smooth(psd_Ha,20)

freqs=np.fft.fftfreq(len(time),dt)
freqs=freqs[:int(len(freqs)/2)]
psd_interp=scipy.interpolate.interp1d(freq,psd_Ha_smooth)
Ninv_ft=1/psd_interp(freqs)

win=scipy.signal.tukey(len(strain_Ha))
template_ft=np.fft.rfft(tp*win)

data_ft=np.fft.rfft(strain_Ha*win)
rhs=np.fft.irfft(Ninv_ft*data_ft[:-1]*np.conj(template_ft)[:-1])
rhs=np.fft.fftshift(rhs)

print(len(rhs))
print(len(time))
plt.plot(time[:-2],rhs)
plt.vlines(data['GW150914']['tevent'],-2*10**8,2*10**8,color='red',zorder=10)


# plt.clf()
# plt.loglog(freq,np.sqrt(psd_Ha))
# plt.loglog(freq,np.sqrt(psd_Ha_smooth))
# # psd_Li,freq=mlab.psd(strain_Li,Fs=fs,NFFT=4*fs)
# # plt.loglog(freq,np.sqrt(psd_Li))
# # plt.xlim([20,2000])
# plt.xlabel('Freq. (Hz)')
# plt.xlim([20,2000])
# plt.ylabel('ASD (strain/rt(Hz))')
# plt.ylim([10**(-25),10**(-19)])
# plt.legend(['Hartford detector ASD','Smoothed ASD'])
# plt.grid()


# plt.plot(time,strain_Ha)

# freq=fftfreq(len(time),dt)
# strain_Ha_FT=fft(strain_Ha)
# plt.plot(freq,strain_Ha_FT)
