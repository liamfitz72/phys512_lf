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
    events_json=json.load(open(data_dir+json_file,'r'))
    data_dict={}
    for event_name in event_names:
        event=events_json[event_name]
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
        data_dict[event_name]=params
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

def interp_Ninv_ft(psd,freqs_psd,freqs):
    psd_interp=scipy.interpolate.interp1d(freqs_psd,psd)
    Ninv_ft=1/psd_interp(freqs)
    return Ninv_ft

def matched_filter(strain,win,template,Ninv_ft):
    data_ft=np.fft.fft(win*strain)
    tp=template[0]
    template_ft=np.fft.fft(win*tp)
    rhs=np.fft.ifft(np.conj(template_ft)*Ninv_ft*data_ft)
    return np.fft.fftshift(rhs)

def signal2noise(matched_filter,template,Ninv_ft,df,win):
    tp=template[0]
    template_ft=np.fft.fft(win*tp)
    sigmasq=(template_ft*Ninv_ft*np.conj(template_ft)).sum()*df
    sigma=np.sqrt(np.abs(sigmasq))
    return np.abs(matched_filter)/sigma

def search_allevents(data_dict,event_names):
    matched_filters={}
    signals2noise={}
    for event_name in event_names:
        params=data_dict[event_name]
        time,dt,strain_Ha,strain_Li,fs,tevent,template=params.values()
        win=scipy.signal.tukey(len(time))
        
        psd_Ha,freqs_psd=mlab.psd(strain_Ha,Fs=fs,NFFT=4*fs,sides='twosided')
        freqs=np.fft.fftfreq(len(time),dt)
        freqs_psd[-1]=freqs.max()
        df=np.abs(freqs[1]-freqs[0])
        Ninv_ft=interp_Ninv_ft(psd_Ha,freqs_psd,freqs)
        matched_filter_Ha=matched_filter(strain_Ha,win,template,Ninv_ft)
        signal2noise_Ha=signal2noise(matched_filter_Ha,template,Ninv_ft,df,win)
        
        psd_Li,freqs_psd=mlab.psd(strain_Li,Fs=fs,NFFT=4*fs,sides='twosided')
        freqs=np.fft.fftfreq(len(time),dt)
        freqs_psd[-1]=freqs.max()
        df=np.abs(freqs[1]-freqs[0])
        Ninv_ft=interp_Ninv_ft(psd_Li,freqs_psd,freqs)
        matched_filter_Li=matched_filter(strain_Li,win,template,Ninv_ft)
        signal2noise_Li=signal2noise(matched_filter_Li,template,Ninv_ft,df,win)
        
        matched_filters[event_name]=matched_filter_Ha,matched_filter_Li
        signals2noise[event_name]=signal2noise_Ha,signal2noise_Li
    return matched_filters,signals2noise

def plot_ASD(params,event_name,save=True):
    time,dt,strain_Ha,strain_Li,fs,tevent,template=params.values()
    psd_Ha,freqs_psd=mlab.psd(strain_Ha,Fs=fs,NFFT=4*fs)
    psd_Li,freqs_psd=mlab.psd(strain_Li,Fs=fs,NFFT=4*fs)
    plt.loglog(freqs_psd,np.sqrt(psd_Ha))
    plt.loglog(freqs_psd,np.sqrt(psd_Li))
    plt.xlabel('Freq. (Hz)')
    plt.xlim([20,2000])
    plt.ylabel('ASD (strain/rt(Hz))')
    plt.ylim([10**(-25),10**(-19)])
    plt.legend(['H1','L1'])
    plt.title('ASD of '+event_name+' Event from both H1 and L1 detectors',fontsize=10)
    plt.grid(which='both',linestyle=':',linewidth=0.6,color='k')
    

        
    

data_dict=read_events(data_dir,event_names)
matched_filters,signals2noise=search_allevents(data_dict,event_names)

event_name='GW150914'
time=data_dict[event_name]['time']
dt=data_dict[event_name]['dt']
N=len(time)
freq=np.fft.fftfreq(N,dt)
strain_Ha=data_dict[event_name]['strain_Ha']
strain_Ha_ft=np.fft.fft(strain_Ha)
ps=np.abs(strain_Ha_ft)**2
plt.plot(freq[:N//2],ps[:N//2])

# sig2noise_Ha=signals2noise[event_name][0]
# matched_filter_Ha=matched_filters[event_name][0]
# plt.plot(time,sig2noise_Ha)
# plt.clf()
# plot_ASD(data_dict[event_name],event_name)

# time=data_dict['GW150914']['time']
# strain_Ha=data_dict['GW150914']['strain_Ha']
# strain_Li=data_dict['GW150914']['strain_Li']
# fs=data_dict['GW150914']['fs']
# dt=data_dict['GW150914']['dt']
# tp,tx=data_dict['GW150914']['template']
# time,dt,strain_Ha,strain_li,fs,tevent,template=data_dict['GW150914'].values()

# psd_Ha,freq=mlab.psd(strain_Ha,Fs=fs,NFFT=4*fs,sides='twosided')

# psd_Ha_smooth=smooth(psd_Ha,10)

# freqs=np.fft.fftfreq(len(time),dt)
# df=np.abs(freqs[1]-freqs[0])
# freq[-1]=freqs.max()


# win=scipy.signal.tukey(len(strain_Ha))


# psd_interp=scipy.interpolate.interp1d(freq,psd_Ha_smooth)
# # freqs=freqs[:len(freqs)//2]
# Ninv_ft=1/psd_interp(freqs)

# data_ft=np.fft.fft(strain_Ha*win)
# template_ft=np.fft.fft(tp*win)
# rhs=np.fft.ifft(Ninv_ft*data_ft*np.conj(template_ft))
# rhs=np.fft.fftshift(rhs)

# # plt.vlines(data['GW150914']['tevent'],-2*10**8,2*10**8,color='red',zorder=10)

# signal2noise_Ha=signal2noise(rhs,template,Ninv_ft,df,win)
# plt.plot(time,rhs,zorder=-10)


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
