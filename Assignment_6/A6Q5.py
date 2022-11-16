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
from A6Q1 import convolve,gaussian
from A6Q2 import correlation
plt.rcParams['figure.dpi'] = 200


data_dir="C:/Users/liamf/phys512_lf/Assignment_6/Data/"
sys.path.append(data_dir)
import readligo


def get_powerspect(time,strain,win,smooth_width):
    dt=time[1]-time[0]
    freqs=np.fft.fftfreq(len(time),dt)
    ps=np.abs(np.fft.fft(strain*win))**2
    freqs=np.fft.fftshift(freqs)  # FFT shift to convolve and plot properly
    ps=np.fft.fftshift(ps)
    smooth_fun=gaussian(freqs,0,smooth_width)
    ps_smooth=convolve(ps,smooth_fun/smooth_fun.sum())  # Normalize smooth_fun
    return freqs,ps,ps_smooth

def matched_filter(strain,win,template,noise_model):
    data_ft=np.fft.fft(win*strain)
    template_ft=np.fft.fft(win*template)
    Ninv_ft=1/noise_model
    rhs=np.fft.ifft(np.conj(template_ft)*Ninv_ft*data_ft)
    return np.fft.fftshift(rhs)  # FFT shift after inverse

def signal2noise(matched_filter,win,template,noise_model):
    noise=np.sqrt(np.mean(matched_filter**2))  # Estimate noise as RMS of matched filter
    return np.abs(matched_filter)/noise

def read_template(data_dir,filename):
    dataFile=h5py.File(data_dir+filename,'r')
    template=dataFile['template']
    tp=template[0]
    tx=template[1]
    return tp,tx

def search_allevents(json_fname,event_names):
    events=json.load(open(data_dir+json_fname,'r'))
    times={}
    ps={}
    ps_smooth={}
    matched_filters={}
    signals2noise={}
    templates={}
    for event_name in event_names:
        event=events[event_name]   # First we have to read from JSON file
        fn_H1=event['fn_H1']
        fn_L1=event['fn_L1']
        tevent=event['tevent']
        strain_Ha,time_Ha,chan_dict_Ha=readligo.loaddata(data_dir+fn_H1,'H1')
        strain_Li,time_Li,chan_dict_Li=readligo.loaddata(data_dir+fn_L1,'L1')
        time=time_Ha-tevent
        template=read_template(data_dir,event['fn_template'])[0]
        win=scipy.signal.tukey(len(time),0.1)
        
        # Get noise model, then matched filter, then signal to noise
        freqs,ps_Ha,ps_smooth_Ha=get_powerspect(time,strain_Ha,win,2)
        noise_model_Ha=np.fft.fftshift(ps_smooth_Ha)
        matched_filter_Ha=matched_filter(strain_Ha,win,template,noise_model_Ha)
        df=np.abs(freqs[1]-freqs[0])
        signal2noise_Ha=signal2noise(matched_filter_Ha,win,template,noise_model_Ha)
        
        freqs,ps_Li,ps_smooth_Li=get_powerspect(time,strain_Li,win,2)
        noise_model_Li=np.fft.fftshift(ps_smooth_Li)
        matched_filter_Li=matched_filter(strain_Li,win,template,noise_model_Li)
        signal2noise_Li=signal2noise(matched_filter_Li,win,template,noise_model_Li)

        times[event_name]=time
        ps[event_name]=ps_Ha,ps_Li
        ps_smooth[event_name]=ps_smooth_Ha,ps_smooth_Li
        matched_filters[event_name]=matched_filter_Ha,matched_filter_Li
        signals2noise[event_name]=signal2noise_Ha,signal2noise_Li
        templates[event_name]=template
    return times,freqs,ps,ps_smooth,matched_filters,signals2noise,templates

def sig2noise_analytic(matched_filter,template,noise_model):
    Ninv_ft=1/noise_model
    Ninv=np.fft.fftshift(np.fft.ifft(Ninv_ft))
    noise=template.T@(Ninv*template)
    return np.abs(matched_filter/noise)

json_fname="BBH_events_v3.json"
event_names=['GW150914','GW151226','LVT151012','GW170104']
times,freqs,ps,ps_smooth,matched_filters,signals2noise,templates=search_allevents(json_fname,event_names)

event_name='GW150914'
plt.loglog(freqs,ps[event_name][0])
plt.loglog(freqs,ps_smooth[event_name][0])
plt.xlim([20,2000])
plt.xlabel('Frequency (Hz)')
plt.ylim([10**(-43),10**(-28)])
plt.legend(['Hartsford detector power spectrum','Smoothed power spectrum'])
plt.savefig('A6Q5_plot1.png',bbox_inches='tight')
plt.clf()

plt.loglog(freqs,ps[event_name][1])
plt.loglog(freqs,ps_smooth[event_name][1])
plt.xlim([20,2000])
plt.xlabel('Frequency (Hz)')
plt.ylim([10**(-43),10**(-28)])
plt.legend(['Livingston detector power spectrum','Smoothed power spectrum'])
plt.savefig('A6Q5_plot2.png',bbox_inches='tight')
plt.clf()

plt.plot(times[event_name],matched_filters[event_name][0])
plt.xlabel('Time since event (s)')
plt.savefig('A6Q5_plot3.png',bbox_inches='tight')
plt.clf()

plt.plot(times[event_name],signals2noise[event_name][0])
plt.xlabel('Time since event (s)')
plt.savefig('A6Q5_plot4.png',bbox_inches='tight')
plt.clf()

for event_name in event_names:
    plt.plot(times[event_name],signals2noise[event_name][0])
    plt.plot(times[event_name],signals2noise[event_name][1])
    plt.xlim([-1,1])
    plt.legend(['Hanford detector','Livingston detector'])
    plt.xlabel('Time since event (s)')
    plt.title('Event '+event_name)
    plt.savefig('A6Q5_mf_'+event_name+'.png',bbox_inches='tight')
    plt.clf()


# s2na=sig2noise_analytic(matched_filters[event_name][0],templates[event_name],ps_smooth[event_name][0])
# plt.plot(times[event_name],s2na)

dts=[]
for event_name in event_names:
    time=times[event_name]
    i=np.argmax(signals2noise[event_name][0])
    j=np.argmax(signals2noise[event_name][1])
    dt=np.abs(time[i]-time[j])
    dts.append(dt)
dts=np.array(dts)
    
print(dts)
print(dts*3*10**8/1000)








# plt.plot(time-tevent,template)
# plt.xlim([-3,3])
# N=len(time)

# time_new=(time-tevent)
# i=np.argmin(np.abs(time_new))
# time_new=time_new[i-10000:i+10000]
# strain_Ha=strain_Ha[i-10000:i+10000]

# plt.plot(time_new,strain_Ha)
# win=scipy.signal.tukey(len(time_new),0.1)
# plt.plot(time_new,strain_Ha*win)
# plt.clf()

# freq=np.fft.fftfreq(len(time_new),dt)
# ps=np.abs(np.fft.fft(strain_Ha*win))**2
# freq=np.fft.fftshift(freq)
# ps=np.fft.fftshift(ps)

# plt.loglog(freq,ps)
# plt.xlim([20,2000])

# M=10
# smooth_fun=gaussian(freq,0,2)
# ps_smooth=convolve(ps,smooth_fun/smooth_fun.sum())
# plt.loglog(freq,ps_smooth)

# plt.clf()
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
