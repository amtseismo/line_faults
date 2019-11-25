#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:15:52 2019

Non-dimensionalized line-fault solving antiplane rate and state equations

@author: amt
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def setopts():
    o={}
    o['G']=30e9 # rigidity (Pa)
    o['nu']=0.25  # Poisson Ratio
    o['vpl']=1e-8 # plate velocity (m/s)
    o['a']=0.008 # frictional parameter a
    o['b']=0.01 # frictional parameter b
    o['dc']=1e-5 # weakening distance (m)
    o['vs']=3e3 # shear wave velocity (m/s)
    o['sl']=1 # slip law or ageing law
    o['W']=600 # asperity size (m)    
    o['WpoW']=5 # this is W'/W (must be odd)
    o['gridfac']=10 # this is what you add to either side to make the FFT from -inf to inf
    o['gridlength']=(2*o['gridfac']+o['WpoW'])*o['W'] # simulation space (m) 
    o['sigma']=1000000; # effective normal stress in Pa
    o['Lb']=o['G']*o['dc']/(o['sigma']*o['b']); # L_b
    o['hstar']=o['b']/(o['b']-o['a'])*o['Lb'] # L_(b-a)
    o['Linf']=1/np.pi*(o['b']/(o['b']-o['a']))**2*o['Lb']
    o['eta']=o['G']*o['vpl']/(2*o['b']*o['sigma']*o['vs']) # radiation damping (Pa-s/m)  
    if o['sl']==1:
        o['dx']=o['Lb']/20; # cell size (m) for slip law
    else:
        o['dx']=o['Lb']/10; # cell size (m) for ageing law
    if np.mod(o['W']/o['dx'],1)!=0:
        tmp=np.ceil(o['W']/o['dx']);
        o['dx']=o['W']/tmp;
    o['WN']=int(o['W']/o['dx']);
    o['N']=int(o['gridlength']/o['dx']); # number of cells         
    o['X']=np.arange(-(o['gridfac']+np.floor(o['WpoW']/2))*o['W'], ((o['gridfac']+(o['WpoW']-np.floor(o['WpoW']/2)))*o['W']), o['dx']).T    
    o['Wstart']=np.where(o['X']==0)[0]  # index of start of fault position (i.e. 0)
    if len(o['Wstart'])==0:
        o['Wstart']=int(np.where(abs(o['X'])==np.min(abs(o['X'])))) # this deals with irrational o.dx
    else:
        o['Wstart']=int(o['Wstart'][0])
    o['Wend']=o['Wstart']+int(o['W']/o['dx'])-1 # end of fault position (i.e.0) 
    o['Fstart']=int(o['Wstart']-np.floor(o['WpoW']/2)*o['WN'])
    o['Fend']=int(o['Wend']+np.floor(o['WpoW']/2)*o['WN'])   
    o['k']=abs(2*np.pi*np.arange(-o['N']/2,o['N']/2))/((o['gridlength']-o['dx'])/o['Lb']) # wavenumber vector
    o['aoverb']=o['b']/o['a']*np.ones((o['N'])) 
    o['aoverb'][o['Wstart']:o['Wend']] = o['a']/o['b']
    o['vdyn']=2*o['sigma']*o['a']*o['vs']/o['G']
    return o

def getstress(v,o):
    fftx = np.fft.fftshift(np.fft.fft(v))
    df = np.fft.ifft(np.fft.ifftshift(o['k']*fftx))
    return df

# ODEs to solve
def ratestate_sl(t,y,o):
    """
    Defines the differential equations for the coupled spring-mass system.
    Arguments:
        t :  time
        y :  vector of the state variables [theta,velocity,slip,]
        o :  structure with parameters
    """
    State=y[:o['N']]
    Vel=y[o['N']:2*o['N']] 
    Disp=y[2*o['N']:3*o['N']]
    # Slip law
    dStatedt =  np.concatenate((np.zeros(o['Fstart']),
        -State[o['Fstart']:o['Fend']]*Vel[o['Fstart']:o['Fend']]*np.log(State[o['Fstart']:o['Fend']]*Vel[o['Fstart']:o['Fend']]),
        np.zeros(o['N']-o['Fend'])))
    stress=getstress(Vel,o);
    dVeldt = np.concatenate((np.zeros(o['Fstart']),
        ((o['aoverb'][o['Fstart']:o['Fend']])/Vel[o['Fstart']:o['Fend']] + o['eta'])**(-1)*
        (-dStatedt[o['Fstart']:o['Fend']]/State[o['Fstart']:o['Fend']] - stress[o['Fstart']:o['Fend']]/2),
        np.zeros(o['N']-o['Fend'])))
    dDispdt = Vel
    dy = np.concatenate((dStatedt,dVeldt,dDispdt))
    print(t)
    return dy

def ratestate_al(t,y,o):
    """
    Defines the differential equations for the coupled spring-mass system.
    Arguments:
        t :  time
        y :  vector of the state variables [theta,velocity,slip,]
        o :  structure with parameters
    """
    State=y[:o['N']]
    Vel=y[o['N']:2*o['N']] 
    Disp=y[2*o['N']:3*o['N']]
    # Ageing law
    dStatedt = np.concatenate((np.zeros(o['Fstart']),
        1 - State[o['Fstart']:o['Fend']]*Vel[o['Fstart']:o['Fend']],
        np.zeros(o['N']-o['Fend'])))
    stress=getstress(Vel,o);
    dVeldt = np.concatenate((np.zeros(o['Fstart']),
        ((o['aoverb'][o['Fstart']:o['Fend']])/Vel[o['Fstart']:o['Fend']] + o['eta'])**(-1)*
        (-dStatedt[o['Fstart']:o['Fend']]/State[o['Fstart']:o['Fend']] - stress[o['Fstart']:o['Fend']]/2),
        np.zeros(o['N']-o['Fend'])))
    dDispdt = Vel
    dy = np.concatenate((dStatedt,dVeldt,dDispdt))
    print(t)
    return dy

## Set parameters
o=setopts()

## Initial conditions
plots=1;
seed=2;
State = np.concatenate((np.ones(o['Wstart']), 2*np.ones(o['Wend']-o['Wstart']), np.ones(o['N']-o['Wend']))) # initial state 
Vel = np.concatenate((np.ones(o['Wstart']), 1/seed*np.ones(o['Wend']-o['Wstart']), np.ones(o['N']-o['Wend']))) # initial velocity 
Disp = np.zeros(np.shape(Vel)) # initial slip (m)
y0=np.concatenate((State,Vel,Disp))

plt.figure()
plt.subplot(3,1,1)
plt.plot(o['X']/o['W'],State)
plt.subplot(3,1,2)
plt.plot(o['X']/o['W'],Vel)
plt.subplot(3,1,3)
plt.plot(o['X']/o['W'],Disp)

# Call the ODE solver.
if o['sl']:
    wsol = solve_ivp(lambda t, y: ratestate_sl(t, y, o), [0,200], y0, max_step=100, rtol=10e-10, atol=10e-10)
else:
    wsol = solve_ivp(lambda t, y: ratestate_al(t, y, o), [0,200], y0, max_step=100, rtol=10e-10, atol=10e-10)

# Plot the result
fig, ax = plt.subplots(figsize=(10,15))
x = wsol.t
y = o['X'][o['Fstart']:o['Fend']]
X, Y = np.meshgrid(x, y)
vel = wsol.y[o['Fstart']+o['N']:o['Fend']+o['N'],:]
cs = ax.pcolormesh(Y, X, np.log10(vel), cmap='jet')
cbar = fig.colorbar(cs)
plt.ylabel('Normalized Time')
plt.xlabel('Normalized Distance')