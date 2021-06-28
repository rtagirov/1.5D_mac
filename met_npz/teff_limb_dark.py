import numpy as np
from scipy.optimize import curve_fit

def Limb_model(x,a1,a2,a3):
    limb = 1. - a1*(1-x) - a2*(1-x**1.5) - a3*(1-x**2)

    return limb

def Teff1(intensity,wave,mu_indx):
    # should probably shift more stuff into subroutines
    """
    Input: 
    intensity='meanint'
    wave='w'
    mu_indx=[0,2,4,6,8]

    Output: 
    effective temperature 
    """
    cl=2.9979e+08                   # speed of light in SI units
    sig=5.67e-08                    # Stefan-Boltzmann in SI units
    pi=2*np.arccos(0)

    muext = np.arange(21)/20   # mu-grid for final integration, 
                               # assumption: input mu is [.2,.3,..,.9,1]

    nw=intensity.shape[1]
    # should have check here that nw is indeed wave.size
    nmu=intensity.shape[0]
    mu=0.2+np.arange(nmu)/10

    imu=mu_indx.size        # not all mu values are filled, only imu of them
    bolint=np.zeros(imu)
    # integrate to get bolometric intensities
    wv=1e-09*wave
    for i in range(0,imu): 
        j = mu_indx[i]
        intw=(0.001*cl/wv**2)*np.reshape(intensity[j,:],nw) 
        bolint[i]=np.trapz(intw,wv)


    # normalise and fit
    # y = bolint/bolint(imu-1)
    np.savetxt('bolints',bolint)
    init_vals = [1, 1, 0.5]
    best_vals, covar = curve_fit(Limb_model,mu[mu_indx],bolint/bolint[imu-1], p0=init_vals)
    np.savetxt('fitvals',best_vals)
    # evaluate fitted bolometric intensities on new mugrid
    boliext = Limb_model(muext,best_vals[0],best_vals[1],best_vals[2])
    flx = bolint[imu-1]*np.trapz(boliext*muext,muext)
    Teff1 = (2*pi*flx/sig)**0.25

    return Teff1
