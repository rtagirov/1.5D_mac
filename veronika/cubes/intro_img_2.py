import netCDF4
import itertools

import numpy as np

from tqdm import tqdm

Nx = 512
Ny = 512
Nz = 601
Na = 5
Ns = 25

mu_id = ['10', '8', '6', '4', '2']

T = np.zeros((Na, Nz, Ny * Nx))
t = np.zeros((Na, Nz, Ny * Nx))

T_ssd = np.zeros((Na, Nz, Ns))
t_ssd = np.zeros((Na, Nz, Ns))

T_300 = np.zeros((Na, Nz, Ns))
t_300 = np.zeros((Na, Nz, Ns))

for i in tqdm(range(Na)):
    
    T[i, :, :] = np.array(netCDF4.Dataset('./ssd/222809/' + mu_id[i] + '/T_onTau.222809.nc.1')['T']).reshape(Nz, Ny * Nx)
    t[i, :, :] = np.array(netCDF4.Dataset('./ssd/222809/' + mu_id[i] + '/taugrid.222809.nc.1')['tau']).reshape(Nz, Ny * Nx)
    
    sample = np.random.choice(Nx * Ny, Ns, replace = False)
    
    T_ssd[i, :, :] = T[i, :, sample].T
    t_ssd[i, :, :] = t[i, :, sample].T

    T[i, :, :] = np.array(netCDF4.Dataset('./300G/627321/' + mu_id[i] + '/T_onTau.627321.nc.1')['T']).reshape(Nz, Ny * Nx)
    t[i, :, :] = np.array(netCDF4.Dataset('./300G/627321/' + mu_id[i] + '/taugrid.627321.nc.1')['tau']).reshape(Nz, Ny * Nx)
    
    sample = np.random.choice(Nx * Ny, Ns, replace = False)
    
    T_300[i, :, :] = T[i, :, sample].T
    t_300[i, :, :] = t[i, :, sample].T
    
#    for k in range(Ns):
        
#        for j in range(Nz - 1):
            
#            delta = np.abs(np.log10(t_ssd[i, j, k]) - np.log10(t_ssd[i, j + 1, k]))
        
#            if np.abs(delta - 0.0001) < 1e-6: T_ssd[i, j, k] = np.nan

#            delta = np.abs(np.log10(t_300[i, j, k]) - np.log10(t_300[i, j + 1, k]))
        
#            if np.abs(delta - 0.0001) < 1e-6: T_300[i, j, k] = np.nan

np.savez('intro_img_2', t_ssd = t_ssd, t_300 = t_300, T_ssd = T_ssd, T_300 = T_300)
