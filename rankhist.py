# Konstantin Ntokas
#
#
# 2019-11-18
# -----------------------------------------------------------------------------
# This function computes the required variables to plot rank histograms 
# (Hamill and Colucci, 1997; Talagrand et al., 1997)
# matrice de previsions d'ensembles. 
# ********. NOTE: The function does *not* plot the histogram! Should you want to plot it, 
# you would have to use (for instance) the "hist" function from the matplotlib.pyplot  
# module. 
#
# Hamill, T.M. and S.J. Colucci. 1997. “Verification of Eta-RSM short-range ensemble
# forecasts.” Monthly Weather Review, 125, 1312-1327
# Talagrand, O., R. Vautard and B. Strauss. 1997. “Evaluation of probabilistic 
#prediction systems.” Proceedings of the ECMWF Workshop on
# predictability, ECMWF, Reading, UK, 1-25
#
# inputs: 
#--------------> ens    =   a (n x m) matrix, where n is the number of time steps 
#                           (validity dates) and m is the ensemble size
#
#--------------> obs    =   a (n x 1) vector, with n the number of time steps 
#                           (validity dates).
# outputs:   
#--------------> Freq     =   Frequency for each bin (frequency with 
#                           which the observation falls into each bin)
#--------------> bins     =   One bin per ensemble member, plus one.
# -----------------------------------------------------------------------------
import numpy as np


def rankhist(ens, obs):
    # transform input into numpy array 
    ens = np.array(ens, dtype='float64')
    obs = np.array(obs, dtype='float64')
    dim1 = ens.shape
    if len(dim1) == 1: 
        ens = ens.reshape((1,dim1[0]))
    dim2 = obs.shape
    if len(dim2) == 0: 
        obs = obs.reshape((1,1))
    elif len(dim2) == 1:
        obs = obs.reshape((dim2[0],1))
        
    #initialisation
    m, n = ens.shape
    ranks = np.empty(m)
    ranks[:] =  np.nan

    # loop over each ensemle sample     
    for i in range(m):
        if ~np.isnan(obs[i]):
            if np.all(~np.isnan(ens[i,:])):
                ens_obs = np.append(ens[i,:], obs[i])
                ens_obs_sort = np.sort(ens_obs)
                idxs, = np.where(ens_obs_sort == obs[i])
                if len(idxs) > 1:  
                    rand_idx = int(np.floor(np.random.rand(1) * len(idxs)))
                    ranks[i] = idxs[rand_idx]
                else: 
                    ranks[i] = idxs[0]   
    
    # find the frequency of each bin 
    ranks_nonnan = ranks[~np.isnan(ranks)]
    bins = np.arange(-0.5, n+1.5, 1)
    Freq, bin_edges = np.histogram(ranks_nonnan, bins=bins)

    return Freq, bins
