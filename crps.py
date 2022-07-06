
# Alireza Amani and Konstantin Ntokas
#
#
# 2019-11-16
# -----------------------------------------------------------------------------
# This function computes the CRPS for a series of ensemble forecasts (ens) and corresponding
# observations (obs). The user can choose which distribution to fit to the ensemble for 
# computation.
#
# INPUTS
#-----------------------------------------------------------------------------------
# ens: a (n x m) matrix, where n is the number of time steps (validity dates) and m is the ensemble size
# obs: a (n x 1) vector, with n the number of time steps (validity dates). 
# distribution = 'emp': an empirical distribution is computed from the ensemble forecasts
# distribution = 'normal_exact':  a normal (Gaussian) distribution is fitted to the ensemble forecasts. 
# The CRPS is computed using an exact formulation derived from the equation of the normal distribution. 
#or 
# distribution = 'gamma_approx': a gamma distribution is fitted to the ensemble forecasts. The CRPS
# is computed using an approximate formulation as per Gneiting and Raftery (2007) equation (21)
# Tilmann Gneiting and Adrian E. Raftery (2007): Strictly Proper Scoring Rules, Prediction, 
# and Estimation, Journal of the American Statistical Association, 102:477, 359-378
#
#-----------------------------------------------------------------------------------
# OUTPUT:
# CRPS: a scalar, representing the mean CRPS for the time series of forecasts 
#
# --------------------------------------------------------------------------------
# Original by Marie-Amelie Boucher, July 2005, Universite Laval
# Modified by Rachel Bazile and Luc Perreault in June 2017: 
# The computation for the empirical version of the CRPS cannot be based on just a few
# ensemble members so we need to "subsample" to have enough members
#
# -----------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm, gamma
 
def crps(ens, obs, distribution):
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

    # initilisation      
    m = np.size(ens, axis=0)
    crps = np.empty((m, 1))
    crps.fill(np.nan)
    
    # non-parametric estimation based on the empirical cumulative distribution of the ensemble. According to Luc Perreault's idea
    if (distribution == "emp"):
        for i in range(m):
            if (np.any(np.isnan(ens[i,:])) == 0 and np.isnan(obs[i]) == 0):
                ssample = np.sort(ens[i,:])
                step_size = 1/(len(ens[i,:]))
                
                # calculation of the area below the observation
                area1 = 0
                sub_sample1 = ssample[ssample <= obs[i]]
                sub_sample1 = np.append(sub_sample1, obs[i])
                for j in range(1,len(sub_sample1)):
                    area1 += (j*step_size)**2 * (sub_sample1[j] - sub_sample1[j-1])

                # calculation of the area above the observation
                area2 = 0
                sub_sample2 = ssample[ssample > obs[i]]
                sub_sample2 = np.insert(sub_sample2, 0, obs[i])
                n2 = len(sub_sample2)
                for j in range(1,n2):
                    area2 += ((n2-j)*step_size)**2 * (sub_sample2[j] - sub_sample2[j-1])
                    
                crps[i] = area1 + area2
                
            else:
                crps[i] = np.nan
                
    # -------------------------------------------------------------------------
    # estimation based on the normal cumulative distribution of the ensemble               
    elif (distribution == "normal_exact"):
        for i in range(m):
            # preparation
            mu, sigma = norm.fit(ens[i,:])
            # transform standard deviation to unbiased estimation of standard deviation
            nb_mb = len(ens[i,:])
            sighat = nb_mb/(nb_mb-1) * sigma
            vcr = (obs[i] - mu) / sighat
            phi = norm.pdf(vcr,  loc=0, scale=1)
            PHI = norm.cdf(vcr,  loc=0, scale=1)
            # calculation of the CRPS according to Gneiting and Raftery 2007
            crps[i] = abs(sighat * ((1/np.sqrt(np.pi)) - 2*phi - (vcr*(2*PHI-1))))
            
    # -------------------------------------------------------------------------
    # estimation based on the gamma cumulative distribution of the ensemble   
    elif (distribution == "gamma_exact"):
        for i in range(m):
            # preparation; exchange negative values in the data
            sample = ens[i,:]
            idxs, = np.where(sample <= 0)
            for idx in idxs: 
                sample = 0.0001
                
            # fit data to gamma distribution 
            alpha, loc, beta = gamma.fit(sample, floc=0)
            # generate cumulative gamma distribution
            data1 = gamma.rvs(alpha, loc=0, scale=beta, size=1000)  
            data2 = gamma.rvs(alpha, loc=0, scale=beta, size=1000)  
            crps[i]= np.mean(np.absolute(data1 - obs[i])) - 0.5 * np.mean(np.absolute(data1 - data2))
      

    return np.mean(crps) 