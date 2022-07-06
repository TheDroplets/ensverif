#Konstantin Ntokas
#
#
# 2019-11-17
# -----------------------------------------------------------------------------
# This function computes the logarithmic (or ignorance) score. Predictive distributions can
# be considered as Gaussian, Gamma distributed, Empirical or "Loi des fuites"
# (a Gamma distribution + a Dirac at zero, suitable for daily precip), and Kernel distribution.
#
# inputs: 
#           ens:    mxn matrix; m = number of records (validity dates)  
#                                       n = number of members in ensemble 
#           obs:    mx1 vector; m = number of records (validity dates, matching the ens)
#           distribution:   - 'Normal'
#                           - 'Gamma'
#                           - 'Kernel'
#                           - 'Fuites'  is made for daily precipitation exclusively 
#                           - 'Empirical'
#           thres:          probability density threshold below which we consider that the
#                           event was missed by the forecasting system. This value must be
#                           small (e.g.: 0.0001 means that f(obs) given the forecasts is 
#                           only 0.0001 --> not forecasted).
#                           By default, thres = 0 and the logarithmic score is unbounded.
#          options         - if 'distribution' = 'Fuites', opt_cas is the threshold to determine data
#                             which contributed to gamma distribution and those who are part of the
#                             Dirac impulsion
#                           - if 'distribution' = 'empirical', opt_cas needed is the number of bins
#                             in which to divide the ensemble, by default, it will be the
#                             number of members (Nan excluded). opt_cas have to be an integer
#                             superior to 1.
#
# outputs:
#          S_LOG:           the logarithmic score (scalar)
#          ind_miss:        Boleans to point out days for which the event was missed according
#                           to the threshold specified by the user (1= missed) (n*1 matrix)
#
# Reference:
#          'Empirical' case is based on Roulston and Smith (2002) with
#          modifications -> quantile and members with similar values
#
#  MARK S. ROULSTON AND LEONARD A. SMITH (2002) "Evaluating Probabilistic Forecasts 
# Using Information Theory", Monthly Weather Review, 130, 1653-1660.
# -----------------------------------------------------------------------------
# History
#
# MAB June 19: Added 2 cases for the empirical distribution: the
# observation can either be the smallest or the largest member of the
# augmented ensemble, in which case we can't use the "DeltaX = X(S+1) -
# X(S-1);" equation.
# -----------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm, gamma, gaussian_kde
import sys


def logscore(ens, obs, distribution, thres=0, options=None):
    # transform inputs into numpy array 
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
        
    # preparation 
    n = np.size(ens, axis=0)
    loga = np.empty(n)
    loga[:] = np.nan
    ind_miss = np.empty(n)
    ind_miss[:] = np.nan
    
    # test if input arguments are correct
    if len(obs) != n:
        sys.exit('Error! The length of the record of observations doesn''t match the length of the forecasting period')
    if thres == 0:
        print('Thelogarithmic score is unbounded')
    elif (thres < 0) or (thres > 1):
        sys.exit('The Threshold has to be between 0 and 1.')
    
    # calculation depending on the distribution
    if distribution == 'Empirical':
        # if no "options" is given, number of bins are determined by the number of nonNaN members
        if options == None:
            print('Bins used for empirical method determined by ensemble members')
        elif (options < 2) or (not isinstance(options, int)):
            sys.exit('Format of options is not valide.')
        
        if  not isinstance(thres, float):
            sys.exit('Format of threshold is not valide. thres needs to be a list with 2 entries, determining the upper and lower bound for aberrant values')

        # loop over the records
        for j in range(n):
            # determine of observation is in the bound of max min of ensemble 
            if ~np.all(np.isnan(ens[j,:])):
                if (np.nanmin(ens[j,:]) <= obs[j]) and (obs[j] <= np.nanmax(ens[j,:])):
                    ind_miss[j] = 0
                    # suppress NaN from the ensemble to determine the number of members
                    sample_nonnan = ens[j,:][~np.isnan(ens[j,:])]
                    sort_sample_nonnan = np.sort(sample_nonnan)
                    
                    # transform data, if bins are specified by user in the options argument 
                    if options != None:
                        sort_sample_nonnan = np.quantile(sort_sample_nonnan, np.arange(0, 1, 1/options))
                    
                    # number of bins 
                    N = len(sort_sample_nonnan) 
                    
                    # if all members of forcast and obervation are the same -> perfect forecast
                    if len(np.unique(np.append(sort_sample_nonnan, obs[j]))) == 1:
                        proba_obs = 1
                    else:
                        # if some members are equal, modify the value slightly 
                        if len(np.unique(sort_sample_nonnan)) != len(sort_sample_nonnan):
                            uni_sample = np.unique(sort_sample_nonnan)
                            bins = np.append(uni_sample, np.inf)
                            hist, binedges = np.histogram(sort_sample_nonnan, bins)
                            idxs, = np.where(hist > 1)
                            new_sample = uni_sample
                            for idx in idxs:
                                new_val = uni_sample[idx] + 0.01 *  np.random.rand(hist[idx]-1)
                                new_sample = np.append(new_sample, new_val)
                            sort_sample_nonnan = np.sort(new_sample)
                        # find position of the observation in the ensemble  
                        X = np.sort(np.concatenate((sort_sample_nonnan, obs[j])))
                        S, = np.where(X == obs[j])
                        # if observation is at the first or last position of the ensemble -> threshold prob
                        if S[0] == len(X)-1: 
                            proba_obs = thres
                        elif S[0] == 0: 
                            proba_obs = thres
                        else:
                            #if the observation falls between two members or occupies the first or last rank
                            if len(S) == 1:
                                # If the observation is between the augmented ensemble bounds
                                DeltaX = X[S[0]+1] - X[S[0]-1]
                                proba_obs = min(1/(DeltaX * (N+1)),1)
                            # if observation is equal to one member, choose the maximum of the probability density associated
                            elif len(S) == 2:
                                DeltaX1 = X[S[1]+1] - X[S[1]]
                                DeltaX2 = X[S[0]] - X[S[0]-1]
                                DeltaX = min(DeltaX1,DeltaX2)
                                proba_obs = min(1/(DeltaX * (N+1)),1)
                            # test if probability below threshold
                            if proba_obs < thres:
                                proba_obs = thres
                                ind_miss[j] = 1
                # if the observation is outside the bounds of the ensemble             
                else:
                    ind_miss[j] = 1
                    proba_obs = thres
                
                # calculate the logarithm 
                loga[j] = - np.log2(proba_obs)
            # if all members of the ensemble are nan   
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan
            
    elif distribution == 'Normal':
        if (options != None):
            sys.exit('No options possible for Normal distribution')
        for j in range(n):
            # filter non nan values 
            sample_nonnan = ens[j,:][~np.isnan(ens[j,:])]
            # if there are values in the ensemble which are not nan
            if len(sample_nonnan) > 0:
                # perfect forecast, all member values are equal to the observation
                if len(np.unique(np.append(sample_nonnan, obs[j]))) == 1:
                    proba_obs = 1
                    ind_miss[j] = 0
                    loga[j] = - np.log2(proba_obs)
                else:
                    mu, sig = norm.fit(sample_nonnan)
                    # transform standard deviation to unbiased estimation of standard deviation
                    nb_mb = len(sample_nonnan)
                    sighat = nb_mb/(nb_mb-1) * sig
                    # all members are the same, but unequal the the observation
                    if sighat == 0:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
                    else:
                        proba_obs = min(norm.pdf(obs[j], mu, sighat), 1)
                        if proba_obs >= thres:
                            ind_miss[j] = 0
                            loga[j] = - np.log2(proba_obs)
                        else:
                            loga[j] = - np.log2(thres)
                            ind_miss[j] = 1
            # if all values in the ensemble are nan      
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan
                
    elif distribution == 'Gamma':
        if (options != None):
            sys.exit('No options possible for Gamma distribution')
        # check if any value is smaller or equal to zero
        idxs = np.where(ens <= 0)
        if len(idxs[0]) == 0:
            for j in range(n):
                # filter non nan values 
                sample_nonnan = ens[j,:][~np.isnan(ens[j,:])]
                # if there are values in the ensemble which are not nan
                if len(sample_nonnan) > 0:
                    if len(np.unique(np.append(sample_nonnan, obs[j]))) == 1:
                        proba_obs = 1
                        ind_miss[j] = 0
                        loga[j] = - np.log2(proba_obs)
                    else:
                        # fit gamma distribtion to data
                        alpha, loc, beta = gamma.fit(sample_nonnan, floc=0)
                        proba_obs = min(gamma.pdf(obs[j], alpha, loc, beta), 1)
                        if (alpha <= 0) or (beta <= 0):
                            loga[j] = - np.log2(thres)
                            ind_miss[j] = 1
                        else:
                            if proba_obs >= thres:
                                ind_miss[j] = 0
                                loga[j] = - np.log2(proba_obs)
                            else:
                                loga[j] = - np.log2(thres)
                                ind_miss[j] = 1
                # if all values in the ensemble are nan      
                else:
                    loga[j] = np.nan
                    ind_miss[j] = np.nan
                
        else:
            sys.exit('Forecasts contain zeros. You must choose a different distribution.')

    elif distribution == 'Kernel':
        if (options != None):
            sys.exit('No options possible for Kernel distribution')
            
        for j in range(n):
            # filter non nan values 
            sample_nonnan = ens[j,:][~np.isnan(ens[j,:])]
            # if there are values in the ensemble which are not nan
            if len(sample_nonnan) > 0:
                # perfect forecast, all member values are equal to the observation
                if len(np.unique(np.append(sample_nonnan, obs[j]))) == 1:
                    proba_obs = 1
                    ind_miss[j] = 0
                    loga[j] = - np.log2(proba_obs)
                else:
                    # all member are the same, but unequal to the observation
                    if len(np.unique(sample_nonnan)) == 1:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
                    else:
                        pd = gaussian_kde(sample_nonnan)
                        proba_obs = min(pd.pdf(obs[j]),1)
                        if proba_obs >= thres:
                            ind_miss[j] = 0
                            loga[j] = - np.log2(proba_obs)
                        else:
                            loga[j] = - np.log2(thres)
                            ind_miss[j] = 1
            # if all values in the ensemble are nan      
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan
                   
    elif distribution == 'Fuites':
        if options == None:
            sys.exit('Option missing for ''Fuites'' distribution.')
            
        for j in range(n):
            # filter non nan values 
            sample_nonnan = ens[j,:][~np.isnan(ens[j,:])]
            # if there are values in the ensemble which are not nan
            if len(sample_nonnan) > 0:
                # perfect forecast, all member values are equal to the observation
                if len(np.unique(np.append(sample_nonnan, obs[j]))) == 1:
                    proba_obs = 1
                    ind_miss[j] = 0
                    loga[j] = - np.log2(proba_obs)
                else:
                    idx_non_null, = np.where(sample_nonnan > options)
                    prop_null = (len(sample_nonnan) - len(idx_non_null)) / len(sample_nonnan)
                    if obs[j] <= options:
                        proba_obs = prop_null
                    else:
                        ens_non_null = sample_nonnan[idx_non_null]
                        # all member values above the threshold are equal, but unequal to observation
                        if len(np.unique(ens_non_null)) == 1:
                            proba_obs = thres
                        else:
                            # Fitting gamma parameters (max. likelihood method))
                            alpha, loc, beta = gamma.fit(ens_non_null, floc=0)
                            obs_val = gamma.pdf(obs[j], alpha, loc, beta) * (1-prop_null)
                            proba_obs = min(obs_val, 1)
                    # check if probability is above threshold
                    if proba_obs > thres:
                        loga[j] = - np.log2(proba_obs)
                        ind_miss[j] = 0
                    else:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
            # if all values in the ensemble are nan      
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan
            
    else:
        sys.exit('Choice of distribution type in ''distribution'' is incorrect. Possible options are : "Normal", "Gamma", "Kernel", "Empirical" or "Fuites" ')
    
    S_LOG = np.nanmean(loga)
    ind_miss = np.nansum(ind_miss) 

    return S_LOG, ind_miss