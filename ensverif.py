#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:51:43 2022

ensverif module

Contains several functions to compute performance scores for ensemble forecasts
or simulations, as well as functions to assist plotting graphical performance
assessment tools (reliability diagram and rank histogram)

@authors: Marie-Amelie Boucher, Rachel Bazile, Konstantin Ntokas, Alireza Amani

marie-amelie.boucher@usherbrooke.ca
This is an improvement over the first release of ensverif. The changes are:
    - Place all the functions in one single module instead of one function per
    module, thus facilitating the call to each function and the use of this module
    - Replace np.mean by np.nanmean in the CRPS function
    - Correct spelling mistake
    - Correct convention mistakes (not perfect yet, but better)
    - Improve documentation (more concise + adding an example on how to use the module)

"""

import sys
import numpy as np
from scipy.stats import norm, gamma, gaussian_kde


def crps(ens, obs, distribution):
    """
    This function computes the Continuous Ranked Probability Score according and
    the user can choose between different methods.

    Parameters
    ----------
    ens : Ensemble forecasts or ensemble simulations. It must be a T x M matrix,
        with T the time steps and M the members.
    obs : A vector of corresponding observations to match the forecasts. It must
        be a T x 1 vector
    distribution : String.
        --> ‘emp’: an empirical distribution is computed from the ensemble
        --> ‘normal_exact’: a normal (Gaussian) distribution is fitted to the ensemble
            forecasts. The CRPS is computed using an exact formulation derived
            from the equation of the normal distribution.
        --> ‘gamma_approx’: a gamma distribution is fitted to the ensemble

    Returns
    -------
    crps: a scalar, representing the mean CRPS for the time series of forecasts

    Original by Marie-Amelie Boucher, July 2005, Universite Laval Modified by
    Rachel Bazile and Luc Perreault in June 2017: The computation for the
    empirical version of the CRPS cannot be based on just a few ensemble
    members so we need to “subsample” to have enough members

    The CRPS is computed using an approximate formulation as per
    Gneiting and Raftery (2007) equation (21)
    Tilmann Gneiting and Adrian E. Raftery (2007): Strictly Proper Scoring Rules,
    Prediction, and Estimation, Journal of the American Statistical Association,
    102:477, 359-378

    """
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
    nb_columns = np.size(ens, axis=0)
    crps_matrix = np.empty((nb_columns, 1))
    crps_matrix.fill(np.nan)

    # non-parametric estimation based on the empirical cumulative distribution
    # of the ensemble. According to Luc Perreault's idea
    if distribution == "emp":
        for i in range(nb_columns):
            if np.any(np.isnan(ens[i,:])) == 0 and np.isnan(obs[i]) == 0:
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
                len_subsample2 = len(sub_sample2)
                for j in range(1,len_subsample2):
                    area2 += ((len_subsample2-j)*step_size)**2 * (sub_sample2[j] - sub_sample2[j-1])

                crps_matrix[i] = area1 + area2

            else:
                crps_matrix[i] = np.nan

    # -------------------------------------------------------------------------
    # Estimation based on the normal cumulative distribution of the ensemble
    elif distribution == "normal_exact":
        for i in range(nb_columns):
            # preparation
            muhat, sigma = norm.fit(ens[i,:])
            # Transform the standard deviation to an unbiased estimator
            nb_mb = len(ens[i,:])
            sighat = nb_mb/(nb_mb-1) * sigma
            vcr = (obs[i] - muhat) / sighat
            phi1 = norm.pdf(vcr,  loc=0, scale=1)
            phi2 = norm.cdf(vcr,  loc=0, scale=1)

            crps_matrix[i] = abs(sighat * ((1/np.sqrt(np.pi)) - 2*phi1 - (vcr*(2*phi2-1))))

    # -------------------------------------------------------------------------
    # Estimation based on the gamma cumulative distribution of the ensemble
    elif distribution == "gamma_exact":
        for i in range(nb_columns):
            # preparation; remove any negative values in the data
            sample = ens[i,:]
            idxs, = np.where(sample <= 0)
            for idx in idxs:
                sample[idx] = 0.0001

            # fit gamma distribution
            alpha, _loc, beta = gamma.fit(sample, floc=0)
            # generate cumulative gamma distribution
            data1 = gamma.rvs(alpha, loc=0, scale=beta, size=1000)
            data2 = gamma.rvs(alpha, loc=0, scale=beta, size=1000)
            crps_matrix[i]= np.nanmean(np.absolute(data1-obs[i]))-0.5*np.nanmean(np.absolute(data1-data2))

    return np.nanmean(crps_matrix)

def crps_hersbach_decomposition(ens, obs):
    """
    This function decomposes the CRPS into reliability and "potential"
    components according to Hersbach (2000). The potential CRPS
    represents the best possible CRPS value that could be achieved, if forecasts
    were perfectly reliable.

    Parameters
    ----------
    ens : Ensemble forecasts or ensemble simulations. It must be a T x M matrix,
        with T the time steps and M the members.
    obs : A vector of corresponding observations to match the forecasts. It must
        be a T x 1 vector

    Returns
    -------
    crps_tot : The total CRPS (reliability + potential)
    reliability_component : The reliability component of the CRPS according to Hersbach (2000)
    potential_component : The potential component of the CRPS according to Hersbach (2000)

    Hersbach, H., 2000. Decomposition of the continuous ranked probability score \
    for ensemble prediction systems. Weather Forecast. 15, 550–570.

    """

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

    rows, columns = ens.shape
    alpha = np.zeros((rows,columns+1))
    beta = np.zeros((rows,columns+1))

    for i in range(rows):
        # if the observation does not exist, no ens for alpha and beta
        if ~np.isnan(obs[i]):
            ensemble_sort = np.sort(ens[i])
            for k in range(columns+1):
                if k == 0:
                    if obs[i] < ensemble_sort[0]:
                        alpha[i,k] = 0
                        beta[i,k] = ensemble_sort[0] - obs[i]
                    else:
                        alpha[i,k] = 0
                        beta[i,k] = 0
                elif k == columns:
                    if obs[i] > ensemble_sort[columns-1]:
                        alpha[i,k] = obs[i] - ensemble_sort[columns-1]
                        beta[i,k] = 0
                    else:
                        alpha[i,k] = 0
                        beta[i,k] = 0
                else:
                    if obs[i] > ensemble_sort[k]:
                        alpha[i,k] = ensemble_sort[k] - ensemble_sort[k-1]
                        beta[i,k] = 0
                    elif obs[i] < ensemble_sort[k-1]:
                        alpha[i,k] = 0
                        beta[i,k] = ensemble_sort[k] - ensemble_sort[k-1]
                    elif (obs[i] >= ensemble_sort[k-1]) and (obs[i] <= ensemble_sort[k]):
                        alpha[i,k] = obs[i] - ensemble_sort[k-1]
                        beta[i,k] = ensemble_sort[k] - obs[i]
                    else:
                        alpha[i,k] = np.nan
                        beta[i,k] = np.nan
        else:
            alpha[i,:] = np.nan
            beta[i,:] = np.nan


    alpha1 = np.nanmean(alpha, axis=0)
    beta1 = np.nanmean(beta, axis=0)

    g_component = alpha1 + beta1
    o_component = beta1 / g_component

    weight = np.arange(columns+1) / columns
    reliability_component = np.nansum(g_component * np.power(o_component - weight, 2))
    potential_component = np.nansum(g_component * o_component * (1 - o_component))
    crps_tot = reliability_component + potential_component

    return crps_tot, reliability_component, potential_component

def logscore(ens, obs, distribution, thres=0, options=None):
    """
    This function computes the logarithmic (or ignorance) score. Predictive \
    distributions can be considered as Gaussian, Gamma distributed, Empirical \
    or “Loi des fuites” (a Gamma distribution + a Dirac at zero, suitable for \
    daily precip), and kernel distribution.

    Parameters
    ----------
    ens : Ensemble forecasts or ensemble simulations. It must be a T x M matrix,
        with T the time steps and M the members.
    obs : A vector of corresponding observations to match the forecasts. It must
        be a T x 1 vector
    distribution : String. The possibilities are:
        --> 'Normal'
        --> 'Gamma'
        --> 'Kernel'
        --> 'Fuites' for daily precipitation
        --> 'Empirical'
    thres : Float. Probability density threshold below which we consider that \
        the event was missed by the forecasting system. This value must be \
        small (e.g.: 0.0001 means that f(obs) given the forecasts is only \
        0.0001 –> not forecasted). The default is 0.
    options :
        --> if ‘distribution = ‘Fuites’, 'options' is the threshold to \
        determine which portion of the ensemble to use to fit the gamma distribution\
        and which portion to use for Dirac impulsion
        --> if ‘ditribution’ = ‘empirical’, 'options' is the number of bins \
            used to divide the ensemble (scalar). Must be > 1. The default is the \
            number of ensemble members.

    Returns
    -------
    s_log : Float. Value of the logarithmic (or ignorance) score
    ind_miss : Integer. Number of time steps for which the observation falls \
        outside of the ensemble

    """
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

    nb_rows = np.size(ens, axis=0)
    loga = np.empty(nb_rows)
    loga[:] = np.nan
    ind_miss = np.empty(nb_rows)
    ind_miss[:] = np.nan

    # test if input arguments are correct
    if len(obs) != nb_rows:
        sys.exit('Error! The length of the record of observations doesn''t \
                 match the length of the forecasting period')
    if thres == 0:
        print('The logarithmic score is unbounded')
    elif (thres < 0) or (thres > 1):
        sys.exit('The threshold must be between 0 and 1.')

    # calculation depending on the distribution
    if distribution == 'Empirical':
        # if no "options" are given, the number of bins is the number of nonNaN members
        if (options < 2) or (not isinstance(options, int)):
            sys.exit('Format of options is not valide.')

        if  not isinstance(thres, float):
            sys.exit('Format of threshold is not valid. It needs to be a \
                     list with 2 entries, determining the upper and lower \
                         bound for aberrant values')

        # loop over the records
        for j in range(nb_rows):
            # check if the observation is contained in the ensemble
            if ~np.all(np.isnan(ens[j,:])):
                if (np.nanmin(ens[j,:]) <= obs[j]) and (obs[j] <= np.nanmax(ens[j,:])):
                    ind_miss[j] = 0
                    # suppress NaN from the ensemble to determine the number of members
                    sample_nonnan = ens[j,:][~np.isnan(ens[j,:])]
                    sort_sample_nonnan = np.sort(sample_nonnan)

                    # transform the data, if bins are specified by user in the options argument
                    if options is not None:
                        sort_sample_nonnan=np.quantile(sort_sample_nonnan,np.arange(0,1,1/options))

                    nb_bins = len(sort_sample_nonnan)

                    # if all forecast members+obs are the same -> perfect forecast
                    if len(np.unique(np.append(sort_sample_nonnan, obs[j]))) == 1:
                        proba_obs = 1
                    else:
                        # if some members are equal, modify their values slightly
                        if len(np.unique(sort_sample_nonnan)) != len(sort_sample_nonnan):
                            uni_sample = np.unique(sort_sample_nonnan)
                            bins = np.append(uni_sample, np.inf)
                            hist, _bin_edges = np.histogram(sort_sample_nonnan, bins)
                            idxs, = np.where(hist > 1)
                            new_sample = uni_sample
                            for idx in idxs:
                                new_val = uni_sample[idx] + 0.01 *  np.random.rand(hist[idx]-1)
                                new_sample = np.append(new_sample, new_val)
                            sort_sample_nonnan = np.sort(new_sample)
                        # find the position of the observation in the sorted ensemble
                        ens_sorted = np.sort(np.concatenate((sort_sample_nonnan, obs[j])))
                        rank_obs = np.where(ens_sorted == obs[j])
                        # if the observation occupies rank 1
                        if rank_obs[0] == len(ens_sorted)-1:
                            proba_obs = thres
                        elif rank_obs[0] == 0:
                            proba_obs = thres
                        else:
                            #if the observation falls between two members or
                            #occupies the first or last rank
                            if len(rank_obs) == 1:
                                # If the observation is between the augmented ensemble bounds
                                delta_x = ens_sorted[rank_obs[0]+1] - ens_sorted[rank_obs[0]-1]
                                proba_obs = min(1/(delta_x * (nb_bins+1)),1)
                            # if observation is equal to one member, choose the
                            # maximum of the probability density associated
                            elif len(rank_obs) == 2:
                                delta_x1 = ens_sorted[rank_obs[1]+1] - ens_sorted[rank_obs[1]]
                                delta_x2 = ens_sorted[rank_obs[0]] - ens_sorted[rank_obs[0]-1]
                                delta_x = min(delta_x1,delta_x2)
                                proba_obs = min(1/(delta_x * (nb_bins+1)),1)
                            # test if probability below threshold
                            if proba_obs < thres:
                                proba_obs = thres
                                ind_miss[j] = 1
                # if the observation is outside the ensemble
                else:
                    ind_miss[j] = 1
                    proba_obs = thres

                loga[j] = - np.log2(proba_obs)
            # if all members of the ensemble are nan
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan

    elif distribution == 'Normal':
        if options is not None:
            sys.exit('No options possible for the Normal distribution')
        for j in range(nb_rows):
            # filter non nan values
            sample_nonnan = ens[j,:][~np.isnan(ens[j,:])]
            # if there are values in the ensemble which are not nan
            if len(sample_nonnan) > 0:
                # perfect forecast, all members are equal to the observation
                if len(np.unique(np.append(sample_nonnan, obs[j]))) == 1:
                    proba_obs = 1
                    ind_miss[j] = 0
                    loga[j] = - np.log2(proba_obs)
                else:
                    muhat, sig = norm.fit(sample_nonnan)
                    # transform the standard deviation to an unbiased estimator
                    nb_mb = len(sample_nonnan)
                    sighat = nb_mb/(nb_mb-1) * sig
                    # all members are the same, but unequal the the observation
                    if sighat == 0:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
                    else:
                        proba_obs = min(norm.pdf(obs[j], muhat, sighat), 1)
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
        if options is not None:
            sys.exit('No options possible for the Gamma distribution')
        # check if any value is smaller or equal to zero
        idxs = np.where(ens <= 0)
        if len(idxs[0]) == 0:
            for j in range(nb_rows):
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
        if options is not None:
            sys.exit('No options possible for the kernel distribution')

        for j in range(nb_rows):
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
                    # all members are identical, but unequal to the observation
                    if len(np.unique(sample_nonnan)) == 1:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
                    else:
                        prob_dist = gaussian_kde(sample_nonnan)
                        proba_obs = min(prob_dist.pdf(obs[j]),1)
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
        if options is None:
            sys.exit('Option missing for ''Fuites'' distribution.')

        for j in range(nb_rows):
            # filter non nan values
            sample_nonnan = ens[j,:][~np.isnan(ens[j,:])]
            # if there are values in the ensemble which are not nan
            if len(sample_nonnan) > 0:
                # perfect forecast, all members are equal to the observation
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
                        # all member values above the threshold are identical, but
                        # unequal to the observation
                        if len(np.unique(ens_non_null)) == 1:
                            proba_obs = thres
                        else:
                            # Fitting gamma parameters (max. likelihood method))
                            alpha, loc, beta = gamma.fit(ens_non_null, floc=0)
                            obs_val = gamma.pdf(obs[j], alpha, loc, beta) * (1-prop_null)
                            proba_obs = min(obs_val, 1)
                    # check if the probability is above the threshold
                    if proba_obs > thres:
                        loga[j] = - np.log2(proba_obs)
                        ind_miss[j] = 0
                    else:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
            # if all ensemble members are nan
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan

    else:
        sys.exit('The choice of distribution type in ''distribution'' is incorrect.\
                 Possible options are : "Normal", "Gamma", "Kernel", "Empirical" or "Fuites" ')

    s_log = np.nanmean(loga)
    ind_miss = np.nansum(ind_miss)

    return s_log, ind_miss

def rankhist(ens, obs):
    """
    This function computes the required variables to plot rank histograms
    (Hamill and Colucci, 1997; Talagrand et al., 1997).

    Parameters
    ----------
    ens : Ensemble forecasts or ensemble simulations. It must be a T x M matrix,
        with T the time steps and M the members.
    obs : A vector of corresponding observations to match the forecasts. It must
        be a T x 1 vector

    Returns
    -------
    rel_freq : Array. Relative frequency of each ensemble member
    bins : Array. Position of each bin on the x axis

    Hamill, T.M. and S.J. Colucci. 1997. “Verification of Eta-RSM short-range
    ensemble forecasts.” Monthly Weather Review, 125, 1312-1327.

    Talagrand, O., R. Vautard and B. Strauss. 1997. “Evaluation of probabilistic
    prediction systems.” Proceedings of the ECMWF Workshop on predictability,
    ECMWF, Reading, UK, 1-25.

    """
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


    nb_rows, nb_cols = ens.shape
    ranks = np.empty(nb_rows)
    ranks[:] =  np.nan

    for i in range(nb_rows):
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

    ranks_nonnan = ranks[~np.isnan(ranks)]
    bins = np.arange(-0.5, nb_cols+1.5, 1)
    freq, _bin_edges = np.histogram(ranks_nonnan, bins=bins)
    rel_freq = freq/np.nansum(freq)

    print('Warning: This function does not plot any figure! To plot the rank \
          histogram, first import matplotlib ("import matplotlib.pyplot as plt")\
         and then "plt.bar(bins[1:len(bins)],rel_freq)"')

    return rel_freq, bins

def reliability(ens, obs):
    """
    This function computes what is needed to plot (modified) reliability diagrams.
    Note that this function does not adopt the definition of Murphy and Winkler (1977),
    which would require the transformation of the ensemble into a binary forecast
    using a threshold. Instead, we verify whether each confidence interval
    (i.e. 10%,…, 90%) corresponds to its definition: for instance the 10%
    confidence interval should include 10% of observations.

    Parameters
    ----------
    ens : Ensemble forecasts or ensemble simulations. It must be a T x M matrix,
        with T the time steps and M the members.
    obs : A vector of corresponding observations to match the forecasts. It must
        be a T x 1 vector

    Returns
    -------
    nominal_coverage : Array. Nominal (or theoretical) probabilities of the intervals
    effective_coverage : Array. Effective probabilities of the intervals (observed \
                       relative frequency)
    length_intervals : Array. Length of each probability interval. Optional

    Murphy AH and Winkler RL (1977) Reliability of Subjective Probability Forecasts
    of Precipitation and Temperature, Journal of the Royal Statistical Society:
    Series C (Applied Statistics) 26: 41-47.

    Boucher M-A, Laliberte J-P and Anctil F. (2010) An experiment on the evolution
    of an ensemble of neural networks for streamflow forecasting, Hydrol.
    Earth Syst. Sci., 14, 603–612.

    """
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


    # Nominal probability of the intervals
    bin_start = np.arange(0.05, 0.5, 0.05)
    bin_end = np.arange(0.95, 0.5, -0.05)
    bins = np.concatenate((bin_start, bin_end))
    nb_bins = len(bin_start)
    nominal_coverage = bin_end - bin_start

    # initialisation
    nb_rows = np.size(ens, axis=0)
    length = np.zeros((nb_rows,nb_bins))
    eff = np.zeros((nb_rows,nb_bins))

    for i in range(nb_rows):
        # get quantile for each bin and the median
        quantiles = np.quantile(ens[i,:], bins)
        qmed = np.median(ens[i,:])

        # Compute lengths of intervals
        length[i,:] = quantiles[nb_bins:] - quantiles[:nb_bins]

        # Locate observation in the ensemble
        if obs[i] <= qmed:
            eff[i,:] = quantiles[:nb_bins] <= obs[i]
        else:
            eff[i,:] = quantiles[nb_bins:] >= obs[i]

    # Compute averages
    effective_coverage = np.nanmean(eff, axis=0)
    length_intervals = np.nanmean(length, axis=0)

    print('Warning: This function does not plot any figure! To plot the reliability \
          diagram, first import matplotlib ("import matplotlib.pyplot as plt")\
         and then "plt.plot(nominal_coverage, effective_coverage)" OR \
         “plt.plot(length, effective_coverage)”, as in Boucher et al. (2010)')

    return nominal_coverage, effective_coverage, length_intervals
