#Konstantin Ntokas
#
#
# 2019-11-17
# -----------------------------------------------------------------------------
# This function decomposes the CRPS into reliability and "potential"
# components according to Hersbach (2000). The potential CRPS
# represents the best possible CRPS value that could be achieved, if the forecasts
# were perfectly reliable. The total (CRPS_tot) is the empirical CRPS according to the definition of
# Hersbach (2000)
#
# Hersbach, H., 2000. Decomposition of the continuous ranked probability score for ensemble
# prediction systems. Weather Forecast. 15, 550–570.
#
# inputs:
#           ens:    mxn matrix; m = number of records (validity dates)
#                                       n = number of members in ensemble
#           obs:    mx1 vector; m = number of records (validity dates, matching the ens)
#
# outputs:
#           CRPS_tot:       Scalar -> reliability + potential
#           reliability:    Scalar -> the reliability component of the CRPS
#           potential:      Scalar -> the potential CRPS that would be reached for a perfectly
#                            reliable system
# -----------------------------------------------------------------------------
import numpy as np


def crps_hersbach_decomposition(ens, obs):
    # preparation
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

    m, n = ens.shape
    alpha = np.zeros((m,n+1))
    beta = np.zeros((m,n+1))

    for i in range(m):
        # if observation does not exist, no ens for alpha and beta
        if ~np.isnan(obs[i]):
            ensemble_sort = np.sort(ens[i]);
            for k in range(n+1):
                if k == 0:
                    if obs[i] < ensemble_sort[0]:
                        alpha[i,k] = 0
                        beta[i,k] = ensemble_sort[0] - obs[i]
                    else:
                        alpha[i,k] = 0
                        beta[i,k] = 0
                elif k == n:
                    if obs[i] > ensemble_sort[n-1]:
                        alpha[i,k] = obs[i] - ensemble_sort[n-1]
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

    g = alpha1 + beta1
    o = beta1 / g

    weight = np.arange(n+1) / n
    reliability = np.nansum(g * np.power(o - weight, 2))
    potential = np.nansum(g * o * (1 - o))
    CRPS_tot = reliability + potential

    return CRPS_tot, reliability, potential