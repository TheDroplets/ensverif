#Konstantin Ntokas
#
#
# 2019-11-17
# -----------------------------------------------------------------------------
# This function computes what is needed to plot (modified) reliability diagrams.
# Note that this function does *not* adopt the definition of Murphy and Winkler (1977), 
# which would require the transformation of the ensemble into a binary forecast using 
# a threshold.
# Instead, we verify whether each confidence interval (i.e. 10%,..., 90%) corresponds
# to its definition: for instance the 10% confidence interval should include 10% of 
# observations.
#
# This function *does not* plot the diagram. To plot the diagram, one has to type 
# "import matplotlib.pyplot as plt" and then "plt.plot(nominal_coverage, effective_coverage)"
#
# Finally, as an additional information, we also compute the mean length of each 
# confidence interval. A variant of the reliability diagram could be:
# "import matplotlib.pyplot as plt" and then "plt.plot(length, effective_coverage)", 
#as in Boucher et al. (2010)
#
# References:
#
# Murphy AH and Winkler RL (1977) Reliability of Subjective Probability Forecasts of 
# Precipitation and Temperature, Journal of the Royal Statistical Society: Series C
# (Applied Statistics) 26: 41-47
#
# Boucher M-A, Laliberte J-P and Anctil F. (2010) An experiment on the evolution of 
# an ensemble of neural networks for streamflow forecasting, 
# Hydrol. Earth Syst. Sci., 14, 603–612
#
# inputs: 
#           ens:    mxn matrix; m = number of records (validity dates)  
#                                       n = number of members in ensemble 
#           obs:    mx1 vector; m = number of records (validity dates, matching the ens) 
#
# outputs:
#   nominal_coverage    : nominal probability of the intervals
#   effective_coverage  : effective coverage of the intervals
#   Length              : mean effective length of the intervals
# -----------------------------------------------------------------------------
import numpy as np


def reliability(ens, obs):
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
    nb = len(bin_start)
    nominal_coverage = bin_end - bin_start
    
    # initialisation
    L = np.size(ens, axis=0)
    length = np.zeros((L,nb))
    eff = np.zeros((L,nb))
    
    for i in range(L):
        # get quantile for each bin and the median
        q = np.quantile(ens[i,:], bins)
        qmed = np.median(ens[i,:])
            
        # Compute lengths of intervals 
        length[i,:] = q[nb:] - q[:nb]
        
        # Locate observation in the ensemble
        if obs[i] <= qmed:
            eff[i,:] = q[:nb] <= obs[i]
        else:
            eff[i,:] = q[nb:] >= obs[i]
    
    # Compute averages
    effective_coverage = np.nanmean(eff, axis=0)
    Length = np.nanmean(length, axis=0)  

    return nominal_coverage, effective_coverage, Length