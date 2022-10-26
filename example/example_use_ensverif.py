#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct. 25 2022

Example script on how to use the ensverif toolbox, using GloFAS streamflow forecasts
and streamflow observations from gauging station 022513 (Du Loup River
1.4 km downstream of the bridge on road 185, Quebec) in Canada. Note that some
elements of this script are very specific to the forecasts used in the example.
For instance, GloFAS forecasts have 50 members and the example file has a total
of 54 columns, so that is why at line 40 I use "iloc[:,4:54]". This is not general

This script assumes that ensverif was already installed in your Python environment
using pip, or from github

@author: Marie-Amelie Boucher marie-amelie.boucher@usherbrooke.ca
"""

import pandas as pd
import ensverif
import matplotlib.pyplot as plt

fcsts_all = pd.read_csv("/Users/marieamelie/Desktop/glofas_fcst_station_022513.csv")
obs_all = pd.read_csv("/Users/marieamelie/Desktop/obs_station_022513.csv")

# Pick a lead time, for instance 24h

fcsts_24h  = fcsts_all[fcsts_all["horizon"]==24]

# The time series for the forecasts and observations are not the same
# (ex. the forecasts start on 2021-08-01 while the obs start on 2022-02-13)
# We must define a time period that is common to both the forecasts and obs.
# For the forecasts, we use the validity date ("echeance") (not the emission date)

obs_valid = obs_all[obs_all['Date'].isin(fcsts_24h.echeance)]
fcsts_24h_valid = fcsts_24h[fcsts_24h['echeance'].isin(obs_valid.Date)]

# Compute the CRPS using the empirical distribution of forecast members
# (other options are possible).
crps_24h = ensverif.crps(fcsts_24h_valid.iloc[:,4:54].values, obs_valid.Obs, 'emp')

# CRPS decomposition according to Hersbach (2000)
crps_hersbach_24h, rel_24h, pot_24h = ensverif.crps_hersbach_decomposition(fcsts_24h_valid.iloc[:,4:54].values, obs_valid.Obs)

# Logarithmic (or ignorance) score
slog_24h = ensverif.logscore(fcsts_24h_valid.iloc[:,4:54].values, obs_valid.Obs, 'Normal', thres=0.001)

# Rank histogram
rel_freq, bins = ensverif.rankhist(fcsts_24h_valid.iloc[:,4:54].values, obs_valid.Obs)
plt.figure()
plt.bar(bins[1:len(bins)],rel_freq)
plt.xlabel('Rank')
plt.ylabel('Relative frequency')

# Reliability diagram
nominal_coverage, effective_coverage, length_intervals = ensverif.reliability(fcsts_24h_valid.iloc[:,4:54].values, obs_valid.Obs)
plt.figure()
plt.plot(nominal_coverage, effective_coverage,'.')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.xlabel('Nominal coverage')
plt.ylabel('Effective coverage')
