# coding: utf-8
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def func(x, u, sig) :
    return  (np.exp(-(x-u)**2/(2*sig**2))/(np.sqrt(2*np.pi)*sig))

def StdFit(PETruth) :
    alpha_label = ParticleTruth.query("Alpha==1")
    alpha_data = PETruth.loc[alpha_label.index]
    alpha_PE_spread = alpha_data.groupby("EventID").apply(lambda x: np.std(x['PETime']))
    alpha_hist, alpha_edge = np.hist(alpha_PE_spread, bins=50, density=True)
    alpha_para, _ = curve_fit(func, alpha_edge, alpha_hist)

    beta_label = ParticleTruth.query("Alpha==0")
    beta_data = PETruth.loc[beta_label.index]
    beta_PE_spread = beta_data.groupby("EventID").apply(lambda x: np.std(x['PETime']))
    beta_hist, beta_edge = np.hist(beta_PE_spread, bins=50, density=True)
    beta_para, _ = curve_fit(func, beta_edge, beta_hist)
    return alpha_para, beta_para 
