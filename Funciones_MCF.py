import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm

def obtener_datos(accion):
    df = yf.download(accion, start="2010-01-01")['Close']
    return df

def calcular_rendimientos(df):
    return df.pct_change().dropna()

'''def VaR_normal(media, stdv, df_rend):
    alphas = [0.95, 0.975, 0.99]
    VaR_n = []
    for a in alphas:
        #VaR parametrico
        VaR_p = (norm.ppf(1-a, media, stdv))
        #VaR historico
        VaR_h = (df_rend.quantile(1-a))
        #VaR Monte Carlo
        simulaciones = 100000
        sim_returns_n = np.random.normal(media,stdv,simulaciones)
        VaR_MC = np.percentile(sim_returns_n,(1-a)*100)
        VaR_n.append((VaR_p, VaR_h, VaR_MC))
    return (VaR_n)'''
