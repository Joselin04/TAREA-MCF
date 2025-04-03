import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm

#Funcion para obtener los datos historicos de nuestro activo, en este caso BIMBO, desde 01 de enero de 2010 hasta la fecha del dia de hoy.
def obtener_datos(accion):
    df = yf.download(accion, start="2010-01-01")['Close']
    return df

#Funcion para calcular nuestros rendimientos diarios a partir de los datos hsitoricos.
def calcular_rendimientos(df):
    return df.pct_change().dropna()

#Funcion para calcular el VaR 
def VaR_parametrico(a, media, stdv, grd_lib):
    pVaR_n = (norm.ppf(1-a, media, stdv,))
    pVaR_t = (stats.t.ppf(1-a, grd_lib, media,stdv))
    return pVaR_n, pVaR_t

def VaR_MonteCarlo (a, media, stdv, grd_lib):
    simulaciones = 100000
    sim_returns_n = np.random.normal(media, stdv, simulaciones)
    sim_returns_t = stats.t.rvs(grd_lib, media, stdv, simulaciones)

    MCVaR_n = np.percentile(sim_returns_n,(1-a)*100)
    MCVaR_t = np.percentile(sim_returns_t,(1-a)*100)
    return MCVaR_n, MCVaR_t

def VaR_historico (df, a):
    hVaR = (df.quantile(1-a))
    return hVaR

#Funcion para calcular ES
def CVaR (df, VaR):
    CVaR = np.mean(df[df <= VaR])
    return CVaR

#Funcion para calcular el VaR rolling parametrico
def VaR_rolling_p (alpha, media, desv):
    VaR_n_rolling = norm.ppf(1-alpha, media, desv)
    VaR_n_rll_percent = (VaR_n_rolling * 100).round(4)
    return VaR_n_rll_percent

#Funcion para calcular el VaR rolling historico
def VaR_rolling_his (df, a):
    hVaR = (df.quantile(1-a))
    hVaR_perc = (hVaR*100).round(4)
    return hVaR_perc

#Funcion para calcular ES Rolling
def CVaR_rolling (df, VaR):
    CVaR = np.mean(df[df <= VaR])
    CVaR_perc = round(CVaR,4)
    return CVaR_perc
