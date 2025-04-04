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

def calcular_violaciones(retornos, alpha, ventana=252):
    violaciones_VaR_p = []
    violaciones_CVaR_p = []
    violaciones_VaR_h = []
    violaciones_CVaR_h = []
    
    for i in range(ventana, len(retornos)):
        datos_historicos = retornos[i-ventana:i]
        media = datos_historicos.mean()
        desv = datos_historicos.std()
         #rolling_mean = df_bimbo_rend.rolling(window=252).mean()
         #rolling_std = df_bimbo_rend.rolling(window=252).std()
        
        VaR_p = (VaR_rolling_p(alpha, media, desv))
        CVaR_p = CVaR_rolling(datos_historicos, VaR_p)
        VaR_h = (VaR_rolling_his(datos_historicos, alpha))
        CVaR_h = CVaR_rolling(datos_historicos, VaR_h)

        
        violaciones_VaR_p.append(retornos.iloc[0] <(VaR_p/1000))
        violaciones_CVaR_p.append(retornos.iloc[0] < CVaR_p)
        violaciones_VaR_h.append(retornos.iloc[0] <(VaR_h/1000))
        violaciones_CVaR_h.append(retornos.iloc[0] < CVaR_h)
    
    total_VaR_p = sum(violaciones_VaR_p)
    total_CVaR_p = sum(violaciones_CVaR_p)
    total_VaR_h = sum(violaciones_VaR_h)
    total_CVaR_h = sum(violaciones_CVaR_h)
    
    porcentaje_VaR_p = (total_VaR_p / (len(retornos) - ventana))*100
    porcentaje_CVaR_p= (total_CVaR_p / (len(retornos) - ventana))*100
    porcentaje_VaR_h = (total_VaR_h / (len(retornos) - ventana))*100
    porcentaje_CVaR_h= (total_CVaR_h / (len(retornos) - ventana))*100
    
    return total_VaR_p, porcentaje_VaR_p, total_CVaR_p, porcentaje_CVaR_p, total_VaR_h, porcentaje_VaR_h, total_CVaR_h, porcentaje_CVaR_h


def calcular_var_volatilidad_movil(serie_rendimientos, alphas=[0.05, 0.01], window=252): 
    # Calculamos volatilidad móvil 
    std_roll = serie_rendimientos.rolling(window).std() 
    resultados = pd.DataFrame(index=serie_rendimientos.index) 
    for alpha in alphas: 
        q_alpha = norm.ppf(alpha) 
        resultados[f'VaR Vol Móvil ({alpha})'] = q_alpha * std_roll 
    return resultados.dropna() 
