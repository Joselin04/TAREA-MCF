import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm
import Funciones_MCF as MCF

col1, col2 = st.columns([3, 1])
with col1:
    st.title('Grupo Bimbo, S.A.B. de C.V. (BIMBOA.MX)')
    st.subheader('Rendimientos diarios de 2010 a la actualidad')

    st.subheader(f'Métricas de BIMBO:')

with col2:
    url_imagen = "https://media.licdn.com/dms/image/v2/C4E12AQFukyhc9QjjBA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1623952516208?e=2147483647&v=beta&t=PdCxO5tMQqPIvdbKgtv3VUEINbyXujyEcjF9UTSHRZE"
    st.image(url_imagen, width=200)

# Descargar datos históricos de BIMBOA.MX
df_bimbo_precios = MCF.obtener_datos("BIMBOA.MX")
df_bimbo_rend = MCF.calcular_rendimientos(df_bimbo_precios)

#Calculamos la media, curtosis, sesgo y desviacion estandar de los rendimientos diarios de nuestro activo. 
media_bimbo = np.mean(df_bimbo_rend)
media =  float(media_bimbo)

curtosis_bimbo = kurtosis(df_bimbo_rend)
curtosis = float(curtosis_bimbo)

sesgo_bimbo = skew(df_bimbo_rend)
sesgo =  float(sesgo_bimbo)

desv_std_bimbo = np.std(df_bimbo_rend)
stdv = float(desv_std_bimbo)

#Calculamos los grados de libertad para usar una distribucion t-student en nuestros redimientos diarios.
grados_lib_bimbo = len(df_bimbo_rend)-1

#print(media, curtosis, stdv, grados_lib_bimbo)

alphas = [0.95, 0.975, 0.99]

VaR_historicos = {}
for h in alphas:
    hVaR = MCF.VaR_historico(df_bimbo_rend, h)
    VaR_historicos[h] = {"VaR historico": hVaR}

print("\nVaR bajo una aproximacion historica: ")
for alpha, valores in VaR_historicos.items():
    hVaR = float(valores["VaR historico"].iloc[0])
    print(f"Alpha: {alpha} | VaR historico: {hVaR:.6f}")
    

#CVaR historico para una dist normal 
print("\nCVaR bajo una aproximacion historica: ")
for alpha, valores in VaR_historicos.items():
    hVaR = float(valores["VaR historico"].iloc[0])
    CVaR_h = MCF.CVaR_Historico(df_bimbo_rend, valores["VaR historico"].iloc[0])
    print(f"Alpha: {alpha} | CVaR historico: {CVaR_h:.6f}")

CVAR_95 = MCF.CVaR_Historico(df_bimbo_rend,-0.028035)
print(CVAR_95)

CVaR_historicos = {}
Val_HVaR = []
for a in alphas:
    hVaR_n = float(VaR_historicos[a]["VaR historico"])
    Val_HVaR.append(hVaR_n)

print(Val_HVaR)
print(Val_HVaR[1])



for v in VaR_historicos:
    hCVaR = MCF.CVaR_Historico(df_bimbo_rend, v)
    CVaR_historicos[v] = {"CVaR historico": hCVaR}

print("\n CVaR bajo una aproximacion historica: ")
#for alpha, valores in VaR_historicos.items():
#  print(f"Alpha: {alpha} | CVaR historico: {Val_HVaR:.6f}")