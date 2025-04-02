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
curtosis = float(curtosis_bimbo.item())

sesgo_bimbo = skew(df_bimbo_rend)
sesgo =  float(sesgo_bimbo.item())

desv_std_bimbo = np.std(df_bimbo_rend)
stdv = float(desv_std_bimbo.iloc[0])

#Calculamos los grados de libertad para usar una distribucion t-student en nuestros redimientos diarios.
grados_lib_bimbo = len(df_bimbo_rend)-1


st.metric("Rendimiento Medio Diario", f"{media_bimbo:.4%}")
st.metric("Desviación Estándar", f"{stdv:.4f}")
st.metric("Curtosis", f"{curtosis:.4f}")
st.metric("Sesgo", f"{sesgo:.4f}")

alphas = [0.95, 0.975, 0.99]

VaR_parametricos = {}
for p in alphas:
    pVaR_n, pVaR_t = MCF.VaR_parametrico(p, media, stdv, grados_lib_bimbo)
    VaR_parametricos[p] = {"VaR Normal": pVaR_n, "VaR t-Student": pVaR_t}

print("\nVaR bajo una aproximacion parametrica: ")
for alpha, valores in VaR_parametricos.items():
    print(f"Alpha: {alpha} | VaR Normal: {pVaR_n:.6f} | VaR t-Student: {pVaR_t:.6f}")

VaR_MonteCarlo = {}
for m in alphas:
    MCVaR_n, MCVaR_t = MCF.VaR_MonteCarlo(m, media, stdv, grados_lib_bimbo)
    VaR_MonteCarlo[m] = {"VaR Normal": MCVaR_n, "VaR t-Student": MCVaR_t}

print("\nVaR bajo una aproximacion Monte Carlo: ")
for alpha, valores in VaR_MonteCarlo.items():
    print(f"Alpha: {alpha} | VaR Normal: {MCVaR_n:.6f} | VaR t-Student: {MCVaR_t:.6f}")

VaR_historicos = {}
for h in alphas:
    hVaR = MCF.VaR_historico(df_bimbo_rend, h)
    VaR_historicos[h] = {"VaR historico": hVaR}

print("\nVaR bajo una aproximacion historica: ")
for alpha, valores in VaR_historicos.items():
    hVaR = float(valores["VaR historico"].iloc[0])
    print(f"Alpha: {alpha} | VaR historico: {hVaR:.6f}")

#CVaR historico para una dist normal
CVaR_95_his = np.mean(df_bimbo_rend[df_bimbo_rend <= VaR_n_95])
CVaR_99_his = np.mean(df_bimbo_rend[df_bimbo_rend <= VaR_n_99])

#CVaR parametrico para una dist normal
CVaR_95_par = np.mean(df_bimbo_rend[df_bimbo_rend <= hVaR_n_95])
CVaR_99_par = np.mean(df_bimbo_rend[df_bimbo_rend <= hVaR_n_99])

#Rolling Window
rolling_mean = df_bimbo_rend.rolling(window=252).mean()
rolling_std = df_bimbo_rend.rolling(window=252).std()

VaR_95_rolling = norm.ppf(1-0.95, rolling_mean, rolling_std)
VaR_95_rolling_percent = (VaR_95_rolling * 100).round(4)

vaR_rolling_df = pd.DataFrame({'Date': df_bimbo_rend.index, '95% VaR Rolling': VaR_95_rolling_percent.squeeze()})
vaR_rolling_df.set_index('Date', inplace=True)

plt.figure(figsize=(14, 7))
plt.plot(df_bimbo_rend.index, df_bimbo_rend * 100, label='Daily Returns (%)', color='blue', alpha=0.5)
plt.plot(vaR_rolling_df.index, vaR_rolling_df['95% VaR Rolling'], label='95% Rolling VaR', color='red')
plt.title('Daily Returns and 95% Rolling VaR')
plt.xlabel('Date')
plt.ylabel('Values (%)')
plt.legend()
plt.tight_layout()
plt.show()

