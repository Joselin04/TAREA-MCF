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

media_bimbo = np.mean(df_bimbo_rend)
curtosis_bimbo = kurtosis(df_bimbo_rend)
sesgo_bimbo = skew(df_bimbo_rend)
desv_std_bimbo = np.std(df_bimbo_rend) 
grados_lib_bimbo = len(df_bimbo_rend)-1

alphas = [0.95, 0.975, 0.99]

'''st.metric("Rendimiento Medio Diario", f"{media_bimbo:.4%}")
st.metric("Desviación Estándar", f"{desv_std_bimbo:.4f}")
st.metric("Curtosis", f"{curtosis_bimbo:.4f}")
st.metric("Sesgo", f"{sesgo_bimbo:.4f}")'''


#for a in alphas:
 #   VaR_n = (norm.ppf(1-a, media_bimbo,desv_std_bimbo))
  #  print (VaR_n)

#VaR parametrico para α = 0.95, 0.975, y 0.99, distribucion normal
VaR_n_95 = (norm.ppf(1-0.95, media_bimbo,desv_std_bimbo))
VaR_n_975 = (norm.ppf(1-0.975, media_bimbo,desv_std_bimbo))
VaR_n_99 = (norm.ppf(1-0.99, media_bimbo,desv_std_bimbo))

#VaR parametrico para α = 0.95, 0.975, y 0.99, distribucion t-student
VaR_t_95 = (stats.t.ppf(1-0.95, grados_lib_bimbo, media_bimbo, desv_std_bimbo))
VaR_t_975 = (stats.t.ppf(1-0.975, grados_lib_bimbo, media_bimbo, desv_std_bimbo))
VaR_t_99 = (stats.t.ppf(1-0.99, grados_lib_bimbo, media_bimbo, desv_std_bimbo))

#VaR historico para α = 0.95, 0.975, y 0.99
hVaR_n_95 = (df_bimbo_rend.quantile(0.05))
hVaR_n_975 = (df_bimbo_rend.quantile(0.025))
hVaR_n_99 = (df_bimbo_rend.quantile(0.01))

#VaR MonteCarlo para α = 0.95, 0.975, y 0.99 
simulaciones = 100000
sim_returns_n = np.random.normal(media_bimbo,desv_std_bimbo,simulaciones)
sim_returns_t = stats.t.rvs(grados_lib_bimbo, media_bimbo,desv_std_bimbo,simulaciones)

#VaR MonteCarlo para distribucion Normal
MCVaR_n_95 = np.percentile(sim_returns_n,5)
MCVaR_n_975 = np.percentile(sim_returns_n,2.5)
MCVaR_n_99 = np.percentile(sim_returns_n,1)

#VaR MonteCarlo para distribucion t-student
MCVaR_t_95 = np.percentile(sim_returns_t,5)
MCVaR_t_975 = np.percentile(sim_returns_t,2.5)
MCVaR_t_99 = np.percentile(sim_returns_t,1)

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

#plt.figure(figsize=(14, 7))
#plt.plot(df_bimbo_rend.index, df_bimbo_rend * 100, label='Daily Returns (%)', color='blue', alpha=0.5)
#plt.plot(vaR_rolling_df.index, vaR_rolling_df['95% VaR Rolling'], label='95% Rolling VaR', color='red')
#plt.title('Daily Returns and 95% Rolling VaR')
#plt.xlabel('Date')
#plt.ylabel('Values (%)')
#plt.legend()
#plt.tight_layout()
#plt.show()

valores = vaR_rolling_df.iloc[[250,251,252,253,254,255]]
print(valores)
#print(vaR_rolling_df.tail())
print(desv_std_bimbo)

