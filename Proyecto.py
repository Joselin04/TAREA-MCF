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

print(f"Media: {media_bimbo},  {stdv}, Sesgo:{sesgo_bimbo}")
print(media_bimbo,curtosis_bimbo,sesgo_bimbo,desv_std_bimbo)
print(media, curtosis, sesgo, stdv)

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
    #pvar_n = float(valores["VaR Normal"].item())  # Convertir a numero
    #pvar_t = float(valores["VaR t-Student"].item())
    print(f"Alpha: {alpha} | VaR Normal: {pVaR_n:.6f} | VaR t-Student: {pVaR_t:.6f}")


VaR_MonteCarlo = {}
for m in alphas:
    MCVaR_n, MCVaR_t = MCF.VaR_MonteCarlo(m, media_bimbo, desv_std_bimbo, grados_lib_bimbo)
    VaR_MonteCarlo[m] = {"VaR Normal": MCVaR_n, "VaR t-Student": MCVaR_t}

print("\nVaR bajo una aproximacion Monte Carlo: ")
for alpha, valores in VaR_MonteCarlo.items():
    MCvar_n = float(valores["VaR Normal"])  # Convertir a numero
    MCvar_t = float(valores["VaR t-Student"])
    print(f"Alpha: {alpha} | VaR Normal: {MCvar_n:.6f} | VaR t-Student: {MCvar_t:.6f}")


VaR_historicos = {}
for h in alphas:
    hVaR = MCF.VaR_historico(df_bimbo_rend, h)
    VaR_historicos[h] = {"VaR historico": hVaR}

print("\nVaR bajo una aproximacion historica: ")
for alpha, valores in VaR_historicos.items():
    hVaR = float(valores["VaR historico"].iloc[0])
    print(f"Alpha: {alpha} | VaR historico: {hVaR:.6f}")


#CVaR historico 
print("\nCVaR bajo una aproximacion historica: ")
for alpha, valores in VaR_historicos.items():
    hVaR = float(valores["VaR historico"].iloc[0])
    CVaR_h = MCF.CVaR(df_bimbo_rend, valores["VaR historico"].iloc[0])
    print(f"Alpha: {alpha} | CVaR historico: {CVaR_h:.6f}")

#CVaR parametrico para una dist normal y t-student
print("\nCVaR bajo una aproximacion parametrica: ")
for alpha, valores in VaR_parametricos.items():
    pVaR_n = float(valores["VaR Normal"])
    pVaR_t = float(valores["VaR t-Student"])
    CVaR_p_n = MCF.CVaR(df_bimbo_rend, valores["VaR Normal"])
    CVaR_p_t = MCF.CVaR(df_bimbo_rend, valores["VaR t-Student"])
    print(f"Alpha: {alpha} | CVaR Normal: {CVaR_p_n:.6f} | CVaR t-Student: {CVaR_p_t:.6f}")

#CVaR Monte Carlo para una dist normal y t-student
print("\nCVaR bajo una aproximacion Monte Carlo: ")
for alpha, valores in VaR_MonteCarlo.items():
    MCvar_n = float(valores["VaR Normal"])
    MCvar_t = float(valores["VaR t-Student"])
    CVaR_MC_n = MCF.CVaR(df_bimbo_rend, valores["VaR Normal"])
    CVaR_MC_t = MCF.CVaR(df_bimbo_rend, valores["VaR t-Student"])
    print(f"Alpha: {alpha} | CVaR Normal: {CVaR_MC_n:.6f} | CVaR t-Student: {CVaR_MC_t:.6f}")



#Rolling Window
rolling_mean = df_bimbo_rend.rolling(window=252).mean()
rolling_std = df_bimbo_rend.rolling(window=252).std()


VaR_roll_p = {}
for a in alphas:
    VaR_roll_n = MCF.VaR_rolling(a, rolling_mean, rolling_std)
    VaR_roll_p [a] = {"VaR Parametrico Rolling":VaR_roll_n}


#Dataframe que visualiza los datos conforme a la fecha
resultados = {}
for alpha, valores in VaR_roll_p.items():
    resultados[alpha] = valores["VaR Parametrico Rolling"].squeeze()

fechas = df_bimbo_rend.index  
var_dataframe = pd.DataFrame(resultados, index=fechas)

var_dataframe.columns = [f"{alpha} VaR Rolling" for alpha in var_dataframe.columns]

print(var_dataframe)

plt.figure(figsize=(14, 7))

# Graficar los rendimientos diarios
plt.plot(df_bimbo_rend.index, df_bimbo_rend * 100, label='Daily Returns (%)', color='blue', alpha=0.5)

# Graficar los VaR para cada alpha
plt.plot(var_dataframe.index, var_dataframe['0.95 VaR Rolling'], label='95% Rolling VaR', color='red')
plt.plot(var_dataframe.index, var_dataframe['0.975 VaR Rolling'], label='97.5% Rolling VaR', color='orange')
plt.plot(var_dataframe.index, var_dataframe['0.99 VaR Rolling'], label='99% Rolling VaR', color='green')


# Títulos 
plt.title('Daily Returns and Rolling VaR for Multiple Alphas')
plt.xlabel('Date')
plt.ylabel('Values (%)')

# Agregar la leyenda
plt.legend()

# Ajustar diseño y mostrar la gráfica
plt.tight_layout()
plt.savefig("VAR_p.png", dpi=300)  # Guarda la figura como archivo PNG

