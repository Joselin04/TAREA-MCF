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

   #print(f"Media: {media_bimbo},  {stdv}, Sesgo:{sesgo_bimbo}")
   #print(media_bimbo,curtosis_bimbo,sesgo_bimbo,desv_std_bimbo)
   #print(media, curtosis, sesgo, stdv)

#Calculamos los grados de libertad para usar una distribucion t-student en nuestros redimientos diarios.
grados_lib_bimbo = len(df_bimbo_rend)-1

metricas = ['Media','Curtosis', 'Sesgo','Desviación estándar']
metrica_seleccionada = st.selectbox('Métricas de Bimbo',metricas)
if metrica_seleccionada == "Media":
    st.metric("Media",f"{media:.4%}")
elif metrica_seleccionada == "Curtosis":
    st.metric("Curtosis",f"{curtosis:.4}")
elif metrica_seleccionada == "Sesgo":
    st.metric("Sesgo",f"{sesgo:.4}")
elif metrica_seleccionada == "Desviación estándar":
    st.metric("Desviación estándar",f"{stdv:.4}")


alphas = [0.95, 0.975, 0.99]

#VaR parametrico
VaR_parametricos = {}
for p in alphas:
    pVaR_n, pVaR_t = MCF.VaR_parametrico(p, media, stdv, grados_lib_bimbo)
    VaR_parametricos[p] = {"VaR Normal": pVaR_n, "VaR t-Student": pVaR_t}

print(VaR_parametricos)

print("\nVaR bajo una aproximacion parametrica: ")
for alpha, valores in VaR_parametricos.items():
    #pvar_n = float(valores["VaR Normal"].item())  # Convertir a numero
    #pvar_t = float(valores["VaR t-Student"].item())
    print(f"Alpha: {alpha} | VaR Normal: {pVaR_n:.6f} | VaR t-Student: {pVaR_t:.6f}")

#VaR monte carlo
VaR_MonteCarlo = {}
for m in alphas:
    MCVaR_n, MCVaR_t = MCF.VaR_MonteCarlo(m, media_bimbo, desv_std_bimbo, grados_lib_bimbo)
    VaR_MonteCarlo[m] = {"VaR Normal": MCVaR_n, "VaR t-Student": MCVaR_t}

print("\nVaR bajo una aproximacion Monte Carlo: ")
for alpha, valores in VaR_MonteCarlo.items():
    MCvar_n = float(valores["VaR Normal"])  # Convertir a numero
    MCvar_t = float(valores["VaR t-Student"])
    print(f"Alpha: {alpha} | VaR Normal: {MCvar_n:.6f} | VaR t-Student: {MCvar_t:.6f}")

#VaR historico
VaR_historicos = {}
for h in alphas:
    hVaR = (MCF.VaR_historico(df_bimbo_rend, h))
    hvar = float(hVaR)
    VaR_historicos[h] = {"VaR historico":hvar}

print("\nVaR bajo una aproximacion historica: ")
for alpha, valores in VaR_historicos.items():
    #hvar = float(valores["VaR historico"].iloc[0])
    print(f"Alpha: {alpha} | VaR historico: {hvar:.6f}")    


#CVaR historico 
CVaR_historico = {}
print("\nCVaR bajo una aproximacion historica: ")
for alpha, valores in VaR_historicos.items():
    CVaR_h = MCF.CVaR(df_bimbo_rend, valores["VaR historico"])
    CVaR_historico[alpha] = {"CVaR historico":CVaR_h}
    print(f"Alpha: {alpha} | CVaR historico: {CVaR_h:.6f}")

   #print(f"CVaR hist es: {CVaR_historico}")

#CVaR parametrico para una dist normal y t-student
CVaR_parametrico = {}
print("\nCVaR bajo una aproximacion parametrica: ")
for alpha, valores in VaR_parametricos.items():
    CVaR_p_n = MCF.CVaR(df_bimbo_rend, valores["VaR Normal"])
    CVaR_p_t = MCF.CVaR(df_bimbo_rend, valores["VaR t-Student"])
    CVaR_parametrico[alpha] = {"CVaR normal":CVaR_p_n, "CVaR t-student":CVaR_p_t}
    print(f"Alpha: {alpha} | CVaR Normal: {CVaR_p_n:.6f} | CVaR t-Student: {CVaR_p_t:.6f}")

#CVaR Monte Carlo para una dist normal y t-student
CVaR_Monte_Carlo = {}
print("\nCVaR bajo una aproximacion Monte Carlo: ")
for alpha, valores in VaR_MonteCarlo.items():
    CVaR_MC_n = MCF.CVaR(df_bimbo_rend, valores["VaR Normal"])
    CVaR_MC_t = MCF.CVaR(df_bimbo_rend, valores["VaR t-Student"])
    CVaR_Monte_Carlo[alpha] = {"CVaR normal":CVaR_MC_n, "CVaR t-student":CVaR_MC_t}
    print(f"Alpha: {alpha} | CVaR Normal: {CVaR_MC_n:.6f} | CVaR t-Student: {CVaR_MC_t:.6f}")


aproximaciones=['Paramétrico','Histórico','Monte Carlo']
medidas = ['Value-At-Risk (VaR)','Expected Shortfall (CVaR)']
aprox_seleccionado = st.selectbox('Tipo de aproximación:', aproximaciones)
if aprox_seleccionado == "Paramétrico":
    medidas_seleccionada =  st.selectbox('Medida de riesgo:', medidas)  
    if medidas_seleccionada == "Value-At-Risk (VaR)":
        st.table(VaR_parametricos)
    elif medidas_seleccionada == "Expected Shortfall (CVaR)":
        st.table(CVaR_parametrico)
elif aprox_seleccionado == "Histórico":
    medidas_seleccionada =  st.selectbox('Medida de riesgo:', medidas)  
    if medidas_seleccionada == "Value-At-Risk (VaR)":
        st.table(VaR_historicos)
    elif medidas_seleccionada == "Expected Shortfall (CVaR)":
        st.table(CVaR_historico)    
elif aprox_seleccionado == "Monte Carlo":
    medidas_seleccionada =  st.selectbox('Medida de riesgo:', medidas)  
    if medidas_seleccionada == "Value-At-Risk (VaR)":
        st.table(VaR_MonteCarlo)
    elif medidas_seleccionada == "Expected Shortfall (CVaR)":
        st.table(CVaR_Monte_Carlo)

#Rolling Window
rolling_mean = df_bimbo_rend.rolling(window=252).mean()
rolling_std = df_bimbo_rend.rolling(window=252).std()
rolling_df = df_bimbo_rend.rolling(window=252)

#VaR parametrico rolling window
VaR_roll_p = {}
for a in alphas:
    VaR_roll_pn = MCF.VaR_rolling_p(a, rolling_mean, rolling_std)
    VaR_roll_p [a] = {"VaR Parametrico Rolling":VaR_roll_pn}

#VaR historico rolling window
VaR_roll_h = {}
for a in alphas:
    VaR_roll_hn  = MCF.VaR_rolling_his(rolling_df,a)
    VaR_roll_h [a] = {"VaR Historico Rolling":VaR_roll_hn}


# Crear un DataFrame para el VaR Paramétrico
resultados = {}
for alpha, valores in VaR_roll_p.items():
    resultados[alpha] = valores["VaR Parametrico Rolling"].squeeze()
parametrico_df = pd.DataFrame(resultados, index=df_bimbo_rend.index)

    #Agregar fechas al DF
fechas = df_bimbo_rend.index  
var_dataframe = pd.DataFrame(resultados, index=fechas)

var_dataframe.columns = [f"{alpha} VaR P Rolling" for alpha in var_dataframe.columns]
print(var_dataframe)

#CVaR
CVaR_df_p = parametrico_df.applymap(lambda x: MCF.CVaR_rolling(parametrico_df, x))

# Crear un DataFrame para el VaR Histórico
historico_resultados = {}
for alpha, valores in VaR_roll_h.items():
    historico_resultados[alpha] = valores["VaR Historico Rolling"].squeeze()
historico_df = pd.DataFrame(historico_resultados, index=df_bimbo_rend.index)

    #historico con fechas
fechas = df_bimbo_rend.index  
VaR_h_df = pd.DataFrame(historico_resultados, index=fechas)

VaR_h_df.columns = [f"{alpha} VaR H Rolling" for alpha in VaR_h_df.columns]

#CVaR
CVaR_df_h = historico_df.applymap(lambda x: MCF.CVaR_rolling(historico_df, x))


plt.figure(figsize=(14, 7))

# Graficar los rendimientos diarios
plt.plot(df_bimbo_rend.index, df_bimbo_rend * 100, label='Daily Returns (%)', color='blue', alpha=0.5)

# Graficar los VaR para cada alpha
plt.plot(var_dataframe.index, var_dataframe['0.95 VaR P Rolling'], label='95% Rolling VaR Parametrico', color='red')
plt.plot(var_dataframe.index, var_dataframe['0.975 VaR P Rolling'], label='97.5% Rolling VaR Parametrico', color='orange')
plt.plot(var_dataframe.index, var_dataframe['0.99 VaR P Rolling'], label='99% Rolling VaR Parametrico', color='green')

plt.plot(VaR_h_df.index, VaR_h_df['0.95 VaR H Rolling'], label='95% Rolling VaR Historico', color='purple')
plt.plot(VaR_h_df.index, VaR_h_df['0.975 VaR H Rolling'], label='97.5% Rolling VaR Historico', color='blue')
plt.plot(VaR_h_df.index, VaR_h_df['0.99 VaR H Rolling'], label='99% Rolling VaR Historico', color='yellow')



# Títulos 
plt.title('Daily Returns and Rolling VaR for Multiple Alphas')
plt.xlabel('Date')
plt.ylabel('Values (%)')

# Agregar la leyenda
plt.legend()

# Ajustar diseño y mostrar la gráfica
plt.tight_layout()
plt.savefig("VAR.png", dpi=300)  # Guarda la figura como archivo PNG

#Graficar CVaR 
plt.figure(figsize=(14, 7))

# Graficar los rendimientos diarios
plt.plot(df_bimbo_rend.index, df_bimbo_rend * 100, label='Daily Returns (%)', color='blue', alpha=0.5)

# Graficar los VaR para cada alpha
plt.plot(CVaR_df_p.index, CVaR_df_p[0.95], label='95% Rolling CVaR Parametrico', color='red')
plt.plot(CVaR_df_p.index, CVaR_df_p[0.975], label='97.5% Rolling CVaR Parametrico', color='orange')
plt.plot(CVaR_df_p.index, CVaR_df_p[0.99], label='99% Rolling CVaR Parametrico', color='green')

plt.plot(CVaR_df_h.index,CVaR_df_h[0.95], label='95% Rolling CVaR Historico', color='purple')
plt.plot(CVaR_df_h.index,CVaR_df_h[0.975], label='97.5% Rolling CVaR Historico', color='blue')
plt.plot(CVaR_df_h.index,CVaR_df_h[0.99], label='99% Rolling CVaR Historico', color='yellow')


# Títulos 
plt.title('Daily Returns and Rolling VaR for Multiple Alphas')
plt.xlabel('Date')
plt.ylabel('Values (%)')

# Agregar la leyenda
plt.legend()

# Ajustar diseño y mostrar la gráfica
plt.tight_layout()
plt.savefig("CVAR.png", dpi=300)  # Guarda la figura como archivo PNG
