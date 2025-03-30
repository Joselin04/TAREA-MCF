import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm

st.title("¡Hola, Streamlit desde VS Code! 🎈")
st.write("Esta es una prueba de Streamlit.")

alumnos = pd.Series( ['Mau', 'Mayte', 'Jair', 'Francisco'] )
calificaciones = pd.Series([10.15, 10.9, 10.0, 10.21])

tabla = pd.DataFrame( { 'Alumno': alumnos, 'Calificacion': calificaciones } )

st.table(tabla)

