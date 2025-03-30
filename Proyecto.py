import pandas as pd
import streamlit as st

st.title("¡Hola, Streamlit desde VS Code! 🎈")
st.write("Esta es una prueba de Streamlit.")

alumnos = pd.Series( ['Mau', 'Mayte', 'Jair', 'Francisco'] )
calificaciones = pd.Series([10.15, 10.9, 10.0, 10.21])

tabla = pd.DataFrame( { 'Alumno': alumnos, 'Calificacion': calificaciones } )

st.table(tabla)

