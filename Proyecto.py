import pandas as pd
alumnos = pd.Series( ['Mau', 'Mayte', 'Jair', 'Francisco'] )
calificaciones = pd.Series([10.15, 10.9, 10.0, 10.21])

tabla = pd.DataFrame( { 'Alumno': alumnos, 'Calificacion': calificaciones } )

print (tabla)

