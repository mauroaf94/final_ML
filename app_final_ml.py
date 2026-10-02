import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Predicción Saber 11 - SGD", layout="wide")
st.title("🎯 Predictor de Puntaje Global - Saber 11")
st.write("Esta aplicación permite ingresar todas las variables del dataset para estimar el puntaje global utilizando el modelo SGDRegressor entrenado.")

# 1. Cargar el modelo serializado
@st.cache_resource
def load_model():
    return joblib.load('sgd_regressor_model.joblib')

model = load_model()

# 2. Cargar las columnas de características para mapear de forma dinámica
# Leeremos un registro de muestra para obtener los nombres exactos de las columnas
@st.cache_data
def get_feature_structure():
    df_temp = pd.read_csv("microdatos_saber_11_procesados.csv", nrows=1)
    if "percentil_global" in df_temp.columns:
        df_temp = df_temp.drop("percentil_global", axis=1)
    if "punt_global" in df_temp.columns:
        df_temp = df_temp.drop("punt_global", axis=1)
    return list(df_temp.columns)

try:
    feature_cols = get_feature_structure()
except Exception as e:
    # Fallback si el archivo no está en la ruta por defecto
    st.error("No se pudo cargar el archivo de datos para mapear las columnas. Asegúrate de que 'microdatos_saber_11_procesados.csv' esté en el mismo directorio.")
    st.stop()

# Agruparemos las columnas codificadas por su prefijo original para crear selectores intuitivos
categories_map = {}
for col in feature_cols:
    if '_' in col:
        # Buscamos el último grupo o dividimos por el prefijo común
        parts = col.split('_')
        # Agrupaciones lógicas basadas en los nombres comunes del ICFES
        if col.startswith("estu_depto_reside"):
            prefix, option = "estu_depto_reside", col.replace("estu_depto_reside_", "")
        elif col.startswith("cole_caracter"):
            prefix, option = "cole_caracter", col.replace("cole_caracter_", "")
        elif col.startswith("cole_area_ubicacion"):
            prefix, option = "cole_area_ubicacion", col.replace("cole_area_ubicacion_", "")
        elif col.startswith("cole_jornada"):
            prefix, option = "cole_jornada", col.replace("cole_jornada_", "")
        elif col.startswith("estu_genero"):
            prefix, option = "estu_genero", col.replace("estu_genero_", "")
        elif col.startswith("estu_privado_libertad"):
            prefix, option = "estu_privado_libertad", col.replace("estu_privado_libertad_", "")
        else:
            # Cualquier otra variable codificada
            prefix = "_".join(parts[:-1])
            option = parts[-1]
        
        if prefix not in categories_map:
            categories_map[prefix] = []
        categories_map[prefix].append(option)

# Añadimos opciones para representar el valor 'por defecto' o valor base que no tiene columna (el que se eliminó en One-Hot)
for key in categories_map:
    categories_map[key] = sorted(list(set(categories_map[key])))
    categories_map[key].insert(0, "Otro / No especificado")

# Traducimos las variables a nombres legibles para la interfaz de usuario
labels_map = {
    "estu_depto_reside": "Departamento de Residencia",
    "cole_caracter": "Carácter del Colegio",
    "cole_area_ubicacion": "Área de Ubicación del Colegio",
    "cole_jornada": "Jornada del Colegio",
    "estu_genero": "Género",
    "estu_privado_libertad": "¿Estudiante Privado de la Libertad?"
}

st.header("📋 Formulario de Entrada de Datos")
user_inputs = {}

# Dividir el formulario en dos columnas para una mejor presentación visual
col1, col2 = st.columns(2)

with col1:
    st.subheader("Información del Estudiante")
    for key in ["estu_genero", "estu_depto_reside", "estu_privado_libertad"]:
        if key in categories_map:
            label = labels_map.get(key, key.replace("_", " ").title())
            user_inputs[key] = st.selectbox(label, categories_map[key])

with col2:
    st.subheader("Información del Colegio")
    for key in ["cole_caracter", "cole_area_ubicacion", "cole_jornada"]:
        if key in categories_map:
            label = labels_map.get(key, key.replace("_", " ").title())
            user_inputs[key] = st.selectbox(label, categories_map[key])

# Procesar el resto de variables del dataset que no se agruparon de forma predeterminada
st.subheader("Otras características del entorno")
remaining_keys = [k for k in categories_map.keys() if k not in ["estu_genero", "estu_depto_reside", "estu_privado_libertad", "cole_caracter", "cole_area_ubicacion", "cole_jornada"]]

if remaining_keys:
    cols_rem = st.columns(3)
    for idx, key in enumerate(remaining_keys):
        with cols_rem[idx % 3]:
            label = key.replace("_", " ").title()
            user_inputs[key] = st.selectbox(label, categories_map[key])

# 3. Realizar predicción
st.markdown("---")
if st.button("🚀 Estimar Puntaje Global", use_container_width=True):
    # Inicializar el vector de entrada con ceros para las 111 características
    input_vector = np.zeros((1, len(feature_cols)))
    
    # Llenar el vector con un 1 en la posición donde coincida la selección del usuario
    for key, selected_value in user_inputs.items():
        if selected_value != "Otro / No especificado":
            # Reconstruimos el nombre exacto de la columna codificada
            expected_col_name = f"{key}_{selected_value}"
            if expected_col_name in feature_cols:
                col_index = feature_cols.index(expected_col_name)
                input_vector[0, col_index] = 1

    # Generar predicción con el modelo SGDRegressor
    prediction = model.predict(input_vector)[0]
    
    # Mostrar el resultado final
    st.balloons()
    st.markdown(f"<div style='text-align: center;'><h2>📊 Puntaje Global Estimado:</h2><h1 style='color: #FF4B4B;'>{prediction:.2f} / 500</h1></div>", unsafe_allow_html=True)