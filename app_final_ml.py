import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Predicción Saber 11 - SGD & LogReg", layout="wide")
st.title("🎯 Predictor de Desempeño y Puntaje Global - Saber 11")
st.write("Esta aplicación evalúa las características del estudiante para estimar su puntaje global (SGD Regressor) y clasificar su nivel de desempeño (Regresión Logística).")

# 1. Cargar ambos modelos serializados
@st.cache_resource
def load_models():
    sgd_model = joblib.load('sgd_regressor_model.joblib')
    logreg_model = joblib.load('modelo_logistico_saber11.joblib')  # Nombre de tu archivo de Regresión Logística
    return sgd_model, logreg_model

try:
    sgd_model, logreg_model = load_models()
except Exception as e:
    st.error("Error al cargar los modelos (.joblib). Verifica que ambos archivos estén en el directorio de la aplicación.")
    st.stop()

# 2. Cargar las columnas de características para mapear de forma dinámica
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
    st.error("No se pudo cargar el archivo de datos para mapear las columnas. Asegúrate de que 'microdatos_saber_11_procesados.csv' esté en el mismo directorio.")
    st.stop()

# Variables con configuración manual/específica
CUSTOM_KEYS = ["estu_genero", "cole_area_ubicacion", "estu_privado_libertad"]

# Agrupar el resto de columnas de manera dinámica
categories_map = {}
for col in feature_cols:
    if '_' in col:
        # Evitar mapear automáticamente las variables personalizadas
        if any(col.startswith(ck) for ck in CUSTOM_KEYS):
            continue
            
        parts = col.split('_')
        if col.startswith("estu_depto_reside"):
            prefix, option = "estu_depto_reside", col.replace("estu_depto_reside_", "")
        elif col.startswith("cole_caracter"):
            prefix, option = "cole_caracter", col.replace("cole_caracter_", "")
        elif col.startswith("cole_jornada"):
            prefix, option = "cole_jornada", col.replace("cole_jornada_", "")
        else:
            prefix = "_".join(parts[:-1])
            option = parts[-1]
        
        if prefix not in categories_map:
            categories_map[prefix] = []
        categories_map[prefix].append(option)

# Añadir valor por defecto para el resto de variables categóricas dinámicas
for key in categories_map:
    categories_map[key] = sorted(list(set(categories_map[key])))
    categories_map[key].insert(0, "Otro / No especificado")

st.header("📋 Formulario de Entrada de Datos")
user_inputs = {}

# Layout en dos columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("Información del Estudiante")
    # 1. Género: opciones 'f' y 'm'
    user_inputs["estu_genero"] = st.selectbox("Género", ["f", "m"])
    
    # Departamento de Residencia
    if "estu_depto_reside" in categories_map:
        user_inputs["estu_depto_reside"] = st.selectbox("Departamento de Residencia", categories_map["estu_depto_reside"])
    
    # 2. Privado de la Libertad: opciones 'no' y 'si'
    user_inputs["estu_privado_libertad"] = st.selectbox("¿Estudiante Privado de la Libertad?", ["no", "si"])

with col2:
    st.subheader("Información del Colegio")
    if "cole_caracter" in categories_map:
        user_inputs["cole_caracter"] = st.selectbox("Carácter del Colegio", categories_map["cole_caracter"])
    
    # 3. Área de Ubicación: opciones 'rural' y 'urbano'
    user_inputs["cole_area_ubicacion"] = st.selectbox("Área de Ubicación del Colegio", ["rural", "urbano"])
    
    if "cole_jornada" in categories_map:
        user_inputs["cole_jornada"] = st.selectbox("Jornada del Colegio", categories_map["cole_jornada"])

# Procesar el resto de variables
st.subheader("Otras características del entorno")
remaining_keys = [k for k in categories_map.keys() if k not in ["estu_depto_reside", "cole_caracter", "cole_jornada"]]

if remaining_keys:
    cols_rem = st.columns(3)
    for idx, key in enumerate(remaining_keys):
        with cols_rem[idx % 3]:
            label = key.replace("_", " ").title()
            user_inputs[key] = st.selectbox(label, categories_map[key])

# 3. Realizar predicción con ambos modelos
st.markdown("---")
if st.button("🚀 Evaluar Estudiante", use_container_width=True):
    # Vector de 0s para todas las características
    input_vector = np.zeros((1, len(feature_cols)))
    
    # A. Mapeo específico de los selectores requeridos
    if user_inputs["estu_genero"] == "m" and "estu_genero_m" in feature_cols:
        input_vector[0, feature_cols.index("estu_genero_m")] = 1

    if user_inputs["cole_area_ubicacion"] == "urbano" and "cole_area_ubicacion_urbano" in feature_cols:
        input_vector[0, feature_cols.index("cole_area_ubicacion_urbano")] = 1

    if user_inputs["estu_privado_libertad"] == "si" and "estu_privado_libertad_s" in feature_cols:
        input_vector[0, feature_cols.index("estu_privado_libertad_s")] = 1

    # B. Mapeo del resto de características
    for key, selected_value in user_inputs.items():
        if key not in CUSTOM_KEYS and selected_value != "Otro / No especificado":
            expected_col_name = f"{key}_{selected_value}"
            if expected_col_name in feature_cols:
                col_index = feature_cols.index(expected_col_name)
                input_vector[0, col_index] = 1

    # 1. Predicción del Puntaje Global (SGD Regressor)
    score_prediction = sgd_model.predict(input_vector)[0]
    
    # 2. Predicción de Clasificación de Desempeño (Regresión Logística)
    class_prediction = logreg_model.predict(input_vector)[0]
    
    # Probabilidad estimada de pertenecer a Alto Desempeño (Clase 1)
    if hasattr(logreg_model, "predict_proba"):
        prob_alto = logreg_model.predict_proba(input_vector)[0][1] * 100
    else:
        prob_alto = None

    # Mostrar Resultados
    st.balloons()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(
            f"""
            <div style='text-align: center; border: 2px solid #4CAF50; padding: 15px; border-radius: 10px;'>
                <h3>📊 Puntaje Global Estimado (SGD)</h3>
                <h1 style='color: #2E7D32;'>{score_prediction:.2f} / 500</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with res_col2:
        if class_prediction == 1:
            badge_color = "#2E7D32"
            texto_desempeno = "Alto Desempeño 📈"
        else:
            badge_color = "#C62828"
            texto_desempeno = "Bajo Desempeño 📉"
            
        prob_str = f"<p style='margin: 0; font-size: 14px;'>Probabilidad de Alto Desempeño: <b>{prob_alto:.1f}%</b></p>" if prob_alto is not None else ""
        
        st.markdown(
            f"""
            <div style='text-align: center; border: 2px solid {badge_color}; padding: 15px; border-radius: 10px;'>
                <h3>🏷️ Clasificación de Desempeño (LogReg)</h3>
                <h1 style='color: {badge_color};'>{texto_desempeno}</h1>
                {prob_str}
            </div>
            """, 
            unsafe_allow_html=True
        )