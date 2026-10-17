import streamlit as st
import pandas as pd

from tabs import entender, limpiar, descriptiva, predictiva, prescriptiva, decisiones

# ----------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------
st.set_page_config(
    page_title="PISA México",
    page_icon="📚",
    layout="wide"
)

# ----------------------------------
# TÍTULO Y CONTEXTO
# ----------------------------------
st.title("📚 PISA México: riesgo en matemáticas")

st.markdown("""
**Contexto:** alumnos mexicanos de 15 años evaluados en PISA 2022 (OCDE).

**Pregunta guía:** ¿Qué alumnos tienen mayor riesgo de no alcanzar el nivel
mínimo de competencia en matemáticas y qué apoyo deberíamos ofrecer a cada grupo?
""")

# ----------------------------------
# CARGA DE DATOS
# ----------------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("data/pisa_2022_mexico.csv")

df = cargar_datos()

# Dataset limpio (decisiones documentadas en la pestaña 2)
df_limpio, _ = limpiar.preparar_datos(df)

# ----------------------------------
# PESTAÑAS
# ----------------------------------
(
    tab_entender,
    tab_limpiar,
    tab_descriptiva,
    tab_predictiva,
    tab_prescriptiva,
    tab_decisiones,
) = st.tabs([
    "1\\. Entender",
    "2\\. Limpiar",
    "3\\. Descriptiva",
    "4\\. Predictiva",
    "5\\. Prescriptiva",
    "6\\. Decisiones",
])

with tab_entender:
    entender.mostrar(df)

with tab_limpiar:
    limpiar.mostrar(df)

with tab_descriptiva:
    descriptiva.mostrar(df_limpio)

with tab_predictiva:
    predictiva.mostrar(df_limpio)

with tab_prescriptiva:
    prescriptiva.mostrar(df_limpio)

with tab_decisiones:
    decisiones.mostrar(df_limpio)

st.divider()
st.caption(
    "Fuente: Programme for International Student Assessment (PISA), "
    "Organisation for Economic Co-operation and Development (OECD), Paris. "
    "Microdatos 2022 vía el paquete abierto learningtower."
)
