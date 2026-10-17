import streamlit as st
import pandas as pd
import plotly.express as px


# Rol de cada variable respecto a la pregunta guía
ROLES = {
    "alcanza_nivel2_mat": "🎯 Resultado (sí/no)",
    "math": "Base del resultado (no se usa como predictor)",
    "read": "Solo descriptiva (fuga de información)",
    "science": "Solo descriptiva (fuga de información)",
    "escs": "⭐ Predictor clave: nivel socioeconómico",
    "book": "⭐ Predictor clave: capital cultural",
    "mother_educ": "⭐ Predictor clave: escolaridad de la madre",
    "father_educ": "⭐ Predictor clave: escolaridad del padre",
    "gender": "⭐ Predictor clave: género",
    "computer": "Predictor: recursos en casa",
    "internet": "Predictor: recursos en casa",
    "room": "Predictor: recursos en casa",
    "computer_n": "Predictor: recursos en casa",
    "car": "Predictor: recursos en casa",
    "television": "Predictor: recursos en casa",
    "stu_wgt": "Peso muestral (para cifras poblacionales)",
    "school_id": "Identificador anónimo",
    "student_id": "Identificador anónimo",
    "year": "Constante (2022)",
    "country": "Constante (MEX)",
    "desk": "⚠️ Sin datos",
    "dishwasher": "⚠️ Sin datos",
    "wealth": "⚠️ Sin datos",
}


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------
    st.header("1. Entender los datos")
    st.markdown("""
    ### Pregunta
    **¿Quiénes son los alumnos y qué información tenemos de ellos?**
    """)

    # ----------------------------------
    # DIMENSIONES DEL DATASET
    # ----------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Número de alumnos", f"{len(df):,}")

    with col2:
        st.metric("Número de variables", len(df.columns))

    with col3:
        st.metric("Número de escuelas", f"{df['school_id'].nunique():,}")

    with col4:
        st.metric(
            "Alcanzan el Nivel 2 en matemáticas",
            f"{df['alcanza_nivel2_mat'].mean():.0%}"
        )

    # ----------------------------------
    # VISTA PREVIA
    # ----------------------------------
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(10), width="stretch")

    # ----------------------------------
    # DIAGNÓSTICO: VARIABLES Y TIPOS
    # ----------------------------------
    st.subheader("Variables, tipos de datos y rol en la pregunta guía")

    diagnostico = pd.DataFrame({
        "Variable": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Valores únicos": df.nunique().values,
        "Valores faltantes": df.isnull().sum().values,
        "% faltantes": (df.isnull().mean() * 100).round(1).values,
        "Rol": [ROLES.get(c, "") for c in df.columns],
    })

    st.dataframe(diagnostico, width="stretch", hide_index=True)

    st.caption(
        "Tipo int64/float64 = numérica; object = categórica (texto). "
        "Variables como `television` o `book` son rangos guardados como texto "
        "(p. ej. '3+', '101-200'), por eso se tratan como categóricas."
    )

    # ----------------------------------
    # PROBLEMAS POTENCIALES
    # ----------------------------------
    st.subheader("Problemas potenciales")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Registros duplicados", int(df.duplicated().sum()))

    with col2:
        vacias = [c for c in df.columns if df[c].isnull().all()]
        st.metric("Columnas 100% vacías", len(vacias))

    if vacias:
        st.warning(
            "Columnas sin ningún dato: " + ", ".join(f"`{c}`" for c in vacias)
            + ". Se tratarán en la pestaña 2."
        )

    # ----------------------------------
    # EXPLORADOR DE VARIABLES
    # ----------------------------------
    st.subheader("Explorar una variable")

    opciones = [c for c in df.columns if not df[c].isnull().all()]

    variable = st.selectbox(
        "Selecciona una variable:",
        opciones,
        index=opciones.index("escs"),
        key="variable_entender"
    )

    # VARIABLE NUMÉRICA
    if pd.api.types.is_numeric_dtype(df[variable]):

        st.write("**Tipo:** variable numérica")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                df[variable].describe().round(2),
                width="stretch"
            )

        with col2:
            fig = px.histogram(
                df,
                x=variable,
                nbins=40,
                title=f"Distribución de {variable}",
                labels={variable: variable, "count": "Número de alumnos"},
            )
            fig.update_layout(yaxis_title="Número de alumnos")
            st.plotly_chart(fig)

    # VARIABLE CATEGÓRICA
    else:

        st.write("**Tipo:** variable categórica")

        frecuencias = (
            df[variable]
            .value_counts(dropna=False)
            .reset_index()
        )
        frecuencias.columns = ["Valor", "Frecuencia"]
        frecuencias["Valor"] = frecuencias["Valor"].fillna("(faltante)")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(frecuencias, width="stretch", hide_index=True)

        with col2:
            fig = px.bar(
                frecuencias,
                x="Valor",
                y="Frecuencia",
                title=f"Frecuencia de {variable}",
                labels={"Frecuencia": "Número de alumnos"},
            )
            st.plotly_chart(fig)
