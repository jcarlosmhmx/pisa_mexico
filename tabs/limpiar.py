import streamlit as st
import pandas as pd
import plotly.express as px


# Columnas categóricas con nulos que se usarán como predictores
CATEGORICAS = [
    "mother_educ", "father_educ", "computer", "internet",
    "room", "television", "computer_n", "car", "book",
]

ETIQUETA_FALTANTE = "Sin respuesta"


def preparar_datos(df):
    """Aplica las decisiones de limpieza y devuelve el dataset limpio
    junto con la bitácora de decisiones. No modifica el CSV original."""

    limpio = df.copy()
    bitacora = []

    # 1. Duplicados
    n_dup = int(limpio.duplicated().sum())
    n_dup_id = int(limpio["student_id"].duplicated().sum())
    limpio = limpio.drop_duplicates()
    bitacora.append({
        "Problema": "Registros duplicados",
        "Afecta": f"{n_dup} filas ({n_dup_id} IDs repetidos)",
        "Decisión": "Verificar y eliminar si existen",
        "Justificación": "No hay duplicados: cada alumno aparece una sola vez.",
    })

    # 2. Columnas 100% vacías
    vacias = [c for c in limpio.columns if limpio[c].isnull().all()]
    limpio = limpio.drop(columns=vacias)
    bitacora.append({
        "Problema": "Columnas 100% vacías",
        "Afecta": ", ".join(vacias),
        "Decisión": "Eliminar columnas",
        "Justificación": "No contienen ningún dato en 2022; no aportan información.",
    })

    # 3. Columnas constantes
    constantes = [c for c in limpio.columns if limpio[c].nunique() == 1]
    limpio = limpio.drop(columns=constantes)
    bitacora.append({
        "Problema": "Columnas constantes",
        "Afecta": ", ".join(constantes),
        "Decisión": "Eliminar columnas",
        "Justificación": "Tienen un único valor (2022, MEX); no distinguen a ningún alumno.",
    })

    # 4. Nulos en categóricas -> categoría propia
    n_filas_con_nulo = int(limpio[CATEGORICAS].isnull().any(axis=1).sum())
    for c in CATEGORICAS:
        limpio[c] = limpio[c].fillna(ETIQUETA_FALTANTE)
    bitacora.append({
        "Problema": "Nulos en variables categóricas (0.5%–5.9%)",
        "Afecta": f"{n_filas_con_nulo} alumnos con al menos un nulo",
        "Decisión": f"Conservar filas y crear la categoría '{ETIQUETA_FALTANTE}'",
        "Justificación": (
            "Los nulos no son aleatorios: quienes no responden alcanzan el "
            "Nivel 2 mucho menos. Borrarlos eliminaría a los alumnos de mayor riesgo."
        ),
    })

    # 5. Nulos en escs -> mediana
    n_escs = int(limpio["escs"].isnull().sum())
    mediana_escs = limpio["escs"].median()
    limpio["escs_imputado"] = limpio["escs"].isnull().astype(int)
    limpio["escs"] = limpio["escs"].fillna(mediana_escs)
    bitacora.append({
        "Problema": "Nulos en escs (índice socioeconómico)",
        "Afecta": f"{n_escs} alumnos (0.3%)",
        "Decisión": f"Imputar con la mediana ({mediana_escs:.2f}) y marcar con escs_imputado",
        "Justificación": (
            "Son muy pocos casos; la mediana no se mueve por valores extremos "
            "y la marca permite saber qué valores fueron imputados."
        ),
    })

    # 6. Rangos de puntajes
    bitacora.append({
        "Problema": "Valores atípicos en puntajes",
        "Afecta": (
            f"math entre {limpio['math'].min():.0f} y {limpio['math'].max():.0f}"
        ),
        "Decisión": "Conservar",
        "Justificación": (
            "Todos los puntajes están dentro de la escala de PISA; los extremos "
            "son alumnos reales con desempeño muy bajo o muy alto."
        ),
    })

    return limpio, pd.DataFrame(bitacora)


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------
    st.header("2. Limpiar los datos")
    st.markdown("""
    ### Pregunta
    **¿Qué tan confiables son los datos?**
    """)

    limpio, bitacora = preparar_datos(df)

    # ----------------------------------
    # ANTES Y DESPUÉS
    # ----------------------------------
    st.subheader("Antes y después de la limpieza")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Alumnos", f"{len(limpio):,}", delta=f"{len(limpio) - len(df)} filas")
    with col2:
        st.metric(
            "Variables",
            len(limpio.columns),
            delta=f"{len(limpio.columns) - len(df.columns)} columnas",
        )
    with col3:
        st.metric("Nulos antes", f"{int(df.isnull().sum().sum()):,}")
    with col4:
        st.metric("Nulos después", int(limpio.isnull().sum().sum()))

    st.caption(
        "De los nulos iniciales, 18,864 provienen de las 3 columnas vacías "
        "(3 × 6,288 alumnos); el resto son respuestas faltantes del cuestionario. "
        "Ningún alumno fue eliminado."
    )

    # ----------------------------------
    # CONTEO DE NULOS
    # ----------------------------------
    st.subheader("Valores faltantes por variable (antes)")

    nulos = (
        df.isnull().sum()
        .rename("Faltantes")
        .to_frame()
        .assign(Porcentaje=lambda t: (t["Faltantes"] / len(df) * 100).round(1))
        .query("Faltantes > 0")
        .sort_values("Faltantes", ascending=False)
        .reset_index(names="Variable")
    )

    st.dataframe(nulos, width="stretch", hide_index=True)

    # ----------------------------------
    # EVIDENCIA: ¿LOS NULOS SON ALEATORIOS?
    # ----------------------------------
    st.subheader("¿Los nulos son aleatorios?")

    filas = []
    for c in CATEGORICAS + ["escs"]:
        falta = df[c].isnull()
        filas.append({
            "Variable": c,
            "Grupo": "No respondió",
            "% alcanza Nivel 2": df.loc[falta, "alcanza_nivel2_mat"].mean() * 100,
        })
        filas.append({
            "Variable": c,
            "Grupo": "Sí respondió",
            "% alcanza Nivel 2": df.loc[~falta, "alcanza_nivel2_mat"].mean() * 100,
        })
    comparacion = pd.DataFrame(filas)

    fig = px.bar(
        comparacion,
        x="Variable",
        y="% alcanza Nivel 2",
        color="Grupo",
        barmode="group",
        title="Porcentaje que alcanza el Nivel 2 en matemáticas según si respondió la pregunta",
        color_discrete_map={"No respondió": "#d1495b", "Sí respondió": "#2e86ab"},
    )
    fig.update_layout(yaxis_ticksuffix="%", xaxis_title="Pregunta del cuestionario")
    st.plotly_chart(fig)

    st.info(
        "**Hallazgo:** en todas las preguntas, quienes no respondieron alcanzan el "
        "Nivel 2 con mucha menor frecuencia (6%–24%) que quienes sí respondieron (~34%). "
        "Si elimináramos las filas con nulos perderíamos "
        f"{int(df[CATEGORICAS + ['escs']].isnull().any(axis=1).sum())} alumnos "
        "y subestimaríamos el riesgo. Por eso conservamos esas filas."
    )

    # ----------------------------------
    # DECISIONES
    # ----------------------------------
    st.subheader("Decisiones de limpieza")
    st.dataframe(bitacora, width="stretch", hide_index=True)

    # ----------------------------------
    # RESULTADO
    # ----------------------------------
    st.subheader("Vista previa del dataset limpio")
    st.dataframe(limpio.head(10), width="stretch")
