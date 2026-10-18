import streamlit as st
import pandas as pd
import plotly.express as px

from tabs.comun import (
    entrenar_modelo, NIVELES_RIESGO, UMBRAL_RIESGO_ALTO, UMBRAL_RIESGO_MEDIO,
)


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------
    st.header("4. Estadística predictiva")
    st.markdown("""
    ### Pregunta
    **¿Qué características tienen quienes alcanzan el Nivel 2
    y podemos predecir quién estará en riesgo?**
    """)

    modelo, metricas, coef, resultado = entrenar_modelo(df)

    st.markdown(
        "**Modelo:** regresión logística (scikit-learn) que estima la probabilidad "
        "de que un alumno alcance el Nivel 2 en matemáticas a partir de su contexto "
        "familiar. **No usa** los puntajes de lectura ni ciencias, porque eso sería "
        "predecir con información casi igual al resultado (fuga de información)."
    )

    # ----------------------------------
    # DESEMPEÑO DEL MODELO
    # ----------------------------------
    st.subheader("¿Qué tan bien predice?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "AUC (datos de prueba)",
            f"{metricas['auc']:.2f}",
            help="0.5 = azar; 1.0 = perfecto. Mide qué tan bien ordena a los alumnos por riesgo.",
        )
    with col2:
        st.metric(
            "Exactitud",
            f"{metricas['exactitud']:.0%}",
            help="Porcentaje de alumnos de prueba clasificados correctamente con umbral 0.5.",
        )
    with col3:
        st.metric(
            "Alumnos de entrenamiento / prueba",
            f"{metricas['n_train']:,} / {metricas['n_test']:,}",
            help="El modelo se evalúa con alumnos que no vio al entrenar (25%).",
        )

    m = metricas["matriz"]
    matriz = pd.DataFrame(
        m,
        index=["Real: no alcanza", "Real: sí alcanza"],
        columns=["Predicho: no alcanza", "Predicho: sí alcanza"],
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Matriz de confusión (datos de prueba)**")
        st.dataframe(matriz, width="stretch")

    with col2:
        detectados = m[0, 0] / m[0].sum()
        st.markdown("**Lectura**")
        st.markdown(
            f"- De cada 10 alumnos que **no** alcanzaron el Nivel 2, el modelo detecta "
            f"a **{detectados * 10:.0f}**.\n"
            f"- AUC de {metricas['auc']:.2f}: el modelo ordena el riesgo mejor que el "
            "azar, pero **no es infalible**; sirve para priorizar, no para etiquetar alumnos.\n"
            f"- Si se dijera 'nadie alcanza' se acertaría {metricas['linea_base']:.0%}, "
            "pero no se distinguiría a ningún alumno. El modelo sacrifica algo de exactitud "
            "para encontrar a quienes sí tienen oportunidad y a quienes están en mayor riesgo."
        )

    # ----------------------------------
    # VARIABLES QUE MÁS PESAN
    # ----------------------------------
    st.subheader("¿Qué variables pesan más?")

    coef_graf = coef.copy()
    coef_graf["Efecto"] = coef_graf["Coeficiente"].map(
        lambda c: "Aumenta la probabilidad de alcanzar el Nivel 2" if c > 0
        else "Disminuye la probabilidad de alcanzar el Nivel 2"
    )

    fig = px.bar(
        coef_graf,
        x="Coeficiente",
        y="Variable",
        orientation="h",
        color="Efecto",
        title="Peso de cada variable en el modelo (coeficientes estandarizados)",
        color_discrete_map={
            "Aumenta la probabilidad de alcanzar el Nivel 2": "#2e86ab",
            "Disminuye la probabilidad de alcanzar el Nivel 2": "#d1495b",
        },
        text=coef_graf["Coeficiente"].map(lambda v: f"{v:+.2f}"),
    )
    fig.update_layout(
        xaxis_title="Coeficiente (variables estandarizadas: comparables entre sí)",
        yaxis_title="",
        legend_title="",
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig)

    principal = coef.iloc[-1]
    negativa = coef.iloc[0]
    st.markdown(
        f"La variable con más peso es **{principal['Variable'].lower()}**. "
        "Tener computadora e internet en casa también aumenta la probabilidad, aun "
        "comparando alumnos con el mismo nivel socioeconómico. Del lado contrario, "
        f"**{negativa['Variable'].lower()}** es el factor que más la reduce. "
        "Los coeficientes indican **asociación, no causalidad**."
    )

    with st.expander("Ver tabla de coeficientes"):
        st.dataframe(
            coef.style.format({"Coeficiente": "{:+.3f}", "Razón de momios": "{:.2f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Razón de momios > 1: aumenta las probabilidades de alcanzar el Nivel 2 "
            "por cada desviación estándar de la variable; < 1: las reduce."
        )

    # ----------------------------------
    # NIVELES DE RIESGO
    # ----------------------------------
    st.subheader("Del modelo a niveles de riesgo")

    st.markdown(
        f"Con la probabilidad estimada de **no** alcanzar el Nivel 2 clasificamos a "
        f"cada alumno: **riesgo alto** (≥ {UMBRAL_RIESGO_ALTO:.0%}), **riesgo medio** "
        f"({UMBRAL_RIESGO_MEDIO:.0%}–{UMBRAL_RIESGO_ALTO:.0%}) y **riesgo bajo** "
        f"(< {UMBRAL_RIESGO_MEDIO:.0%})."
    )

    niveles = (
        resultado.groupby("nivel_riesgo")
        .agg(
            Alumnos=("student_id", "count"),
            real_no_alcanza=("alcanza_nivel2_mat", lambda s: 1 - s.mean()),
            prob_media=("prob_riesgo", "mean"),
        )
        .reindex(NIVELES_RIESGO)
        .reset_index()
        .rename(columns={"nivel_riesgo": "Nivel de riesgo"})
    )
    niveles["% real que NO alcanzó el Nivel 2"] = niveles.pop("real_no_alcanza") * 100
    niveles["Riesgo promedio estimado"] = niveles.pop("prob_media") * 100

    fig = px.bar(
        niveles,
        x="Nivel de riesgo",
        y="% real que NO alcanzó el Nivel 2",
        text=niveles["% real que NO alcanzó el Nivel 2"].map(lambda v: f"{v:.0f}%"),
        color="Nivel de riesgo",
        color_discrete_map={
            "Riesgo alto": "#d1495b", "Riesgo medio": "#edae49", "Riesgo bajo": "#2e86ab",
        },
        title="¿El modelo acierta? % de alumnos que realmente no alcanzó el Nivel 2 en cada nivel",
    )
    fig.update_layout(yaxis_ticksuffix="%", showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig)

    st.dataframe(
        niveles.style.format({
            "% real que NO alcanzó el Nivel 2": "{:.0f}%",
            "Riesgo promedio estimado": "{:.0f}%",
        }),
        width="stretch",
        hide_index=True,
    )

    alto = niveles.set_index("Nivel de riesgo").loc["Riesgo alto"]
    st.info(
        f"**Hallazgo:** de los **{int(alto['Alumnos']):,}** alumnos clasificados en riesgo "
        f"alto, **{alto['% real que NO alcanzó el Nivel 2']:.0f}%** efectivamente no alcanzó "
        "el Nivel 2. Los niveles de riesgo ordenan bien a los alumnos y sirven para "
        "priorizar a quién apoyar primero."
    )
