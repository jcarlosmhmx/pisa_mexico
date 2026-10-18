import streamlit as st
import pandas as pd
import plotly.express as px

from tabs.comun import (
    UMBRAL_NIVEL2, ETIQUETAS, ORDEN,
    agregar_segmentos, tasa_ponderada, resumen_por_segmento,
)


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------
    st.header("3. Estadística descriptiva")
    st.markdown("""
    ### Pregunta
    **¿Cómo se distribuye el desempeño, qué segmentos de alumnos existen
    y qué KPIs los describen mejor?**
    """)

    df = agregar_segmentos(df)

    q1 = df[df["cuartil_escs"] == "Q1 (más bajo)"]
    q4 = df[df["cuartil_escs"] == "Q4 (más alto)"]
    tasa_global = tasa_ponderada(df)
    tasa_q1 = tasa_ponderada(q1)
    tasa_q4 = tasa_ponderada(q4)

    # ----------------------------------
    # KPIs
    # ----------------------------------
    st.subheader("KPIs principales")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "% que alcanza el Nivel 2",
            f"{tasa_global:.1%}",
            help="Ponderado con stu_wgt: representa a la población de alumnos de 15 años.",
        )
    with col2:
        st.metric(
            "Puntaje medio en matemáticas",
            f"{tasa_ponderada(df, 'math'):.0f}",
            delta=f"{tasa_ponderada(df, 'math') - UMBRAL_NIVEL2:.0f} vs. umbral Nivel 2",
        )
    with col3:
        st.metric(
            "Brecha socioeconómica (Q4 − Q1)",
            f"{(tasa_q4 - tasa_q1) * 100:.0f} pp",
            help="Diferencia en puntos porcentuales entre el cuartil más alto y el más bajo de ESCS.",
        )
    with col4:
        st.metric(
            "Alumnos en riesgo en el cuartil más bajo",
            f"{1 - tasa_q1:.0%}",
            help="Porcentaje del cuartil Q1 de ESCS que NO alcanza el Nivel 2.",
        )

    st.caption(
        "Porcentajes y promedios ponderados con el peso muestral `stu_wgt`, "
        "como recomienda la OCDE, para que representen a la población y no solo a la muestra."
    )

    # ----------------------------------
    # DISTRIBUCIÓN DEL DESEMPEÑO
    # ----------------------------------
    st.subheader("¿Cómo se distribuye el desempeño?")

    fig = px.histogram(
        df,
        x="math",
        nbins=50,
        color="cuartil_escs",
        category_orders={"cuartil_escs": ORDEN["cuartil_escs"]},
        barmode="overlay",
        opacity=0.6,
        title="Distribución del puntaje en matemáticas por nivel socioeconómico",
        labels={"math": "Puntaje en matemáticas", "cuartil_escs": "Cuartil ESCS"},
        color_discrete_sequence=["#d1495b", "#edae49", "#66a182", "#2e86ab"],
    )
    fig.add_vline(
        x=UMBRAL_NIVEL2,
        line_dash="dash",
        annotation_text="Umbral Nivel 2 (420)",
        annotation_position="top",
    )
    fig.update_layout(yaxis_title="Número de alumnos")
    st.plotly_chart(fig)

    st.markdown(
        f"La mayoría de los alumnos queda **a la izquierda del umbral**. "
        f"Las distribuciones se desplazan a la derecha conforme sube el nivel "
        f"socioeconómico: en Q1 solo **{tasa_q1:.0%}** alcanza el Nivel 2, "
        f"contra **{tasa_q4:.0%}** en Q4."
    )

    # ----------------------------------
    # GRÁFICA POR SEGMENTO
    # ----------------------------------
    st.subheader("¿Qué segmentos existen?")

    segmento = st.selectbox(
        "Segmentar por:",
        list(ETIQUETAS.keys()),
        format_func=lambda c: ETIQUETAS[c],
        key="segmento_descriptiva",
    )

    tabla = resumen_por_segmento(df, segmento)

    fig = px.bar(
        tabla,
        x=segmento,
        y="% alcanza Nivel 2",
        text=tabla["% alcanza Nivel 2"].map(lambda v: f"{v:.0f}%"),
        title=f"% que alcanza el Nivel 2 en matemáticas según {ETIQUETAS[segmento].lower()}",
        labels={segmento: ETIQUETAS[segmento]},
        color="% alcanza Nivel 2",
        color_continuous_scale=["#d1495b", "#edae49", "#2e86ab"],
        range_color=[0, 60],
    )
    fig.add_hline(
        y=tasa_global * 100,
        line_dash="dash",
        annotation_text=f"Promedio México ({tasa_global:.0%})",
        annotation_position="top left",
    )
    fig.update_layout(yaxis_ticksuffix="%", coloraxis_showscale=False)
    # Etiquetas como "0", "1", "3+" se tratan como categorías, no como números
    fig.update_xaxes(type="category")
    st.plotly_chart(fig)

    st.dataframe(
        tabla.style.format({
            "% alcanza Nivel 2": "{:.1f}%",
            "Puntaje medio en matemáticas": "{:.0f}",
        }),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Categorías con pocos alumnos (p. ej. 'More than 500' libros, n = 62) "
        "tienen porcentajes menos estables; interprétalas con cautela."
    )

    # ----------------------------------
    # EL PAPEL DE LA ESCUELA
    # ----------------------------------
    st.subheader("¿Importa la escuela?")

    por_escuela = (
        df.groupby("school_id")
        .apply(lambda g: pd.Series({
            "% alcanza Nivel 2": tasa_ponderada(g) * 100,
            "ESCS medio": g["escs"].mean(),
            "Alumnos": len(g),
        }), include_groups=False)
        .reset_index()
    )

    fig = px.scatter(
        por_escuela,
        x="ESCS medio",
        y="% alcanza Nivel 2",
        size="Alumnos",
        title="Cada punto es una escuela: nivel socioeconómico medio vs. % que alcanza el Nivel 2",
        labels={"ESCS medio": "ESCS medio de la escuela"},
        color_discrete_sequence=["#2e86ab"],
        opacity=0.6,
    )
    fig.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(fig)

    n_cero = int((por_escuela["% alcanza Nivel 2"] == 0).sum())
    st.markdown(
        f"Hay una gran dispersión entre escuelas: en **{n_cero} de "
        f"{len(por_escuela)}** ningún alumno evaluado alcanzó el Nivel 2. "
        "Las escuelas con alumnos de mayor nivel socioeconómico tienden a tener "
        "mejores resultados, lo que sugiere que el apoyo debe focalizarse "
        "tanto en alumnos como en escuelas."
    )
