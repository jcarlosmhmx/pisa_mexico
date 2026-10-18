import streamlit as st
import pandas as pd
import plotly.express as px

from tabs.comun import (
    entrenar_modelo, perfil_por_riesgo, escuelas_prioritarias, PLAN, NIVELES_RIESGO,
)


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------
    st.header("5. Estadística prescriptiva")
    st.markdown("""
    ### Pregunta
    **¿Qué apoyo deberíamos ofrecer a cada grupo de alumnos?**
    """)

    _, _, _, resultado = entrenar_modelo(df)
    perfil = perfil_por_riesgo(resultado)
    prioritarias = escuelas_prioritarias(resultado)

    st.markdown(
        "Partimos de los **niveles de riesgo** de la pestaña 4 y revisamos el perfil "
        "de cada grupo. La acción se elige según **qué recursos tienen** los alumnos, "
        "porque un mismo apoyo no sirve para todos."
    )

    # ----------------------------------
    # PERFIL DE CADA GRUPO
    # ----------------------------------
    st.subheader("¿Cómo es cada grupo?")

    perfil_largo = (
        perfil[["sin_computadora", "sin_internet", "sin_libros", "mujeres"]]
        .mul(100)
        .rename(columns={
            "sin_computadora": "Sin computadora en casa",
            "sin_internet": "Sin internet en casa",
            "sin_libros": "Sin libros en casa",
            "mujeres": "Mujeres",
        })
        .reset_index()
        .melt(id_vars="nivel_riesgo", var_name="Característica", value_name="Porcentaje")
    )

    fig = px.bar(
        perfil_largo,
        x="Característica",
        y="Porcentaje",
        color="nivel_riesgo",
        barmode="group",
        text=perfil_largo["Porcentaje"].map(lambda v: f"{v:.0f}%"),
        category_orders={"nivel_riesgo": NIVELES_RIESGO},
        color_discrete_map={
            "Riesgo alto": "#d1495b", "Riesgo medio": "#edae49", "Riesgo bajo": "#2e86ab",
        },
        title="Perfil de recursos y género por nivel de riesgo",
        labels={"nivel_riesgo": "Nivel de riesgo"},
    )
    fig.update_layout(yaxis_ticksuffix="%", xaxis_title="", yaxis_range=[0, 100])
    st.plotly_chart(fig)

    alto = perfil.loc["Riesgo alto"]
    st.markdown(
        f"En el grupo de **riesgo alto**, **{alto['sin_computadora']:.0%} no tiene "
        f"computadora** en casa y {alto['sin_libros']:.0%} no tiene libros; además, "
        f"**{alto['mujeres']:.0%} son mujeres**. Su puntaje medio es "
        f"{alto['puntaje']:.0f}, unos {420 - alto['puntaje']:.0f} puntos debajo del umbral. "
        "**Conclusión:** para este grupo, un apoyo digital desde casa no funcionaría."
    )

    # ----------------------------------
    # TABLA DE ACCIÓN RECOMENDADA
    # ----------------------------------
    st.subheader("Acción recomendada por nivel de riesgo")

    filas = []
    for nivel in NIVELES_RIESGO:
        p = perfil.loc[nivel]
        filas.append({
            "Nivel de riesgo": nivel,
            "Prioridad": PLAN[nivel]["prioridad"],
            "Alumnos": f"{int(p['alumnos']):,}",
            "% real que no alcanza": f"{p['no_alcanza']:.0%}",
            "Sin computadora": f"{p['sin_computadora']:.0%}",
            "Acción recomendada": PLAN[nivel]["accion"],
            "Canal": PLAN[nivel]["canal"],
        })

    st.table(pd.DataFrame(filas).set_index("Nivel de riesgo"))

    # ----------------------------------
    # ¿DÓNDE EMPEZAR?
    # ----------------------------------
    st.subheader("¿Dónde empezar?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Escuelas prioritarias", len(prioritarias),
                  help="Escuelas donde al menos la mitad de sus alumnos está en riesgo alto.")
    with col2:
        st.metric("Alumnos en esas escuelas", f"{int(prioritarias['alumnos'].sum()):,}")
    with col3:
        en_prioritarias = resultado[resultado["school_id"].isin(prioritarias.index)]
        cobertura = (en_prioritarias["nivel_riesgo"] == "Riesgo alto").sum() / (
            resultado["nivel_riesgo"] == "Riesgo alto").sum()
        st.metric("Alumnos de riesgo alto que cubren", f"{cobertura:.0%}")

    st.info(
        f"**Recomendación:** empezar el programa de tutoría intensiva en las "
        f"**{len(prioritarias)} escuelas prioritarias**. Al concentrar el apoyo ahí se llega "
        f"a **{cobertura:.0%}** de todos los alumnos de riesgo alto trabajando solo en el "
        f"**{len(prioritarias) / resultado['school_id'].nunique():.0%} de las escuelas**, "
        "lo que hace el programa más eficiente."
    )

    st.caption(
        "Las acciones son una propuesta basada en los datos de la app; antes de aplicarlas "
        "deben validarse con docentes y directivos, y medir su efecto con una evaluación posterior."
    )
