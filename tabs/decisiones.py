import streamlit as st

from tabs.comun import (
    entrenar_modelo, perfil_por_riesgo, escuelas_prioritarias, PLAN, NIVELES_RIESGO,
)


COLORES = {"Riesgo alto": "🔴", "Riesgo medio": "🟠", "Riesgo bajo": "🔵"}


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------
    st.header("6. Toma de decisiones")
    st.markdown("""
    ### Pregunta
    **¿Qué mensaje deberíamos mostrarle a cada grupo y por qué?**
    """)

    _, metricas, coef, resultado = entrenar_modelo(df)
    perfil = perfil_por_riesgo(resultado)
    prioritarias = escuelas_prioritarias(resultado)

    # ----------------------------------
    # SELECTORES
    # ----------------------------------
    col1, col2 = st.columns(2)

    with col1:
        nivel = st.selectbox(
            "Selecciona un grupo de alumnos:",
            NIVELES_RIESGO,
            format_func=lambda n: f"{COLORES[n]} {n}",
            key="nivel_decisiones",
        )

    with col2:
        audiencia = st.radio(
            "¿A quién va dirigido el mensaje?",
            list(PLAN[nivel]["mensajes"].keys()),
            horizontal=True,
            key="audiencia_decisiones",
        )

    plan = PLAN[nivel]
    p = perfil.loc[nivel]

    # ----------------------------------
    # TAMAÑO DEL GRUPO
    # ----------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Alumnos en el grupo", f"{int(p['alumnos']):,}",
                  help=f"{p['alumnos'] / len(resultado):.0%} de la muestra")
    with col2:
        st.metric("No alcanzó el Nivel 2", f"{p['no_alcanza']:.0%}")
    with col3:
        st.metric("Sin computadora en casa", f"{p['sin_computadora']:.0%}")
    with col4:
        st.metric("Puntaje medio en matemáticas", f"{p['puntaje']:.0f}",
                  delta=f"{p['puntaje'] - 420:.0f} vs. umbral", delta_color="normal")

    # ----------------------------------
    # ACCIÓN, MENSAJE Y JUSTIFICACIÓN
    # ----------------------------------
    st.subheader(f"{COLORES[nivel]} Decisión para el grupo: {nivel}")

    st.markdown(f"**Prioridad:** {plan['prioridad']}  \n**Canal:** {plan['canal']}")

    st.markdown("#### 🎯 Acción")
    st.success(plan["accion"])

    st.markdown(f"#### 💬 Mensaje para: {audiencia}")
    st.info(f"“{plan['mensajes'][audiencia]}”")

    st.markdown("#### 📊 Justificación con datos")

    escs_top = coef.iloc[-1]
    justificacion = [
        f"El modelo (AUC {metricas['auc']:.2f}) identifica que la variable con más peso "
        f"es **{escs_top['Variable'].lower()}**, seguida de tener computadora e internet en casa.",
        f"En este grupo, **{p['no_alcanza']:.0%}** de los alumnos realmente no alcanzó el "
        f"Nivel 2 y su puntaje medio es **{p['puntaje']:.0f}**.",
        f"**{p['sin_computadora']:.0%}** no tiene computadora y **{p['sin_internet']:.0%}** "
        f"no tiene internet en casa → {plan['por_que'][0].lower()}{plan['por_que'][1:]}",
    ]
    if nivel == "Riesgo alto":
        justificacion.append(
            f"**{p['mujeres']:.0%} son mujeres**, por lo que el mensaje evita estereotipos "
            "y refuerza que las matemáticas se aprenden con práctica."
        )
        justificacion.append(
            f"Concentrar el programa en **{len(prioritarias)} escuelas prioritarias** "
            "permite llegar a una gran parte de este grupo con menos recursos."
        )

    for linea in justificacion:
        st.markdown(f"- {linea}")

    st.markdown(f"**Meta medible:** {plan['meta']}.")

    # ----------------------------------
    # ALCANCE
    # ----------------------------------
    with st.expander("⚠️ Alcance y límites de esta decisión"):
        st.markdown(
            "- El modelo detecta aproximadamente 6 de cada 10 alumnos que no alcanzan el "
            "Nivel 2: **sirve para priorizar grupos, no para etiquetar a un alumno**.\n"
            f"- Incluso en el grupo de **riesgo bajo**, {perfil.loc['Riesgo bajo', 'no_alcanza']:.0%} "
            "no alcanza el Nivel 2: el riesgo es **relativo** y el reto es de todo el sistema.\n"
            "- Las variables muestran **asociación, no causalidad**: dar una computadora "
            "no garantiza por sí sola una mejora.\n"
            "- Los datos son de **PISA 2022**; en PISA 2025 México bajó a 388 puntos en "
            "matemáticas, por lo que el reto probablemente es mayor.\n"
            "- La decisión final debe validarse con docentes y familias."
        )
