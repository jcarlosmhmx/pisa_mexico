import numpy as np
import pandas as pd


UMBRAL_NIVEL2 = 420.07

# Orden lógico de las categorías para gráficas y tablas
ORDEN = {
    "cuartil_escs": ["Q1 (más bajo)", "Q2", "Q3", "Q4 (más alto)"],
    "book": ["0", "1-10", "11-25", "26-100", "101-200", "201-500",
             "More than 500", "Sin respuesta"],
    "mother_educ": ["less than ISCED1", "ISCED 1", "ISCED 2", "ISCED 3B, C",
                    "ISCED 3A", "Sin respuesta"],
    "father_educ": ["less than ISCED1", "ISCED 1", "ISCED 2", "ISCED 3B, C",
                    "ISCED 3A", "Sin respuesta"],
    "gender": ["female", "male"],
    "computer": ["no", "yes", "Sin respuesta"],
    "internet": ["no", "yes", "Sin respuesta"],
    "room": ["no", "yes", "Sin respuesta"],
    "computer_n": ["0", "1", "2", "3+", "Sin respuesta"],
    "car": ["0", "1", "2", "3+", "Sin respuesta"],
    "television": ["0", "1", "2", "3+", "Sin respuesta"],
}

ETIQUETAS = {
    "cuartil_escs": "Nivel socioeconómico (cuartil ESCS)",
    "book": "Libros en casa",
    "mother_educ": "Escolaridad de la madre",
    "father_educ": "Escolaridad del padre",
    "gender": "Género",
    "computer": "Tiene computadora",
    "internet": "Tiene internet",
    "room": "Tiene cuarto propio",
    "computer_n": "Número de computadoras",
    "car": "Número de autos",
    "television": "Número de televisores",
}


def agregar_segmentos(df):
    """Agrega columnas de segmento al dataset limpio."""
    df = df.copy()
    df["cuartil_escs"] = pd.qcut(
        df["escs"], 4, labels=ORDEN["cuartil_escs"]
    ).astype(str)
    for c in ["television", "computer_n", "car"]:
        df[c] = df[c].astype(str)
    return df


def tasa_ponderada(g, col="alcanza_nivel2_mat"):
    """Promedio ponderado por el peso muestral de PISA."""
    return np.average(g[col], weights=g["stu_wgt"])


def resumen_por_segmento(df, segmento):
    """% que alcanza el Nivel 2 y puntaje medio por categoría, ponderados."""
    filas = []
    for valor, g in df.groupby(segmento, observed=True):
        filas.append({
            segmento: str(valor),
            "Alumnos (muestra)": len(g),
            "% alcanza Nivel 2": tasa_ponderada(g) * 100,
            "Puntaje medio en matemáticas": tasa_ponderada(g, "math"),
        })
    tabla = pd.DataFrame(filas)
    if segmento in ORDEN:
        orden = [v for v in ORDEN[segmento] if v in tabla[segmento].values]
        tabla[segmento] = pd.Categorical(tabla[segmento], orden, ordered=True)
        tabla = tabla.sort_values(segmento)
    return tabla.reset_index(drop=True)


# ----------------------------------
# MODELO PREDICTIVO (compartido por las pestañas 4, 5 y 6)
# ----------------------------------
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

CATEGORICAS_CUESTIONARIO = [
    "mother_educ", "father_educ", "computer", "internet",
    "room", "television", "computer_n", "car", "book",
]

PREDICTORES = {
    "escs": "Índice socioeconómico (ESCS)",
    "computadora_en_casa": "Tiene computadora en casa",
    "internet_en_casa": "Tiene internet en casa",
    "mas_de_25_libros": "Tiene más de 25 libros en casa",
    "sin_libros": "No tiene libros en casa",
    "mujer": "Es mujer",
    "respuestas_faltantes": "Preguntas sin responder en el cuestionario",
}

# Umbrales de probabilidad de NO alcanzar el Nivel 2
UMBRAL_RIESGO_MEDIO = 0.50
UMBRAL_RIESGO_ALTO = 0.65
NIVELES_RIESGO = ["Riesgo alto", "Riesgo medio", "Riesgo bajo"]


def construir_predictores(df):
    """Convierte el dataset limpio en variables interpretables (0/1 o numéricas)."""
    return pd.DataFrame({
        "escs": df["escs"],
        "computadora_en_casa": (df["computer"] == "yes").astype(int),
        "internet_en_casa": (df["internet"] == "yes").astype(int),
        "mas_de_25_libros": df["book"].isin(
            ["26-100", "101-200", "201-500", "More than 500"]).astype(int),
        "sin_libros": (df["book"] == "0").astype(int),
        "mujer": (df["gender"] == "female").astype(int),
        "respuestas_faltantes": (
            df[CATEGORICAS_CUESTIONARIO] == "Sin respuesta").sum(axis=1),
    }, index=df.index)


@st.cache_resource
def entrenar_modelo(df):
    """Entrena una regresión logística y devuelve modelo, métricas y
    el dataset con la probabilidad de riesgo de cada alumno."""
    X = construir_predictores(df)
    y = df["alcanza_nivel2_mat"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    modelo = Pipeline([
        ("escalar", StandardScaler()),
        ("logistica", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    modelo.fit(X_train, y_train)

    prob_test = modelo.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)

    metricas = {
        "auc": roc_auc_score(y_test, prob_test),
        "exactitud": (pred_test == y_test).mean(),
        "linea_base": 1 - y_test.mean(),
        "matriz": confusion_matrix(y_test, pred_test, labels=[0, 1]),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    coeficientes = pd.DataFrame({
        "Variable": [PREDICTORES[c] for c in X.columns],
        "Coeficiente": modelo.named_steps["logistica"].coef_[0],
    })
    coeficientes["Razón de momios"] = np.exp(coeficientes["Coeficiente"])
    coeficientes = coeficientes.sort_values("Coeficiente").reset_index(drop=True)

    resultado = df.copy()
    resultado["prob_nivel2"] = modelo.predict_proba(X)[:, 1]
    resultado["prob_riesgo"] = 1 - resultado["prob_nivel2"]
    resultado["nivel_riesgo"] = np.select(
        [resultado["prob_riesgo"] >= UMBRAL_RIESGO_ALTO,
         resultado["prob_riesgo"] >= UMBRAL_RIESGO_MEDIO],
        ["Riesgo alto", "Riesgo medio"],
        default="Riesgo bajo",
    )

    return modelo, metricas, coeficientes, resultado


# ----------------------------------
# PLAN DE ACCIÓN (compartido por las pestañas 5 y 6)
# ----------------------------------
def perfil_por_riesgo(resultado):
    """Perfil de cada nivel de riesgo: tamaño, resultado real y recursos en casa."""
    X = construir_predictores(resultado)
    datos = resultado.join(X[["computadora_en_casa", "internet_en_casa", "sin_libros", "mujer"]])
    perfil = (
        datos.groupby("nivel_riesgo")
        .agg(
            alumnos=("student_id", "count"),
            no_alcanza=("alcanza_nivel2_mat", lambda s: 1 - s.mean()),
            sin_computadora=("computadora_en_casa", lambda s: 1 - s.mean()),
            sin_internet=("internet_en_casa", lambda s: 1 - s.mean()),
            sin_libros=("sin_libros", "mean"),
            mujeres=("mujer", "mean"),
            escs=("escs", "mean"),
            puntaje=("math", "mean"),
        )
        .reindex(NIVELES_RIESGO)
    )
    return perfil


def escuelas_prioritarias(resultado, umbral=0.5):
    """Escuelas donde al menos la mitad de sus alumnos está en riesgo alto."""
    por_escuela = resultado.groupby("school_id").agg(
        alumnos=("student_id", "count"),
        pct_riesgo_alto=("nivel_riesgo", lambda s: (s == "Riesgo alto").mean()),
    )
    return por_escuela[por_escuela["pct_riesgo_alto"] >= umbral]


PLAN = {
    "Riesgo alto": {
        "prioridad": "1 (inmediata)",
        "accion": (
            "Tutoría intensiva de matemáticas en grupos pequeños dentro de la escuela, "
            "con material impreso y una biblioteca de aula; foco en las escuelas "
            "con mayor concentración de riesgo alto."
        ),
        "por_que": (
            "La gran mayoría no tiene computadora en casa, así que un apoyo digital "
            "no les llegaría: tiene que ser presencial y no depender de recursos del hogar."
        ),
        "canal": "Presencial en la escuela + SMS/WhatsApp a las familias",
        "meta": "Reducir en 10 puntos porcentuales el % que no alcanza el Nivel 2 en este grupo",
        "mensajes": {
            "Alumno": (
                "Las matemáticas se aprenden practicando, no se nace sabiendo. "
                "Únete al club de matemáticas: martes y jueves después de clases, "
                "en tu escuela, con todo el material incluido."
            ),
            "Familia": (
                "Su hija o hijo puede mejorar en matemáticas. La escuela ofrece "
                "tutorías gratuitas dos tardes por semana. No necesita computadora "
                "ni internet: todo el material se entrega impreso."
            ),
            "Director/a": (
                "Su escuela tiene una alta concentración de alumnos en riesgo en "
                "matemáticas. Le proponemos abrir dos grupos de tutoría por semana "
                "y una biblioteca de aula; nosotros aportamos el material."
            ),
        },
    },
    "Riesgo medio": {
        "prioridad": "2 (este semestre)",
        "accion": (
            "Refuerzo semanal focalizado en los temas con más errores, combinando "
            "clase presencial con práctica en el laboratorio de cómputo escolar, "
            "y seguimiento mensual del avance."
        ),
        "por_que": (
            "Una parte importante no tiene computadora en casa, pero casi todos "
            "tienen internet; la escuela puede cerrar la brecha de equipo."
        ),
        "canal": "Clase de refuerzo + laboratorio de cómputo escolar",
        "meta": "Que la mitad de este grupo suba a riesgo bajo en la siguiente evaluación",
        "mensajes": {
            "Alumno": (
                "Estás más cerca de lo que crees. Con una hora de práctica a la semana "
                "en el laboratorio de la escuela puedes subir tu nivel en matemáticas."
            ),
            "Familia": (
                "Su hija o hijo tiene buenas bases y puede avanzar. Le pedimos "
                "motivarle a asistir al refuerzo semanal; le informaremos su avance cada mes."
            ),
            "Director/a": (
                "Con abrir el laboratorio de cómputo una hora a la semana para práctica "
                "guiada de matemáticas, este grupo puede avanzar de forma medible."
            ),
        },
    },
    "Riesgo bajo": {
        "prioridad": "3 (mantener y aprovechar)",
        "accion": (
            "Práctica autónoma en plataformas digitales gratuitas, retos y olimpiadas, "
            "e invitar a quienes ya alcanzan el Nivel 2 a ser tutores de sus compañeros."
        ),
        "por_que": (
            "Casi todos tienen computadora e internet en casa, así que el apoyo "
            "digital y autónomo es viable y de bajo costo; además pueden apoyar a otros."
        ),
        "canal": "Plataforma en línea + programa de tutores pares",
        "meta": "Formar un tutor par por cada cinco alumnos en riesgo alto de la misma escuela",
        "mensajes": {
            "Alumno": (
                "Tu esfuerzo en matemáticas puede llegar más lejos: participa en los "
                "retos mensuales y conviértete en tutor de un compañero."
            ),
            "Familia": (
                "Su hija o hijo puede aprovechar plataformas gratuitas de práctica "
                "desde casa y participar como tutor en su escuela."
            ),
            "Director/a": (
                "Sus mejores alumnos en matemáticas pueden ser tutores pares: es una "
                "forma de bajo costo de apoyar a quienes más lo necesitan."
            ),
        },
    },
}
