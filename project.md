# project.md — PISA México: riesgo en matemáticas

Archivo de contexto del proyecto. Léelo antes de escribir o modificar código.

## 1. Contexto

- **Curso:** Ciencia de Datos – Toma de decisiones en los negocios (Maestría en Big Data e IA, Universidad Anáhuac Puebla).
- **Actividad:** F05 – App de Streamlit que lleva los datos de Entender → Limpiar → Describir → Predecir → Prescribir → Decidir.
- **Contexto elegido:** Escuela / alumnos. Alumnos mexicanos de 15 años evaluados en PISA.
- **Unidad de análisis:** alumno (una fila = un alumno).
- **Pregunta guía:**
  > ¿Qué alumnos tienen mayor riesgo de no alcanzar el nivel mínimo de competencia en matemáticas y qué apoyo deberíamos ofrecer a cada grupo?

## 2. Datos

- **Archivo:** `data/pisa_2022_mexico.csv` (6,288 alumnos, 280 escuelas, 23 variables).
- **Fuente:** Programme for International Student Assessment (PISA), OCDE, París. Microdatos de PISA 2022 en la versión depurada del paquete abierto `learningtower` (licencia MIT), filtrados a México (`country == "MEX"`).
- **Por qué 2022 y no 2025:** los microdatos de PISA 2025 requieren registro y sus términos de uso prohíben redistribuirlos, y este repositorio es público. Los resultados de 2025 se usan solo como contexto agregado (nota de país de la OCDE, sept. 2026).
- **Anonimización:** solo hay IDs numéricos de alumno y escuela; no hay nombres, correos ni teléfonos. Prohibido intentar reidentificar alumnos o escuelas.

### Diccionario de variables

| Variable | Tipo | Descripción |
|---|---|---|
| `year` | num | Año del ciclo PISA (2022) |
| `country` | cat | País (MEX) |
| `school_id` | id | ID anónimo de escuela |
| `student_id` | id | ID anónimo de alumno |
| `mother_educ` | cat | Escolaridad de la madre (ISCED: less than ISCED1, ISCED 1, 2, 3B/C, 3A) |
| `father_educ` | cat | Escolaridad del padre (misma escala) |
| `gender` | cat | Género (female / male) |
| `computer` | cat | Tiene computadora en casa (yes / no) |
| `internet` | cat | Tiene internet en casa (yes / no) |
| `math` | num | Puntaje en matemáticas |
| `read` | num | Puntaje en lectura |
| `science` | num | Puntaje en ciencias |
| `stu_wgt` | num | Peso muestral del alumno (cuántos alumnos de la población representa) |
| `desk` | — | Tiene escritorio (**100% vacía en 2022**) |
| `room` | cat | Tiene cuarto propio (yes / no) |
| `dishwasher` | — | Tiene lavavajillas (**100% vacía en 2022**) |
| `television` | cat | Número de televisores (0, 1, 2, 3+) |
| `computer_n` | cat | Número de computadoras (0, 1, 2, 3+) |
| `car` | cat | Número de autos (0, 1, 2, 3+) |
| `book` | cat | Libros en casa (0, 1-10, 11-25, 26-100, 101-200, 201-500, More than 500) |
| `wealth` | — | Índice de riqueza (**100% vacía en 2022**) |
| `escs` | num | Índice de estatus económico, social y cultural (OCDE; 0 = promedio OCDE) |
| `alcanza_nivel2_mat` | sí/no | **Variable resultado.** 1 si `math >= 420.07` (umbral oficial del Nivel 2 de la OCDE), 0 si no |

### Hechos conocidos del dataset

- 34% de los alumnos alcanza el Nivel 2 en matemáticas (coincide con la cifra publicada por la OCDE para México 2022).
- Sin duplicados.
- Nulos: `desk`, `dishwasher` y `wealth` están 100% vacías; `computer` y `computer_n` tienen ~6%, `father_educ` ~3% y el resto ≤ 2.5%.
- `math`, `read` y `science` están muy correlacionadas: **no usar `read` ni `science` como predictores** de `alcanza_nivel2_mat` (fuga de información). Úsalas solo en la parte descriptiva.

### Contexto PISA 2025 (México, solo cifras agregadas)

- Matemáticas 388 (2022: 395), lectura 408, ciencias 414.
- 30% alcanza el Nivel 2 en matemáticas (OCDE: 65%).
- El estatus socioeconómico explica ~12% de la variación en ciencias.

## 3. Entorno

```bash
mkdir pisa_mexico && cd pisa_mexico
python3 -m venv pisa_env
source pisa_env/bin/activate          # Windows: pisa_env\Scripts\Activate.ps1
pip install streamlit pandas scikit-learn plotly
pip freeze > requirements.txt
streamlit run app.py
```

## 4. Arquitectura

```
pisa_mexico/
├── pisa_env/            # entorno virtual (NO se sube a git)
├── data/
│   └── pisa_2022_mexico.csv
├── tabs/
│   ├── __init__.py
│   ├── entender.py
│   ├── limpiar.py
│   ├── descriptiva.py
│   ├── predictiva.py
│   ├── prescriptiva.py
│   └── decisiones.py
├── app.py
├── project.md
├── requirements.txt
├── README.md
└── .gitignore
```

- `app.py` configura la página, carga el CSV (con `@st.cache_data`), crea las 6 pestañas con `st.tabs()` y llama a `modulo.mostrar(df)` en cada una.
- Cada archivo de `tabs/` expone una función `mostrar(df)`.
- Cada pestaña debe funcionar antes de pasar a la siguiente.

## 5. Las seis pestañas

| # | Pestaña | Pregunta que responde | Qué construye |
|---|---|---|---|
| 1 | Entender | ¿Quiénes son los alumnos y qué información tenemos? | Métricas de filas y variables, vista previa, variables y tipos, diagnóstico, explorador de una variable |
| 2 | Limpiar | ¿Qué tan confiables son los datos? | Conteo de nulos y duplicados, y la decisión tomada con cada uno |
| 3 | Descriptiva | ¿Cómo se comportan, qué segmentos existen y qué KPIs los describen? | Gráficas por segmento (ESCS, libros, género, recursos) y KPIs con `st.metric` |
| 4 | Predictiva | ¿Qué caracteriza a quienes alcanzan el Nivel 2 y podemos predecirlo? | Regresión logística (scikit-learn) y variables con más peso |
| 5 | Prescriptiva | ¿Qué apoyo dar a cada grupo? | Tabla de acción recomendada por segmento / nivel de probabilidad |
| 6 | Decisiones | ¿Qué mensaje mostrar a cada grupo? | Selector de segmento que muestra acción, mensaje y justificación |

## 6. Convenciones de código

- Todo el texto de la interfaz, comentarios y nombres de secciones **en español**.
- Usar `width="stretch"` en lugar de `use_container_width=True` (deprecado en Streamlit).
- Dar `key` única a cada widget (`st.selectbox`, etc.) para evitar conflictos entre pestañas.
- No modificar el CSV original: la limpieza se hace en memoria y se documenta en la pestaña 2.
- Las gráficas llevan título, ejes con etiqueta en español y una frase de interpretación debajo.
- Al reportar porcentajes poblacionales, indicar si se usa o no el peso muestral `stu_wgt`.

## 7. Seguridad y repositorio

- `.gitignore` debe incluir `pisa_env/`, `.env`, `.streamlit/secrets.toml`, `__pycache__/` y `.DS_Store` **antes del primer commit**.
- Nunca escribir llaves API en el código.
- El repositorio es público: solo contiene datos anonimizados y redistribuibles.

## 8. Evidencias para la entrega (7 momentos clave)

Cada momento: título + captura verificable (terminal o URL local `localhost:8501`) + explicación de 80–120 palabras.

1. **Entorno y estructura:** terminal con `(pisa_env)` y árbol de carpetas con el CSV en `data/`.
2. **Primera ejecución:** navegador con título propio, vista previa y URL local.
3. **Entender los datos:** pestaña 1 con métricas, variables y tipos.
4. **Error o decisión técnica:** error y solución, o antes y después. Declarar uso de IA y qué se verificó.
5. **Limpiar y describir:** pestañas 2 y 3 con nulos tratados, gráfica por segmento y KPI.
6. **De los datos a la decisión:** pestañas 4 a 6 con predictores, acción por segmento y mensaje.
7. **Reflexión:** pestaña 6 final; tres límites (calidad, sesgos, privacidad) con propuestas, y un cambio hecho tras la revisión de un compañero.

## 9. Uso de IA

Este proyecto se desarrolla con apoyo de un asistente de IA. Todo código generado se ejecuta y se verifica localmente antes de aceptarlo, y lo verificado se documenta en el Momento 4.
