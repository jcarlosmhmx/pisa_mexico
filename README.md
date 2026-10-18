# 📚 PISA México: riesgo en matemáticas

App de Streamlit que lleva los datos de PISA 2022 de México desde **entender** hasta **decidir**:

> **¿Qué alumnos tienen mayor riesgo de no alcanzar el nivel mínimo de competencia en matemáticas y qué apoyo deberíamos ofrecer a cada grupo?**

Actividad F05 – Ciencia de Datos, Maestría en Big Data e Inteligencia Artificial, Universidad Anáhuac.

## Las seis pestañas

| # | Pestaña | Pregunta que responde |
|---|---|---|
| 1 | Entender | ¿Quiénes son los alumnos y qué información tenemos? |
| 2 | Limpiar | ¿Qué tan confiables son los datos? |
| 3 | Descriptiva | ¿Cómo se distribuye el desempeño y qué segmentos existen? |
| 4 | Predictiva | ¿Podemos predecir quién no alcanzará el Nivel 2? (regresión logística) |
| 5 | Prescriptiva | ¿Qué apoyo dar a cada nivel de riesgo? |
| 6 | Decisiones | ¿Qué mensaje mostrar a cada grupo y por qué? |

## Datos

- **Archivo:** `data/pisa_2022_mexico.csv` – 6,288 alumnos de 15 años, 280 escuelas, 23 variables.
- **Fuente:** Programme for International Student Assessment (PISA), Organisation for Economic Co-operation and Development (OECD), Paris. Microdatos 2022 en la versión depurada del paquete abierto [learningtower](https://github.com/kevinwang09/learningtower) (licencia MIT), filtrados a México.
- **Variable resultado:** `alcanza_nivel2_mat` = 1 si el puntaje en matemáticas es ≥ 420.07 (umbral oficial del Nivel 2 de la OCDE).
- **Anonimización:** solo contiene identificadores numéricos de alumno y escuela; no hay nombres, correos ni teléfonos.
- **¿Por qué 2022 y no 2025?** Los microdatos de PISA 2025 requieren registro y sus términos prohíben redistribuirlos; este repositorio es público. Los resultados de 2025 se citan solo como contexto agregado.

## Cómo ejecutarla

```bash
git clone https://github.com/jcarlosmhmx/pisa_mexico.git
cd pisa_mexico
python3 -m venv pisa_env
source pisa_env/bin/activate        # Windows: pisa_env\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```
pisa_mexico/
├── app.py                 # Configuración, carga de datos y pestañas
├── data/
│   └── pisa_2022_mexico.csv
├── tabs/
│   ├── comun.py           # Segmentos, modelo y plan de acción compartidos
│   ├── entender.py
│   ├── limpiar.py
│   ├── descriptiva.py
│   ├── predictiva.py
│   ├── prescriptiva.py
│   └── decisiones.py
├── project.md             # Contexto del proyecto para trabajar con agentes
├── requirements.txt
└── .gitignore
```

## Límites

- El modelo (AUC ≈ 0.68) sirve para **priorizar grupos**, no para etiquetar a un alumno.
- Las variables muestran **asociación, no causalidad**.
- PISA estima varios valores plausibles por alumno; aquí se usa uno solo.
- Los datos son de 2022; en PISA 2025 México bajó a 388 puntos en matemáticas.

## Uso de IA

El código se desarrolló con apoyo de un asistente de IA. Todo el código se ejecutó y verificó localmente; los errores encontrados y corregidos se documentan en la entrega.
