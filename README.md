# Predicción de la deserción estudiantil con aprendizaje automático supervisado

Proyecto del curso **Taller de Investigación** (Ingeniería de Sistemas e Informática, Universidad Tecnológica del Perú).

Compara tres algoritmos de clasificación (**Decision Tree**, **Random Forest** y **XGBoost**) para predecir si un estudiante universitario **abandona**, **se gradúa** o **sigue matriculado**. Lo hace en dos escenarios, según la información disponible:

| Escenario | Información usada                                                                        | N.º de variables |
| --------- | ---------------------------------------------------------------------------------------- | :--------------: |
| **E1**    | Datos al momento de la matrícula (personales, familiares, de admisión y macroeconómicos) |        24        |
| **E2**    | Matrícula + rendimiento académico del primer semestre                                    |        30        |

Las 6 variables del segundo semestre se excluyen en ambos escenarios porque son posteriores al momento en que se busca intervenir.

## Autores

| Autor                       | GitHub                                           | LinkedIn                                                                   |
| --------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------- |
| Llanco Inga, Franklyn Ruben | [@Franklyn2349](https://github.com/Franklyn2349) | [franklynllanco](https://www.linkedin.com/in/franklynllanco)               |
| López Bernal, Edgar Piero   | [@Shidowar](https://github.com/Shidowar)         | [edgarpierolopezbernal](https://www.linkedin.com/in/edgarpierolopezbernal) |

## Dataset

Se usa el dataset público **Predict Students' Dropout and Academic Success** del UCI Machine Learning Repository: 4 424 estudiantes, 36 variables y una variable objetivo con tres clases.

> Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). _Predict Students' Dropout and Academic Success_ [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Artículo que describe el dataset:

> Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting Student Dropout and Academic Success. _Data, 7_(11), 146. https://doi.org/10.3390/data7110146

**Licencia del dataset:** [Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.es).

**Cambios realizados:** el archivo `data/raw/DataSets.xlsx` es una versión **traducida al español** del dataset original. Se tradujeron los nombres de las columnas y los valores de las variables categóricas, y la variable objetivo `Target` se renombró como `Resultado` (`Graduado`, `Abandono`, `Matriculado`). No se modificaron los registros.

## Estructura del proyecto

```
TallerInvestigacion_ML/
├── data/
│   └── raw/
│       └── DataSets.xlsx        # Dataset traducido (hoja "Datos")
├── python/
│   ├── config.py                # Rutas, semilla y listas de variables de E1 y E2
│   ├── carga_datos.py           # Carga y verificación del dataset
│   ├── preprocesamiento.py      # Codificación y división train/test
│   ├── objetivo1.py             # Objetivo específico 1: modelos con E1
│   ├── objetivo2.py             # Objetivo específico 2: modelos con E2
│   ├── objetivo3.py             # Objetivo específico 3: comparación E1 vs E2
│   └── graficos.py              # Figuras del informe
├── results/                     # Métricas (.csv) y figuras (.png) generadas
├── requirements.txt
└── README.md
```

## Instalación

Requiere **Python 3.12 o superior**.

```bash
git clone https://github.com/Franklyn2349/prediccion-desercion-estudiantil-ml.git
cd prediccion-desercion-estudiantil-ml
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Los scripts se ejecutan desde la carpeta `python/` y **en este orden**, porque `objetivo3.py` y `graficos.py` leen los CSV que generan los dos primeros:

```bash
cd python
python objetivo1.py
python objetivo2.py
python objetivo3.py
python graficos.py
```

Los resultados se guardan en `results/`. Para verificar solo el dataset y la configuración: `python carga_datos.py`.

## Metodología

- **Clasificación multiclase**: Graduado / Abandono / Matriculado (≈ 50 % / 32 % / 18 %).
- **Codificación ordinal** de las 17 variables categóricas.
- **División estratificada 80 % / 20 %** (3 539 entrenamiento / 885 prueba), con la misma partición en ambos escenarios.
- **SMOTE dentro del pipeline**: el sobremuestreo se aplica solo al entrenamiento (y a cada fold de la validación cruzada), nunca al conjunto de prueba.
- **Validación cruzada estratificada** de 5 folds sobre el entrenamiento.
- **Hiperparámetros por defecto** en los tres modelos, para aislar el efecto del escenario.
- **Semilla fija** (`42`) para que los resultados sean reproducibles.
- **Métricas**: F1-score macro (principal, por el desbalance de clases), Accuracy, F1 por clase, curvas ROC y pseudo-R² de McFadden.

## Resultados principales

Métricas sobre el conjunto de prueba (885 estudiantes):

| Modelo        | Accuracy E1 | Accuracy E2 | F1-macro E1 | F1-macro E2 | Mejora F1-macro |
| ------------- | :---------: | :---------: | :---------: | :---------: | :-------------: |
| Decision Tree |   0.5401    |   0.6305    |   0.4907    |   0.5718    |    +16.53 %     |
| Random Forest |   0.6497    | **0.7322**  |   0.5502    | **0.6587**  |  **+19.72 %**   |
| XGBoost       |   0.6181    |   0.7220    |   0.5412    |   0.6364    |    +17.59 %     |

- Agregar el rendimiento del primer semestre (E2) mejora a los tres modelos.
- **Random Forest** es el mejor modelo en E2 y también el que más mejora.
- Las variables con mayor importancia en E2 son las unidades curriculares aprobadas y la nota del primer semestre, junto con tener la matrícula al día.

**Pseudo-R² de McFadden** (`results/pseudo_r2.csv`): Random Forest pasa de 0.150 (E1) a 0.359 (E2) y XGBoost de 0.115 a 0.292. El Decision Tree da valores negativos porque, con hiperparámetros por defecto, asigna probabilidades de 0 o 1 y cada error se penaliza mucho.

## Uso de inteligencia artificial

Durante el desarrollo se utilizaron herramientas de IA generativa (Claude, de Anthropic) como apoyo para revisar el código, corregir comentarios, generar la figura del pseudo-R² y redactar este README. El diseño metodológico, el análisis y la interpretación de los resultados son responsabilidad de los autores.

## Licencia

El dataset original pertenece a sus autores y se distribuye bajo CC BY 4.0. Este repositorio incluye una versión traducida al español con la atribución indicada arriba.
