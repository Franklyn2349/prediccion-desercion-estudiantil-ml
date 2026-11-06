"""Figuras del informe.

Genera seis figuras:
  1. Distribución de clases del dataset       (sección de datos)
  2. Esquema del diseño experimental          (sección de metodología)
  3. Importancia de variables (RF y XGBoost)  (resultados)
  4. F1 por clase, E1 vs E2                   (resultados)
  5. Curvas ROC multiclase                    (resultados)
  6. Pseudo-R² de McFadden, E1 vs E2          (resultados)

Las figuras 3, 5 y 6 reentrenan los modelos internamente. Como la semilla
está fija en config.py, son idénticos a los de objetivo1.py y objetivo2.py.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib import rcParams

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, log_loss
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import config
from carga_datos import cargar_datos
from preprocesamiento import preparar_datos


# ===============================================================
# Identidad visual del informe
# ===============================================================
COLOR_CLASE = {
    "Abandono":    "#C1443C",   # terracota
    "Graduado":    "#1F6F78",   # verde azulado
    "Matriculado": "#D99A2B",   # ámbar
}

COLOR_E1 = "#B4D4A8"        # verde claro -> Escenario 1
COLOR_E2 = "#FB0000"        # rojo intenso -> Escenario 2

AZUL_PROFUNDO = "#1D3F66"
AZUL_MEDIO = "#2C5F8D"
VERDE = "#1F6F78"
ROJO = "#C1443C"
GRIS_TEXTO = "#333333"
GRIS_SUAVE = "#DDDDDD"
GRIS_EXCLUIDO = "#9A9A9A"

rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.edgecolor"] = GRIS_SUAVE
rcParams["axes.labelcolor"] = GRIS_TEXTO
rcParams["text.color"] = GRIS_TEXTO
rcParams["xtick.color"] = GRIS_TEXTO
rcParams["ytick.color"] = GRIS_TEXTO
rcParams["axes.titlesize"] = 13
rcParams["axes.labelsize"] = 11
rcParams["figure.facecolor"] = "white"
rcParams["axes.facecolor"] = "white"


def estilizar(ax, grid_eje="y"):
    """Quita bordes superiores/derechos y agrega rejilla sutil."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_eje, color=GRIS_SUAVE, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def guardar(nombre):
    plt.savefig(config.RUTA_RESULTADOS / nombre, dpi=200, bbox_inches="tight")
    plt.close()
    print("  Guardado:", nombre)


# ===============================================================
# Entrenamiento auxiliar (figuras 3, 5 y 6)
# ===============================================================
def entrenar_escenario(df, columnas, nombre):
    X_train, X_test, y_train, y_test = preparar_datos(df, columnas, nombre)

    le = LabelEncoder()
    y_train_cod = le.fit_transform(y_train)
    y_test_cod = le.transform(y_test)

    # Mismos modelos e hiperparámetros que objetivo1.py y objetivo2.py
    modelos = {
        "Decision Tree": DecisionTreeClassifier(random_state=config.SEMILLA),
        "Random Forest": RandomForestClassifier(random_state=config.SEMILLA),
        "XGBoost": XGBClassifier(random_state=config.SEMILLA,
                                 eval_metric="mlogloss"),
    }

    entrenados = {}
    for nombre_modelo, modelo in modelos.items():
        pipeline = Pipeline([
            ("smote", SMOTE(random_state=config.SEMILLA)),
            ("modelo", modelo),
        ])
        pipeline.fit(X_train, y_train_cod)
        entrenados[nombre_modelo] = pipeline

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train_cod": y_train_cod, "y_test_cod": y_test_cod,
        "etiquetas": list(le.classes_),
        "modelos": entrenados,
    }


# ===============================================================
# FIGURA 1 — Distribución de clases
# ===============================================================
def grafico_distribucion(df):
    conteo = df[config.OBJETIVO].value_counts()
    total = conteo.sum()

    orden = ["Graduado", "Abandono", "Matriculado"]
    valores = [conteo[c] for c in orden]
    porcentajes = [v / total * 100 for v in valores]
    colores = [COLOR_CLASE[c] for c in orden]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    barras = ax.bar(orden, valores, color=colores, width=0.6,
                    edgecolor="white", linewidth=1.5)

    for barra, valor, pct in zip(barras, valores, porcentajes):
        centro = barra.get_x() + barra.get_width() / 2
        ax.annotate(f"{valor:,}".replace(",", " "), xy=(centro, valor),
                    xytext=(0, 22), textcoords="offset points",
                    ha="center", fontsize=13, fontweight="bold")
        ax.annotate(f"{pct:.1f} %", xy=(centro, valor),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=11, color=GRIS_TEXTO)

    equilibrio = total / 3
    ax.axhline(equilibrio, linestyle="--", linewidth=1.3, color="#888888")
    ax.annotate("Reparto equilibrado (33.3 %)", xy=(2.42, equilibrio),
                xytext=(0, 9), textcoords="offset points",
                ha="right", fontsize=9, color="#666666", style="italic")

    ax.set_ylabel("N.º de estudiantes")
    ax.set_ylim(0, max(valores) * 1.22)
    ax.set_title(f"Distribución de la variable objetivo  (n = {total:,})".replace(",", " "),
                 fontsize=14, fontweight="bold", pad=18)
    ax.tick_params(axis="x", labelsize=12)
    estilizar(ax)

    fig.text(0.5, -0.03,
             "El desbalance entre clases motiva el uso de SMOTE en entrenamiento "
             "y de F1-score macro como métrica principal.",
             ha="center", fontsize=9.5, color="#555555", style="italic")

    guardar("distribucion_clases.png")


# ===============================================================
# FIGURA 2 — Esquema del diseño experimental
# ===============================================================
def _caja(ax, x, y, ancho, alto, texto, color_fondo, color_texto="white",
          tam=10, negrita=True, borde=None, estilo_borde="solid"):
    ax.add_patch(FancyBboxPatch(
        (x, y), ancho, alto,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=color_fondo,
        edgecolor=borde if borde else color_fondo,
        linewidth=1.6 if borde else 0,
        linestyle=estilo_borde,
    ))
    ax.text(x + ancho / 2, y + alto / 2, texto,
            ha="center", va="center", fontsize=tam, color=color_texto,
            fontweight="bold" if negrita else "normal", linespacing=1.6)


def _flecha(ax, x1, y1, x2, y2, color="#808080", ancho=1.5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=ancho, color=color, shrinkA=0, shrinkB=0,
    ))


def grafico_diseno():
    fig, ax = plt.subplots(figsize=(13.5, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---- Nivel 1: dataset ----------------------------------
    _caja(ax, 2.30, 9.15, 5.40, 0.62,
          "Dataset  —  4 424 estudiantes  ×  36 variables",
          AZUL_PROFUNDO, tam=11.5)

    # ---- Nivel 2: bloques de variables ---------------------
    cx = [1.80, 5.00, 8.20]          # centros
    _caja(ax, cx[0] - 1.42, 7.45, 2.84, 1.15,
          "Matrícula\n24 variables", AZUL_MEDIO, tam=10.5)
    _caja(ax, cx[1] - 1.42, 7.45, 2.84, 1.15,
          "Primer semestre\n6 variables", ROJO, tam=10.5)
    _caja(ax, cx[2] - 1.42, 7.45, 2.84, 1.15,
          "Segundo semestre\n6 variables  ·  EXCLUIDAS\nposteriores a la intervención",
          "white", color_texto=GRIS_EXCLUIDO, tam=9.5,
          borde=GRIS_EXCLUIDO, estilo_borde="dashed")

    _flecha(ax, 4.10, 9.13, 2.00, 8.63)
    _flecha(ax, 5.00, 9.13, 5.00, 8.63)
    _flecha(ax, 5.90, 9.13, 8.00, 8.63)

    # ---- Nivel 3: escenarios -------------------------------
    e1x, e2x = 2.60, 7.40
    _caja(ax, e1x - 1.75, 6.10, 3.50, 0.95,
          "ESCENARIO 1\n24 variables  (matrícula)",
          COLOR_E1, color_texto=AZUL_PROFUNDO, tam=11)
    _caja(ax, e2x - 1.75, 6.10, 3.50, 0.95,
          "ESCENARIO 2\n30 variables  (matrícula + 1.er sem)",
          COLOR_E2, tam=11)

    _flecha(ax, 1.90, 7.43, 2.35, 7.08, AZUL_MEDIO)      # matrícula -> E1
    _flecha(ax, 3.00, 7.43, 6.60, 7.08, AZUL_MEDIO)      # matrícula -> E2
    _flecha(ax, 5.40, 7.43, 7.55, 7.08, ROJO)            # 1er sem   -> E2

    # ---- Nivel 4: tratamiento común ------------------------
    ax.add_patch(FancyBboxPatch(
        (0.45, 1.45), 9.10, 3.60,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#F5F8FA", edgecolor=GRIS_SUAVE, linewidth=1.4,
    ))

    # Ambos escenarios convergen en el mismo pipeline
    ax.add_line(Line2D([e1x, e2x], [5.55, 5.55], color="#808080", linewidth=1.5))
    ax.add_line(Line2D([e1x, e1x], [6.08, 5.55], color="#808080", linewidth=1.5))
    ax.add_line(Line2D([e2x, e2x], [6.08, 5.55], color="#808080", linewidth=1.5))
    _flecha(ax, 5.00, 5.55, 5.00, 4.88)

    # Pasos del pipeline, encadenados
    pasos = [
        "Codificación ordinal\nde categóricas",
        "División estratificada\n80 % / 20 %",
        "SMOTE\nsolo en entrenamiento",
        "Validación cruzada\nestratificada (5 folds)",
    ]
    ancho_p, sep_p = 2.00, 0.26
    x0 = 5.00 - (len(pasos) * ancho_p + (len(pasos) - 1) * sep_p) / 2
    centros_p = []
    for i, paso in enumerate(pasos):
        x = x0 + i * (ancho_p + sep_p)
        centros_p.append(x + ancho_p / 2)
        _caja(ax, x, 3.90, ancho_p, 0.95, paso, AZUL_MEDIO,
              tam=8.8, negrita=False)
        if i < len(pasos) - 1:
            _flecha(ax, x + ancho_p + 0.02, 4.38, x + ancho_p + sep_p - 0.02, 4.38)

    # Del pipeline a los tres modelos (bus de distribución)
    modelos = ["Decision Tree", "Random Forest", "XGBoost"]
    ancho_m, sep_m = 2.30, 0.34
    x0m = 5.00 - (len(modelos) * ancho_m + (len(modelos) - 1) * sep_m) / 2
    centros_m = [x0m + i * (ancho_m + sep_m) + ancho_m / 2
                 for i in range(len(modelos))]

    ax.add_line(Line2D([5.00, 5.00], [3.88, 3.52], color="#808080", linewidth=1.5))
    ax.add_line(Line2D([centros_m[0], centros_m[-1]], [3.52, 3.52],
                       color="#808080", linewidth=1.5))
    for i, modelo in enumerate(modelos):
        x = x0m + i * (ancho_m + sep_m)
        _caja(ax, x, 2.40, ancho_m, 0.72, modelo, "white",
              color_texto=AZUL_PROFUNDO, tam=10, borde=AZUL_PROFUNDO)
        _flecha(ax, centros_m[i], 3.52, centros_m[i], 3.16)

    # ---- Nivel 5: evaluación -------------------------------
    _caja(ax, 1.20, 0.50, 7.60, 0.70,
          "Evaluación sobre el conjunto de prueba  —  885 estudiantes, "
          "distribución real sin SMOTE",
          VERDE, tam=10.5)

    for c in centros_m:
        _flecha(ax, c, 2.38, c, 1.23)

    # Notas al pie del bloque común (se dibujan al final para quedar
    # por encima de las flechas, con fondo del mismo color del bloque)
    fondo = dict(facecolor="#F5F8FA", edgecolor="none", pad=3)
    ax.text(5.00, 2.08,
            "Tratamiento idéntico en ambos escenarios  ·  semilla fija = 42  ·  "
            "hiperparámetros por defecto",
            ha="center", va="center", fontsize=9.5, color="#555555",
            fontweight="bold", bbox=fondo)
    ax.text(5.00, 1.75,
            "La única diferencia entre escenarios son las 6 variables del primer semestre",
            ha="center", va="center", fontsize=9, color="#666666",
            style="italic", bbox=fondo)

    ax.text(5.00, 0.15,
            "Métricas:  Accuracy  ·  F1-score macro  ·  F1 por clase",
            ha="center", fontsize=9.5, color="#555555")

    ax.set_title("Diseño experimental del estudio",
                 fontsize=15, fontweight="bold", pad=14)

    guardar("diseno_experimental.png")


# ===============================================================
# FIGURA 3 — Importancia de variables
# ===============================================================
def grafico_importancias(datos, top=15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Solo modelos de ensamble (el Decision Tree no se muestra en esta figura)
    for ax, nombre_modelo in zip(axes, ["Random Forest", "XGBoost"]):
        pipeline = datos["modelos"][nombre_modelo]
        modelo = pipeline.named_steps["modelo"]
        y_pred = pipeline.predict(datos["X_test"])
        accuracy = accuracy_score(datos["y_test_cod"], y_pred)
        f1 = f1_score(datos["y_test_cod"], y_pred, average="macro")
        importancias = pd.Series(
            modelo.feature_importances_,
            index=datos["X_train"].columns
        ).sort_values(ascending=False).head(top).sort_values()

        colores = [ROJO if "1er sem" in v else AZUL_PROFUNDO
                   for v in importancias.index]

        barras = ax.barh(range(len(importancias)), importancias.values,
                         color=colores, height=0.7)
        ax.set_yticks(range(len(importancias)))
        ax.set_yticklabels(
            [v.replace("Unidades curriculares 1er sem ", "1er sem ")
             for v in importancias.index], fontsize=9)
        ax.bar_label(barras, fmt="%.3f", fontsize=8, padding=3)
        ax.set_xlabel("Importancia relativa")
        ax.set_title(nombre_modelo, fontweight="bold", pad=28)
        ax.text(0.5, 1.02,
                f"Accuracy = {accuracy:.4f}  |  F1-macro = {f1:.4f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
        ax.set_xlim(0, importancias.max() * 1.18)
        estilizar(ax, grid_eje="x")

    fig.suptitle(f"Importancia de variables — Escenario 2 (top {top})",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005,
             "En rojo: variables académicas del primer semestre   |   "
             "En azul: variables de matrícula",
             ha="center", fontsize=9, color=GRIS_TEXTO)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    guardar("importancia_variables_E2.png")


# ===============================================================
# FIGURA 4 — F1 por clase, E1 vs E2
# ===============================================================
def grafico_clases(modelo_objetivo="Random Forest"):
    c1 = pd.read_csv(config.RUTA_RESULTADOS / "reporte_clases_E1.csv")
    c2 = pd.read_csv(config.RUTA_RESULTADOS / "reporte_clases_E2.csv")
    c1 = c1[c1["Modelo"] == modelo_objetivo].set_index("Clase")
    c2 = c2[c2["Modelo"] == modelo_objetivo].set_index("Clase")

    clases = ["Matriculado", "Graduado", "Abandono"]
    f1_e1 = [c1.loc[c, "F1"] for c in clases]
    f1_e2 = [c2.loc[c, "F1"] for c in clases]

    x = np.arange(len(clases))
    ancho = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - ancho/2, f1_e1, ancho, label="E1 — matrícula",
                color=COLOR_E1, edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + ancho/2, f1_e2, ancho, label="E2 — matrícula + 1er semestre",
                color=COLOR_E2, edgecolor="white", linewidth=1.2)

    ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2)

    for i, (v1, v2) in enumerate(zip(f1_e1, f1_e2)):
        ax.annotate(f"+{v2 - v1:.3f}", xy=(x[i], max(v1, v2) + 0.07),
                    ha="center", fontsize=10, fontweight="bold",
                    color=COLOR_CLASE[clases[i]])

    ax.set_xticks(x)
    ax.set_xticklabels(clases, fontsize=11)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"F1-score por clase — {modelo_objetivo}",
                 fontweight="bold", pad=15)
    ax.legend(frameon=False, loc="upper right")
    estilizar(ax)
    guardar("f1_por_clase_comparacion.png")


# ===============================================================
# FIGURA 5 — Curvas ROC multiclase
# ===============================================================
def grafico_roc(datos, escenario, modelo_objetivo="Random Forest"):
    pipeline = datos["modelos"][modelo_objetivo]
    etiquetas = datos["etiquetas"]
    y_bin = label_binarize(datos["y_test_cod"], classes=range(len(etiquetas)))
    y_proba = pipeline.predict_proba(datos["X_test"])

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, clase in enumerate(etiquetas):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, linewidth=2.4, color=COLOR_CLASE[clase],
                label=f"{clase}  (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2,
            color="#999999", label="Clasificador aleatorio")

    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curvas ROC por clase — {modelo_objetivo} (Escenario {escenario[1:]})",
                 fontweight="bold", pad=15)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    estilizar(ax, grid_eje="both")
    guardar(f"curvas_roc_{escenario}.png")


# ===============================================================
# FIGURA 6 — Pseudo-R² de McFadden
# ===============================================================
def calcular_pseudo_r2(datos):
    """Pseudo-R² de McFadden de cada modelo sobre el conjunto de prueba

    pseudo-R² = 1 - (log-loss del modelo / log-loss del modelo nulo)

    El modelo nulo asigna a todos los estudiantes las proporciones de clase
    del entrenamiento (no usa ninguna variable). Un valor de 0 equivale al
    modelo nulo; valores más altos indican mejores probabilidades; un valor
    negativo indica que el modelo predice peor que el modelo nulo.
    """
    y_test = datos["y_test_cod"]
    clases = list(range(len(datos["etiquetas"])))

    # Log-loss del modelo nulo
    proporciones = np.bincount(datos["y_train_cod"]) / len(datos["y_train_cod"])
    ll_nulo = log_loss(y_test, np.tile(proporciones, (len(y_test), 1)),
                       labels=clases)

    resultados = {}
    for nombre_modelo, pipeline in datos["modelos"].items():
        ll_modelo = log_loss(y_test, pipeline.predict_proba(datos["X_test"]),
                             labels=clases)
        resultados[nombre_modelo] = 1 - ll_modelo / ll_nulo
    return resultados


def grafico_pseudo_r2(datos_e1, datos_e2):
    r2_e1 = calcular_pseudo_r2(datos_e1)
    r2_e2 = calcular_pseudo_r2(datos_e2)
    modelos = list(r2_e1.keys())

    # Tabla para citar los valores en el informe
    pd.DataFrame({
        "Modelo": modelos,
        "Pseudo_R2_E1": [round(r2_e1[m], 4) for m in modelos],
        "Pseudo_R2_E2": [round(r2_e2[m], 4) for m in modelos],
    }).to_csv(config.RUTA_RESULTADOS / "pseudo_r2.csv", index=False)

    # Límite inferior del eje: los valores más bajos se recortan y se
    # rotulan con su valor real (el Decision Tree sale muy negativo)
    y_min, y_max = -0.15, 0.5
    x = np.arange(len(modelos))
    ancho = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))
    for desplazamiento, valores, color in [
        (-ancho / 2, r2_e1, COLOR_E1),
        (+ancho / 2, r2_e2, COLOR_E2),
    ]:
        for i, m in enumerate(modelos):
            valor = valores[m]
            recortado = valor < y_min
            altura = y_min if recortado else valor
            ax.bar(x[i] + desplazamiento, altura, ancho, color=color,
                   edgecolor="white", linewidth=1.2,
                   hatch="//" if recortado else None)
            if recortado:
                ax.text(x[i] + desplazamiento, y_min / 2, f"{valor:.2f}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=GRIS_TEXTO,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
            else:
                ax.annotate(f"{valor:.3f}", xy=(x[i] + desplazamiento, valor),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=9)

    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=11)
    ax.set_ylabel("Pseudo-R² de McFadden")
    ax.set_ylim(y_min, y_max)
    ax.set_title("Pseudo-R² de McFadden por modelo — E1 vs E2",
                 fontweight="bold", pad=15)
    # Leyenda con colores lisos (sin el rayado de las barras recortadas)
    ax.legend(handles=[Patch(color=COLOR_E1, label="E1 — matrícula"),
                       Patch(color=COLOR_E2, label="E2 — matrícula + 1er semestre")],
              frameon=False, loc="upper left")
    estilizar(ax)

    fig.text(0.5, -0.04,
             "Pseudo-R² = 1 − (log-loss del modelo / log-loss del modelo nulo), "
             "calculado sobre el conjunto de prueba.\n"
             "Barras rayadas: valor fuera de escala. El Decision Tree con "
             "hiperparámetros por defecto asigna probabilidades de 0 o 1,\n"
             "por lo que cada error se penaliza mucho y su pseudo-R² resulta negativo.",
             ha="center", fontsize=8.5, color="#555555", style="italic")

    guardar("pseudo_r2_comparacion.png")


# ===============================================================
def main():
    df = cargar_datos()

    print("\nGenerando figuras:")
    grafico_distribucion(df)
    grafico_diseno()

    datos_e1 = entrenar_escenario(df, config.COLUMNAS_E1, "Escenario 1")
    grafico_roc(datos_e1, "E1", "Random Forest")

    datos_e2 = entrenar_escenario(df, config.COLUMNAS_E2, "Escenario 2")
    grafico_importancias(datos_e2)
    grafico_clases("Random Forest")
    grafico_roc(datos_e2, "E2", "Random Forest")
    grafico_pseudo_r2(datos_e1, datos_e2)

    print("\nFiguras guardadas en:", config.RUTA_RESULTADOS)


if __name__ == "__main__":
    main()