"""Objetivo específico 3: comparación entre escenarios E1 y E2."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def cargar_metricas():
    e1 = pd.read_csv(config.RUTA_RESULTADOS / "metricas_E1.csv")
    e2 = pd.read_csv(config.RUTA_RESULTADOS / "metricas_E2.csv")
    return e1, e2


def comparar_global(e1, e2):
    """Diferencia y mejora relativa por modelo."""
    comp = e1.merge(e2, on="Modelo", suffixes=("_E1", "_E2"))

    comp["Dif_Accuracy"] = (comp["Accuracy_E2"] - comp["Accuracy_E1"]).round(4)
    comp["Dif_F1_macro"] = (comp["F1_macro_E2"] - comp["F1_macro_E1"]).round(4)
    comp["Mejora_%"] = ((comp["Dif_F1_macro"] / comp["F1_macro_E1"]) * 100).round(2)

    columnas = ["Modelo", "Accuracy_E1", "Accuracy_E2", "Dif_Accuracy",
                "F1_macro_E1", "F1_macro_E2", "Dif_F1_macro", "Mejora_%"]
    return comp[columnas]


def comparar_por_clase():
    """Diferencia de F1 por clase entre escenarios."""
    c1 = pd.read_csv(config.RUTA_RESULTADOS / "reporte_clases_E1.csv")
    c2 = pd.read_csv(config.RUTA_RESULTADOS / "reporte_clases_E2.csv")

    comp = c1.merge(c2, on=["Modelo", "Clase"], suffixes=("_E1", "_E2"))
    comp["Dif_F1"] = (comp["F1_E2"] - comp["F1_E1"]).round(4)
    comp["Dif_Recall"] = (comp["Recall_E2"] - comp["Recall_E1"]).round(4)

    return comp[["Modelo", "Clase", "F1_E1", "F1_E2", "Dif_F1",
                 "Recall_E1", "Recall_E2", "Dif_Recall"]]


def graficar_comparacion(comp):
    """Barras agrupadas E1 vs E2 por modelo."""
    modelos = comp["Modelo"].tolist()
    x = np.arange(len(modelos))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - ancho/2, comp["F1_macro_E1"], ancho,
                label="E1 (matrícula)", color="#8FB8DE")
    b2 = ax.bar(x + ancho/2, comp["F1_macro_E2"], ancho,
                label="E2 (matrícula + 1er sem)", color="#2C5F8D")

    ax.bar_label(b1, fmt="%.3f", fontsize=9)
    ax.bar_label(b2, fmt="%.3f", fontsize=9)

    ax.set_ylabel("F1-macro")
    ax.set_title("Comparación de escenarios por modelo")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.set_ylim(0, 0.85)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.RUTA_RESULTADOS / "comparacion_escenarios.png", dpi=150)
    plt.close()


def main():
    e1, e2 = cargar_metricas()

    comp = comparar_global(e1, e2)
    comp.to_csv(config.RUTA_RESULTADOS / "comparacion_escenarios.csv", index=False)

    comp_clases = comparar_por_clase()
    comp_clases.to_csv(config.RUTA_RESULTADOS / "comparacion_por_clase.csv", index=False)

    graficar_comparacion(comp)

    print("=== COMPARACIÓN GLOBAL E1 vs E2 ===")
    print(comp.to_string(index=False))

    print("\n=== COMPARACIÓN POR CLASE ===")
    print(comp_clases.to_string(index=False))

    mejor = comp.loc[comp["F1_macro_E2"].idxmax()]
    print(f"\nMejor modelo en E2: {mejor['Modelo']} "
          f"(F1-macro {mejor['F1_macro_E2']:.4f})")
    print(f"Mayor mejora: {comp.loc[comp['Mejora_%'].idxmax(), 'Modelo']} "
          f"(+{comp['Mejora_%'].max():.2f}%)")

    print("\nArchivos guardados en:", config.RUTA_RESULTADOS)


if __name__ == "__main__":
    main()