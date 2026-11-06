"""Objetivo específico 2: matrícula + primer semestre (Escenario 2)."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import config
from carga_datos import cargar_datos
from preprocesamiento import preparar_datos

ESCENARIO = "E2"


def construir_modelos():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=config.SEMILLA),
        "Random Forest": RandomForestClassifier(random_state=config.SEMILLA),
        "XGBoost": XGBClassifier(random_state=config.SEMILLA, eval_metric="mlogloss"),
    }


def graficar_matriz(y_real, y_pred, nombre_modelo, etiquetas):
    mc = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(mc, annot=True, fmt="d", cmap="Reds",
                xticklabels=etiquetas, yticklabels=etiquetas)
    plt.title(f"Matriz de confusión - {nombre_modelo} ({ESCENARIO})")
    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.tight_layout()
    archivo = config.RUTA_RESULTADOS / f"matriz_{ESCENARIO}_{nombre_modelo.replace(' ', '_')}.png"
    plt.savefig(archivo, dpi=150)
    plt.close()


def main():
    df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(
        df, config.COLUMNAS_E2, "Escenario 2"
    )

    le = LabelEncoder()
    y_train_cod = le.fit_transform(y_train)
    y_test_cod = le.transform(y_test)
    etiquetas = list(le.classes_)

    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                         random_state=config.SEMILLA)
    resultados = []
    reportes = []

    for nombre, modelo in construir_modelos().items():
        print(f"\n>>> {nombre}")

        pipeline = Pipeline([
            ("smote", SMOTE(random_state=config.SEMILLA)),
            ("modelo", modelo),
        ])

        f1_cv = cross_val_score(pipeline, X_train, y_train_cod,
                                cv=cv, scoring="f1_macro")

        pipeline.fit(X_train, y_train_cod)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test_cod, y_pred)
        f1 = f1_score(y_test_cod, y_pred, average="macro")

        print(f"  F1-macro CV ({config.N_FOLDS} folds): {f1_cv.mean():.4f} (+/- {f1_cv.std():.4f})")
        print(f"  Accuracy test: {acc:.4f}")
        print(f"  F1-macro test: {f1:.4f}")

        resultados.append({
            "Escenario": ESCENARIO,
            "Modelo": nombre,
            "Accuracy": round(acc, 4),
            "F1_macro": round(f1, 4),
            "F1_macro_CV": round(f1_cv.mean(), 4),
            "Desv_CV": round(f1_cv.std(), 4),
        })

        rep = classification_report(y_test_cod, y_pred,
                                    target_names=etiquetas,
                                    output_dict=True, zero_division=0)
        for clase in etiquetas:
            reportes.append({
                "Escenario": ESCENARIO,
                "Modelo": nombre,
                "Clase": clase,
                "Precision": round(rep[clase]["precision"], 4),
                "Recall": round(rep[clase]["recall"], 4),
                "F1": round(rep[clase]["f1-score"], 4),
            })

        graficar_matriz(y_test_cod, y_pred, nombre, etiquetas)

    tabla = pd.DataFrame(resultados)
    tabla.to_csv(config.RUTA_RESULTADOS / f"metricas_{ESCENARIO}.csv", index=False)

    tabla_clases = pd.DataFrame(reportes)
    tabla_clases.to_csv(config.RUTA_RESULTADOS / f"reporte_clases_{ESCENARIO}.csv", index=False)

    print("\n=== RESULTADOS ESCENARIO 2 ===")
    print(tabla.to_string(index=False))
    print("\nArchivos guardados en:", config.RUTA_RESULTADOS)


if __name__ == "__main__":
    main()