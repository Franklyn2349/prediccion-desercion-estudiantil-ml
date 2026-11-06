"""Objetivo específico 2: matrícula + primer semestre (Escenario 2)."""

# --- Librerías para manejo de datos y gráficos ---
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica: las figuras solo se guardan en archivo
import matplotlib.pyplot as plt
import seaborn as sns

# --- Modelos de clasificación, validación y métricas ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

# --- Balanceo de clases con SMOTE dentro de un pipeline ---
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# --- Módulos propios del proyecto ---
import config
from carga_datos import cargar_datos
from preprocesamiento import preparar_datos

# Identificador del escenario (se usa en nombres de archivos y títulos)
ESCENARIO = "E2"


def construir_modelos():
    """Devuelve un diccionario con los tres modelos a comparar."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=config.SEMILLA),
        "Random Forest": RandomForestClassifier(random_state=config.SEMILLA),
        # mlogloss: métrica de evaluación para clasificación multiclase
        "XGBoost": XGBClassifier(random_state=config.SEMILLA, eval_metric="mlogloss"),
    }


def graficar_matriz(y_real, y_pred, nombre_modelo, etiquetas):
    """Genera y guarda la matriz de confusión de un modelo como imagen PNG."""
    # Calcular la matriz de confusión
    mc = confusion_matrix(y_real, y_pred)

    # Dibujar el mapa de calor con los conteos en cada celda
    plt.figure(figsize=(6, 5))
    sns.heatmap(mc, annot=True, fmt="d", cmap="Reds",
                xticklabels=etiquetas, yticklabels=etiquetas)
    plt.title(f"Matriz de confusión - {nombre_modelo} ({ESCENARIO})")
    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.tight_layout()

    # Guardar la figura en la carpeta de resultados y liberar memoria
    archivo = config.RUTA_RESULTADOS / f"matriz_{ESCENARIO}_{nombre_modelo.replace(' ', '_')}.png"
    plt.savefig(archivo, dpi=150)
    plt.close()


def main():
    # --- 1. Carga y preparación de datos ---
    # Se usan las columnas del Escenario 2 (matrícula + primer semestre)
    df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(
        df, config.COLUMNAS_E2, "Escenario 2"
    )

    # --- 2. Codificación de la variable objetivo ---
    # Convierte las clases de texto a números (requerido por XGBoost)
    le = LabelEncoder()
    y_train_cod = le.fit_transform(y_train)
    y_test_cod = le.transform(y_test)
    etiquetas = list(le.classes_)  # Nombres originales de las clases

    # --- 3. Configuración de la validación cruzada estratificada ---
    # Mantiene la proporción de clases en cada fold
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                         random_state=config.SEMILLA)
    resultados = []  # Métricas generales por modelo
    reportes = []    # Métricas detalladas por clase

    # --- 4. Entrenamiento y evaluación de cada modelo ---
    for nombre, modelo in construir_modelos().items():
        print(f"\n>>> {nombre}")

        # Pipeline: SMOTE balancea solo los datos de entrenamiento de cada fold,
        # evitando fuga de información hacia los datos de validación
        pipeline = Pipeline([
            ("smote", SMOTE(random_state=config.SEMILLA)),
            ("modelo", modelo),
        ])

        # Validación cruzada sobre el conjunto de entrenamiento (F1-macro)
        f1_cv = cross_val_score(pipeline, X_train, y_train_cod,
                                cv=cv, scoring="f1_macro")

        # Entrenamiento final con todo el train y predicción sobre test
        pipeline.fit(X_train, y_train_cod)
        y_pred = pipeline.predict(X_test)

        # Métricas en el conjunto de prueba
        acc = accuracy_score(y_test_cod, y_pred)
        f1 = f1_score(y_test_cod, y_pred, average="macro")

        # Mostrar resultados en consola
        print(f"  F1-macro CV ({config.N_FOLDS} folds): {f1_cv.mean():.4f} (+/- {f1_cv.std():.4f})")
        print(f"  Accuracy test: {acc:.4f}")
        print(f"  F1-macro test: {f1:.4f}")

        # Guardar métricas generales del modelo
        resultados.append({
            "Escenario": ESCENARIO,
            "Modelo": nombre,
            "Accuracy": round(acc, 4),
            "F1_macro": round(f1, 4),
            "F1_macro_CV": round(f1_cv.mean(), 4),
            "Desv_CV": round(f1_cv.std(), 4),
        })

        # Reporte de clasificación: precisión, recall y F1 por cada clase
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

        # Generar la matriz de confusión del modelo
        graficar_matriz(y_test_cod, y_pred, nombre, etiquetas)

    # --- 5. Exportación de resultados a CSV ---
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(config.RUTA_RESULTADOS / f"metricas_{ESCENARIO}.csv", index=False)

    tabla_clases = pd.DataFrame(reportes)
    tabla_clases.to_csv(config.RUTA_RESULTADOS / f"reporte_clases_{ESCENARIO}.csv", index=False)

    # --- 6. Resumen final en consola ---
    print("\n=== RESULTADOS ESCENARIO 2 ===")
    print(tabla.to_string(index=False))
    print("\nArchivos guardados en:", config.RUTA_RESULTADOS)


# Punto de entrada: ejecuta main() solo si el archivo se corre directamente
if __name__ == "__main__":
    main()
