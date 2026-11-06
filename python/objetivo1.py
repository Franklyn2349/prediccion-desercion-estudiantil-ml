"""Objetivo específico 1: línea base con datos de matrícula (Escenario 1)."""
# Entrena y evalúa Decision Tree, Random Forest y XGBoost usando solo
# las variables disponibles al momento de la matrícula (COLUMNAS_E1).

# ---------- Librerías de datos y gráficos ----------
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Backend sin interfaz gráfica: permite guardar imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Modelos y evaluación ----------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

# ---------- Balanceo de clases ----------
# Se usa el Pipeline de imblearn (no el de sklearn) porque admite SMOTE como paso
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ---------- Módulos propios del proyecto ----------
import config
from carga_datos import cargar_datos    
from preprocesamiento import preparar_datos 

ESCENARIO = "E1"    # Identificador usado en nombres de archivos y tablas
CLASES = ["Abandono", "Graduado", "Matriculado"]    # Nota: no se usa en el script; las etiquetas se toman de le.classes_


def construir_modelos():
    """Los tres algoritmos con hiperparámetros por defecto."""
    # Todos usan la misma semilla para que los resultados sean reproducibles
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=config.SEMILLA),
        "Random Forest": RandomForestClassifier(random_state=config.SEMILLA),
        # mlogloss: función de pérdida para clasificación multiclase
        "XGBoost": XGBClassifier(random_state=config.SEMILLA, eval_metric="mlogloss"),
    }


def graficar_matriz(y_real, y_pred, nombre_modelo, etiquetas):
    """Guarda la matriz de confusión como imagen."""
    mc = confusion_matrix(y_real, y_pred) # Filas: clase real / Columnas: clase predicha
    plt.figure(figsize=(6, 5))
    # Mapa de calor con el número de casos en cada celda
    sns.heatmap(mc, annot=True, fmt="d", cmap="Reds",
                xticklabels=etiquetas, yticklabels=etiquetas)
    plt.title(f"Matriz de confusión - {nombre_modelo} ({ESCENARIO})")
    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.tight_layout()
    # Nombre del archivo: matriz_E1_Random_Forest.png (espacios reemplazados por "_")
    archivo = config.RUTA_RESULTADOS / f"matriz_{ESCENARIO}_{nombre_modelo.replace(' ', '_')}.png"
    plt.savefig(archivo, dpi=150)
    plt.close() # Libera memoria antes de generar la siguiente figura


def main():
    # ---------- 1. Carga y preparación de datos ----------
    df = cargar_datos()
    # Selecciona las columnas de E1 y divide en entrenamiento y prueba
    X_train, X_test, y_train, y_test = preparar_datos(
        df, config.COLUMNAS_E1, "Escenario 1"
    )

    # ---------- 2. Codificación de la variable objetivo ----------
    # XGBoost exige que el objetivo sea numérico
    le = LabelEncoder()
    y_train_cod = le.fit_transform(y_train) # Aprende la codificación solo con entrenamiento
    y_test_cod = le.transform(y_test)       # Aplica la misma codificación al test
    etiquetas = list(le.classes_)           # Nombres originales de las clases, en orden

    # ---------- 3. Configuración de la validación cruzada ----------
    # Estratificada: cada fold conserva la proporción de clases del conjunto original
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                         random_state=config.SEMILLA)
    resultados = [] # Métricas globales por modelo
    reportes = []   # Métricas por clase y por modelo

    # ---------- 4. Entrenamiento y evaluación de cada modelo ----------
    for nombre, modelo in construir_modelos().items():
        print(f"\n>>> {nombre}")

        # SMOTE dentro del pipeline: el sobremuestreo se aplica solo a los datos
        # de entrenamiento de cada fold, evitando fuga de información hacia la validación
        pipeline = Pipeline([
            ("smote", SMOTE(random_state=config.SEMILLA)),
            ("modelo", modelo),
        ])

        # Validación cruzada sobre el conjunto de entrenamiento
        f1_cv = cross_val_score(pipeline, X_train, y_train_cod,
                                cv=cv, scoring="f1_macro")

        # Entrenamiento final y evaluación sobre el test intacto
        pipeline.fit(X_train, y_train_cod)
        y_pred = pipeline.predict(X_test)   # SMOTE no se aplica al predecir

        # Métricas globales en el conjunto de prueba
        acc = accuracy_score(y_test_cod, y_pred)
        f1 = f1_score(y_test_cod, y_pred, average="macro")  # Promedio simple entre clases

        print(f"  F1-macro CV ({config.N_FOLDS} folds): {f1_cv.mean():.4f} (+/- {f1_cv.std():.4f})")
        print(f"  Accuracy test: {acc:.4f}")
        print(f"  F1-macro test: {f1:.4f}")

        # Guarda las métricas globales del modelo
        resultados.append({
            "Escenario": ESCENARIO,
            "Modelo": nombre,
            "Accuracy": round(acc, 4),
            "F1_macro": round(f1, 4),
            "F1_macro_CV": round(f1_cv.mean(), 4),
            "Desv_CV": round(f1_cv.std(), 4),   # Variabilidad entre folds
        })

        # Reporte detallado por clase (precision, recall y F1)
        # zero_division=0 evita advertencias si una clase no recibe predicciones
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

    # ---------- 5. Exportación de resultados ----------
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(config.RUTA_RESULTADOS / f"metricas_{ESCENARIO}.csv", index=False)

    tabla_clases = pd.DataFrame(reportes)
    tabla_clases.to_csv(config.RUTA_RESULTADOS / f"reporte_clases_{ESCENARIO}.csv", index=False)

    print("\n=== RESULTADOS ESCENARIO 1 ===")
    print(tabla.to_string(index=False))
    print("\nArchivos guardados en:", config.RUTA_RESULTADOS)

# Solo se ejecuta al correr este archivo directamente
if __name__ == "__main__":
    main()