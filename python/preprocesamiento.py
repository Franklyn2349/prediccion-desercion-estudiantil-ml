"""Preprocesamiento: codificación de variables y división train/test."""
# Prepara los datos de cualquier escenario (E1 o E2) antes del entrenamiento.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import config


def codificar_categoricas(X):
    """Convierte las columnas de texto a códigos numéricos."""
    X = X.copy()    # Copia para no modificar el DataFrame original
    # Toma solo las categóricas presentes en el escenario actual
    categoricas = [c for c in config.COLUMNAS_CATEGORICAS if c in X.columns]
    # Asigna un número entero a cada categoría (ej.: "Diurno" -> 0, "Nocturno" -> 1)
    # Se ajusta antes de la división train/test; solo mapea categorías a códigos,
    # sin usar información de la variable objetivo
    encoder = OrdinalEncoder()
    X[categoricas] = encoder.fit_transform(X[categoricas])
    return X


def preparar_datos(df, columnas, nombre_escenario=""):
    """Devuelve X_train, X_test, y_train, y_test para el escenario indicado."""
    # ---------- 1. Separación de predictores y objetivo ----------
    X = df[columnas].copy()         # Solo las variables del escenario indicado
    y = df[config.OBJETIVO].copy()  # Variable a predecir

    # ---------- 2. Codificación ----------
    X = codificar_categoricas(X)

    # ---------- 3. División entrenamiento / prueba ----------
    # stratify=y: ambos conjuntos mantienen la misma proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TAM_TEST,      # Usa el 20% definido en config
        random_state=config.SEMILLA,    # Siempre divide igual
        stratify=y,                     # Mantiene la proporción de clases
    )

    # ---------- 4. Resumen y controles ----------
    print(f"--- Preprocesamiento {nombre_escenario} ---")
    print("Variables utilizadas:", X.shape[1])
    # Control de fuga de información: ninguna variable del 2do semestre
    # debe estar en el modelo (el resultado esperado es 0)
    fuga = [c for c in X.columns if "2do sem" in c]
    print("Columnas de 2do semestre detectadas:", len(fuga))
    print("Entrenamiento:", X_train.shape[0], "| Prueba:", X_test.shape[0])
    # Confirma que la estratificación conservó el balance de clases
    print("Distribución en train:")
    print((y_train.value_counts(normalize=True) * 100).round(2).to_string())

    return X_train, X_test, y_train, y_test

# Prueba rápida del módulo con el Escenario 1 al ejecutarlo directamente
if __name__ == "__main__":
    from carga_datos import cargar_datos
    df = cargar_datos()
    preparar_datos(df, config.COLUMNAS_E1, "Escenario 1")