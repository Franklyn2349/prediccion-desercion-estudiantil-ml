"""Configuración central del proyecto."""
# Todos los módulos importan este archivo para usar los mismos parámetros,
# rutas y listas de variables, evitando valores repetidos en el código.
from pathlib import Path

# ---------- Rutas ----------
# Carpeta raíz del proyecto (dos niveles arriba de este archivo)
RAIZ = Path(__file__).resolve().parent.parent
RUTA_EXCEL = RAIZ / "data" / "raw" / "DataSets.xlsx"    # Dataset original sin procesar
HOJA = "Datos"                                          # Hoja del Excel que contiene los registros
RUTA_RESULTADOS = RAIZ / "results"                      # Carpeta de salida (métricas, gráficos, etc.)
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)      # La crea si no existe

# ---------- Reproducibilidad ----------
SEMILLA = 42    # Semilla fija para que los resultados sean replicables
TAM_TEST = 0.20 # 20% de los datos se reserva para prueba (80% entrenamiento)
N_FOLDS = 5     # Número de particiones para la validación cruzada

# ---------- Variable objetivo ----------
OBJETIVO = "Resultado"  # Columna que el modelo debe predecir

# ---------- Escenario 1: datos de matrícula ----------
# Variables disponibles al momento de la matrícula
# (datos personales, familiares, de admisión y macroeconómicos)
COLUMNAS_E1 = [
    "Estado civil",
    "Modalidad de admisión",
    "Orden de solicitud",
    "Carrera",
    "Turno",
    "Calificación previa",
    "Nota de calificación previa",
    "Nacionalidad",
    "Calificación de la madre",
    "Calificación del padre",
    "Ocupación de la madre",
    "Ocupación del padre",
    "Nota de admisión",
    "Desplazado",
    "Necesidades educativas especiales",
    "Deudor",
    "Matrícula al día",
    "Género",
    "Becario",
    "Edad al matricularse",
    "Internacional",
    "Tasa de desempleo",
    "Tasa de inflación",
    "PIB",
]

# ---------- Variables académicas del primer semestre ----------
# Indicadores de rendimiento obtenidos al finalizar el primer semestre
COLUMNAS_1ER_SEM = [
    "Unidades curriculares 1er sem (acreditadas)",
    "Unidades curriculares 1er sem (matriculadas)",
    "Unidades curriculares 1er sem (evaluaciones)",
    "Unidades curriculares 1er sem (aprobadas)",
    "Unidades curriculares 1er sem (nota)",
    "Unidades curriculares 1er sem (sin evaluación)",
]

# ---------- Escenario 2: matrícula + primer semestre ----------
# E2 reutiliza todas las variables de E1 y les suma las del primer semestre
COLUMNAS_E2 = COLUMNAS_E1 + COLUMNAS_1ER_SEM

# ---------- Columnas categóricas (requieren codificación) ----------
# Variables cuyos valores son texto (categorías, no cantidades);
# por eso deben codificarse a números antes de entrenar los modelos
COLUMNAS_CATEGORICAS = [
    "Estado civil",
    "Modalidad de admisión",
    "Carrera",
    "Turno",
    "Calificación previa",
    "Nacionalidad",
    "Calificación de la madre",
    "Calificación del padre",
    "Ocupación de la madre",
    "Ocupación del padre",
    "Desplazado",
    "Necesidades educativas especiales",
    "Deudor",
    "Matrícula al día",
    "Género",
    "Becario",
    "Internacional",
]

# ---------- Verificación ----------
# Solo se ejecuta al correr este archivo directamente:
# confirma que la ruta del Excel es correcta y muestra cuántas variables tiene cada escenario
if __name__ == "__main__":
    print("Ruta del Excel:", RUTA_EXCEL)
    print("¿El archivo existe?:", RUTA_EXCEL.exists())
    print("Variables E1:", len(COLUMNAS_E1))
    print("Variables E2:", len(COLUMNAS_E2))