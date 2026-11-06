"""Carga y verificación del dataset."""
import pandas as pd
import config # Rutas, nombre de hoja, variable objetivo y listas de columnas


def cargar_datos():
    """Lee el Excel y devuelve el DataFrame completo."""
    # Lee la hoja definida en config usando openpyxl (motor para archivos .xlsx)
    df = pd.read_excel(config.RUTA_EXCEL, sheet_name=config.HOJA, engine="openpyxl")
    return df


def verificar_datos(df):
    """Comprueba la integridad del dataset cargado."""
    # --- Revisión general: tamaño, duplicados y valores faltantes ---
    print("Dimensiones:", df.shape)                         # (filas, columnas)
    print("Filas duplicadas:", df.duplicated().sum())       # Registros repetidos completos
    print("Valores nulos totales:", df.isna().sum().sum())  # Nulos en todo el dataset

    # --- Balance de clases de la variable objetivo ---
    # Permite identificar desbalance (justifica el uso de SMOTE más adelante)
    print("\nDistribución de la variable objetivo:")
    conteo = df[config.OBJETIVO].value_counts()     # Frecuencia absoluta
    porcentaje = (df[config.OBJETIVO].value_counts(normalize=True) * 100).round(2)  # Frecuencia relativa (%)
    for clase in conteo.index:
        print(f"  {clase}: {conteo[clase]} ({porcentaje[clase]}%)")

    # --- Validación de columnas configuradas ---
    # Verifica que todas las columnas configuradas existan en el Excel
    faltantes = [c for c in config.COLUMNAS_E2 if c not in df.columns]
    if faltantes:
        print("\nATENCIÓN - columnas no encontradas en el Excel:")
        for c in faltantes:
            print("  -", c)
    else:
        print("\nTodas las columnas de E1 y E2 existen en el dataset.")

#--Comprobación de valores--
# Solo se ejecuta al correr este archivo directamente (no al importarlo desde otro módulo)
if __name__ == "__main__":
    df = cargar_datos()
    verificar_datos(df)