# Paso 1: Instalar bibliotecas necesarias
# pip install transformers pandas matplotlib openpyxl

import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Paso 2: Cargar los datos
def load_data(file_path):
    """Carga los comentarios desde un archivo Excel."""
    data = pd.read_excel(file_path, engine='openpyxl')  # Cambiado para soportar archivos .xlsx
    return data

# Paso 3: Configuración del modelo
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Paso 4: Analizar los sentimientos
def analyze_sentiments(comments):
    """Analiza los sentimientos de una lista de comentarios."""
    results = []
    for comment in comments:
        result = sentiment_analyzer(comment)[0]  # Obtener la predicción
        results.append({"Comentario": comment, "Sentimiento": result['label'], "Puntuación": result['score']})
    return pd.DataFrame(results)

# Paso 5: Visualización de resultados
def visualize_results(df):
    """Crea un gráfico de barras con los resultados."""
    counts = df["Sentimiento"].value_counts()
    counts.plot(kind="bar", title="Distribución de Sentimientos")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad")
    plt.show()

# Paso 6: Guardar los resultados
def save_results(df, output_path):
    """Guarda los resultados en un archivo Excel."""
    df.to_excel(output_path, index=False, engine='openpyxl')

# Ejecución del Proyecto
if __name__ == "__main__":
    file_path = "comentarios_estudiantes.xlsx"  # Cambia esto al nombre de tu archivo
    output_path = "resultados_sentimientos.xlsx"

    # Cargar datos
    data = load_data(file_path)

    # Analizar sentimientos
    resultados = analyze_sentiments(data["Comentario"])

    # Visualizar resultados
    visualize_results(resultados)

    # Guardar resultados
    save_results(resultados, output_path)
    print(f"Análisis completado. Resultados guardados en {output_path}")


