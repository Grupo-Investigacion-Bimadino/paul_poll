# Paso 1: Instalar bibliotecas necesarias
# pip install transformers pandas matplotlib openpyxl

import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Paso 2: Cargar los datos
def load_data(file_path):
    """Carga los comentarios desde un archivo Excel."""
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        if "Comentario" not in data.columns:
            raise ValueError("La columna 'Comentario' no se encontró en el archivo.")
        return data.dropna(subset=["Comentario"])
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return pd.DataFrame()

# Paso 3: Configuración del modelo
def configure_model():
    """Configura el modelo de análisis de sentimientos."""
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Paso 4: Analizar los sentimientos
def analyze_sentiments(comments, model):
    """Analiza los sentimientos de una lista de comentarios."""
    results = []
    for comment in comments:
        try:
            result = model(comment)[0]
            results.append({"Comentario": comment, "Sentimiento": result['label'], "Puntuación": result['score']})
        except Exception as e:
            print(f"Error al analizar el comentario '{comment}': {e}")
    return pd.DataFrame(results)

# Paso 5: Visualización de resultados
def visualize_results(df):
    """Crea un gráfico de barras con los resultados."""
    try:
        if not df.empty:
            counts = df["Sentimiento"].value_counts()
            counts.plot(kind="bar", title="Distribución de Sentimientos")
            plt.xlabel("Sentimiento")
            plt.ylabel("Cantidad")
            plt.show()
        else:
            print("No hay datos para graficar.")
    except Exception as e:
        print(f"Error al visualizar resultados: {e}")

# Paso 6: Guardar los resultados
def save_results(df, output_path):
    """Guarda los resultados en un archivo Excel."""
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"Resultados guardados en {output_path}")
    except Exception as e:
        print(f"Error al guardar resultados: {e}")

# Ejecución del Proyecto
if __name__ == "__main__":
    file_path = "data/comentarios_estudiantes.xlsx"  # Cambia esto al nombre de tu archivo
    output_path = "data/resultados_sentimientos.xlsx"

    # Cargar datos
    data = load_data(file_path)

    # Configurar modelo
    sentiment_analyzer = configure_model()
    if sentiment_analyzer and not data.empty:
        # Analizar sentimientos
        resultados = analyze_sentiments(data["Comentario"], sentiment_analyzer)

        # Visualizar resultados
        visualize_results(resultados)

        # Guardar resultados
        save_results(resultados, output_path)
