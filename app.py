import streamlit as st
import pandas as pd
import re
import unicodedata
import concurrent.futures
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==========================================
# CONFIGURACIÓN DEL MÓDULO SEN (Despliegue)
# ==========================================
st.set_page_config(page_title="Clasificador de Reseñas E-commerce", layout="wide")
st.title("📊 Sistema de Clasificación y Concurrencia de Reseñas")
st.markdown("Procesamiento en memoria RAM utilizando **Multithreading** para análisis de Big Data.")

# ==========================================
# MÓDULO CAR: PREPROCESAMIENTO LINGÜÍSTICO
# ==========================================
# Lista básica de palabras vacías (Stop-words) para reseñas en inglés (Amazon/Yelp)
STOP_WORDS = set(["the", "and", "a", "to", "of", "was", "i", "in", "for", "is", "it", 
                  "this", "that", "on", "with", "as", "my", "very", "but", "have", 
                  "they", "you", "we", "so", "are", "not", "be", "were", "had", "just"])

def clean_text(text):
    """
    Función pura para limpiar texto que será ejecutada por los hilos.
    1. Convierte a minúsculas.
    2. Quita acentos.
    3. Elimina caracteres especiales.
    4. Filtra stop-words y palabras cortas.
    """
    if not isinstance(text, str):
        return []
    
    # Minúsculas
    text = text.lower()
    # Quitar acentos (Normalización NFD)
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Quitar signos de puntuación y números
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenización y filtrado de stop-words
    words = text.split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

# ==========================================
# LECTURA DE DATOS EN MEMORIA (Sin BD)
# ==========================================
@st.cache_data
def load_data():
    """
    Lee el archivo comprimido directamente a la RAM usando Pandas.
    Se utiliza @st.cache_data para que Streamlit no lo lea cada vez que interactúas con la web.
    """
    try:
        # Asegúrate de que el nombre del archivo coincida con el que subas a GitHub
        df = pd.read_csv("dataset.zip", compression="zip")
        # Filtramos valores nulos
        df = df.dropna(subset=['text', 'stars'])
        return df
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return pd.DataFrame()

df_reviews = load_data()

if not df_reviews.empty:
    st.success(f"✅ Dataset cargado en memoria exitosamente: {len(df_reviews):,} reseñas listas para procesar.")
    
    # Separación lógica (Calificaciones 4-5 son positivas, 1-2 negativas)
    df_positive = df_reviews[df_reviews['stars'] >= 4]
    df_negative = df_reviews[df_reviews['stars'] <= 2]
    
    # ==========================================
    # MÓDULO CAR: PROCESAMIENTO PARALELO (Hilos)
    # ==========================================
    st.header("⚙️ Procesamiento Concurrente (Módulo CAR)")
    
    if st.button("Iniciar Procesamiento con Hilos"):
        with st.spinner("Ejecutando limpieza de texto en paralelo..."):
            
            # Procesamiento de reseñas positivas con ThreadPoolExecutor
            positive_words = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Mapeamos la función clean_text a toda la columna de reseñas
                results_pos = executor.map(clean_text, df_positive['text'].astype(str).tolist())
                for result in results_pos:
                    positive_words.extend(result)
                    
            # Procesamiento de reseñas negativas
            negative_words = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results_neg = executor.map(clean_text, df_negative['text'].astype(str).tolist())
                for result in results_neg:
                    negative_words.extend(result)

            # Conteo de frecuencias masivas
            top_pos = Counter(positive_words).most_common(10)
            top_neg = Counter(negative_words).most_common(10)

            st.success("Limpieza y conteo finalizados usando concurrencia.")

            # ==========================================
            # MÓDULO VDB: VISUALIZACIÓN DE DATOS
            # ==========================================
            st.header("📈 Visualización Multivariable (Módulo VDB)")
            
            col1, col2 = st.columns(2)
            
            # --- SECCIÓN POSITIVA ---
            with col1:
                st.subheader("Alegrías del Cliente (Positivas)")
                
                # Gráfico de Barras Multivariable (Cuantitativo)
                fig_pos, ax_pos = plt.subplots(figsize=(6,4))
                words_p, counts_p = zip(*top_pos)
                ax_pos.barh(words_p, counts_p, color='mediumseagreen')
                ax_pos.invert_yaxis()
                ax_pos.set_title("Top 10 Términos Positivos")
                st.pyplot(fig_pos)
                
                # Nube de Palabras (Cualitativo)
                st.write("**Nube de Palabras Positivas**")
                text_pos = " ".join(positive_words)
                wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(text_pos)
                fig_wc_pos, ax_wc_pos = plt.subplots()
                ax_wc_pos.imshow(wordcloud_pos, interpolation='bilinear')
                ax_wc_pos.axis("off")
                st.pyplot(fig_wc_pos)

            # --- SECCIÓN NEGATIVA ---
            with col2:
                st.subheader("Frustraciones del Cliente (Negativas)")
                
                # Gráfico de Barras Multivariable (Cuantitativo)
                fig_neg, ax_neg = plt.subplots(figsize=(6,4))
                words_n, counts_n = zip(*top_neg)
                ax_neg.barh(words_n, counts_n, color='indianred')
                ax_neg.invert_yaxis()
                ax_neg.set_title("Top 10 Términos Negativos")
                st.pyplot(fig_neg)
                
                # Nube de Palabras (Cualitativo)
                st.write("**Nube de Palabras Negativas**")
                text_neg = " ".join(negative_words)
                wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text_neg)
                fig_wc_neg, ax_wc_neg = plt.subplots()
                ax_wc_neg.imshow(wordcloud_neg, interpolation='bilinear')
                ax_wc_neg.axis("off")
                st.pyplot(fig_wc_neg)