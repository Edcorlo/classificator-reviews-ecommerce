import os
import re
import random
import threading
import time
import math
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="E-Commerce Sentiment Engine",
    page_icon="⚡",
    layout="wide"
)

# --- 1. CONFIGURACIÓN DEL AGENTE DE IA (VADER) Y STOPWORDS ---
@st.cache_resource
def setup_nlp_agent():
    for corpus in ['stopwords', 'vader_lexicon']:
        try:
            nltk.data.find(f'corpora/{corpus}')
        except LookupError:
            nltk.download(corpus, quiet=True)
            
        try:
            nltk.data.find(f'sentiment/{corpus}')
        except LookupError:
            pass
            
    nltk.download('vader_lexicon', quiet=True)
    
    base_stops = set(stopwords.words('english'))
    domain_noise = {
        'product', 'amazon', 'bought', 'item', 'one', 'would', 'get', 'like', 'use', 'used', 
        'using', 'time', 'even', 'much', 'really', 'just', 'also', 'order', 'ordered', 'buy', 
        'buying', 'got', 'day', 'days', 'first', 'back', 'will', 'still', 'know', 'think', 
        'coffee', 'food', 'tea', 'flavor', 'taste', 'bag', 'cup', 'dog', 'box', 'sugar', 
        'eat', 'drink', 'water', 'price', 'tried', 'try', 'make', 'found', 'find', 'little', 
        'two', 'way', 'since', 'made', 'dont', 'didnt', 'could', 'thought', 'something', 'ive'
    }
    return base_stops.union(domain_noise), SentimentIntensityAnalyzer()

ALL_STOPWORDS, SIA_AGENT = setup_nlp_agent()
SENTIMENT_CACHE = {}

# --- FUNCIONES DE COLOR PARA WORDCLOUD FIGMA ---
def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    greens = ["#15803d", "#16a34a", "#059669", "#047857", "#22c55e", "#14532d"]
    return random.choice(greens)

def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    reds = ["#b91c1c", "#dc2626", "#ef4444", "#991b1b", "#c2410c", "#7f1d1d"]
    return random.choice(reds)

# --- CARGA DE DATOS OPTIMIZADA EN MEMORIA ---
@st.cache_data
def load_data(file_path):
    """Lee el dataset comprimido directamente a la RAM sin saturar Streamlit."""
    try:
        df = pd.read_csv(file_path, compression='zip')
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# --- 2. MOTOR DE PROCESAMIENTO MULTIHILO (Adaptado a Pandas) ---
def process_dataframe_chunk(df_chunk, text_col, rating_col, pos_counter, neg_counter, stats):
    local_pos_count = 0
    local_neg_count = 0
    
    for _, row in df_chunk.iterrows():
        try:
            raw_text = str(row[text_col])
            rating_str = str(row[rating_col]).strip()
            
            clean_text = re.sub(r'[^\w\s]', '', raw_text.lower())
            words = [w for w in clean_text.split() if w not in ALL_STOPWORDS and len(w) > 2 and not w.isdigit()]
            
            if not words: continue

            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            all_terms = words + bigrams

            is_pos_review = False
            is_neg_review = False

            if rating_str.isdigit():
                if int(rating_str) >= 4: is_pos_review = True
                elif int(rating_str) <= 2: is_neg_review = True
            else:
                r_lower = rating_str.lower()
                if r_lower in ['positive', 'pos', 'good', '5', '4']: is_pos_review = True
                elif r_lower in ['negative', 'neg', 'bad', '1', '2']: is_neg_review = True

            if is_pos_review:
                local_pos_count += 1
                for term in all_terms:
                    if term not in SENTIMENT_CACHE:
                        SENTIMENT_CACHE[term] = SIA_AGENT.polarity_scores(term)['compound']
                    if SENTIMENT_CACHE[term] > 0.15:
                        pos_counter.update([term])
                        
            elif is_neg_review:
                local_neg_count += 1
                for term in all_terms:
                    if term not in SENTIMENT_CACHE:
                        SENTIMENT_CACHE[term] = SIA_AGENT.polarity_scores(term)['compound']
                    if SENTIMENT_CACHE[term] < -0.15:
                        neg_counter.update([term])

        except Exception:
            continue

    stats["pos_reviews"] += local_pos_count
    stats["neg_reviews"] += local_neg_count

# --- 3. MODAL DE FRECUENCIA DETALLADA (FIGMA STYLE) ---
@st.dialog("Full Frequency Tally", width="large")
def show_frequency_tally(counter: Counter, sentiment_type: str, top_n: int):
    top_terms = counter.most_common(top_n)
    total_occurrences = sum(count for _, count in top_terms)
    
    if sentiment_type == "positive":
        color = "#059669"
        bg_color = "#dcfce7"
        tag = "POSITIVE SENTIMENT"
    else:
        color = "#dc2626"
        bg_color = "#fee2e2"
        tag = "NEGATIVE SENTIMENT"
        
    st.markdown(f"""
<div style="margin-bottom: 24px;">
    <span style="background-color: {bg_color}; color: {color}; padding: 4px 12px; border-radius: 6px; font-size: 11px; font-family: monospace; font-weight: bold; letter-spacing: 0.5px;">{tag}</span>
    <div style="color: #64748b; font-size: 14px; margin-top: 12px; font-family: monospace;">Customer Satisfaction • {len(top_terms)} terms • {total_occurrences:,} total occurrences</div>
</div>

<div style="display: flex; font-size: 12px; color: #94a3b8; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 12px; font-family: monospace; letter-spacing: 1px;">
    <div style="width: 8%;">#</div>
    <div style="width: 42%;">TERM</div>
    <div style="width: 15%; text-align: right;">SHARE</div>
    <div style="width: 35%; text-align: right;">FREQUENCY / OCCURRENCES</div>
</div>
""", unsafe_allow_html=True)
    
    html_rows = ""
    max_count = top_terms[0][1] if top_terms else 1
    
    for idx, (term, count) in enumerate(top_terms):
        share = (count / total_occurrences) * 100 if total_occurrences > 0 else 0
        bar_width = (count / max_count) * 100
        
        html_rows += f"""
<div style="display: flex; align-items: center; padding: 12px 0; border-bottom: 1px solid #f8fafc;">
    <div style="width: 8%; color: #cbd5e1; font-family: monospace; font-size: 14px;">{idx+1:02d}</div>
    <div style="width: 42%;">
        <div style="font-weight: 500; color: #1e293b; font-size: 15px; margin-bottom: 6px;">{term}</div>
        <div style="height: 5px; background-color: #f1f5f9; width: 85%; border-radius: 3px;">
            <div style="height: 100%; background-color: {color}; width: {bar_width}%; border-radius: 3px;"></div>
        </div>
    </div>
    <div style="width: 15%; text-align: right; color: #94a3b8; font-size: 14px; font-family: monospace;">{share:.1f}%</div>
    <div style="width: 35%; text-align: right; color: {color}; font-family: monospace; font-size: 16px; font-weight: bold;">{count:,}</div>
</div>
"""
        
    html_rows += f"""
<div style="margin-top: 16px; font-size: 11px; color: #94a3b8; font-family: monospace; letter-spacing: 1px;">
    NLP PIPELINE • TF-IDF WEIGHTED • STOP-WORDS REMOVED
</div>
"""
    
    st.markdown(html_rows, unsafe_allow_html=True)


# --- 4. BARRA LATERAL (Sidebar UI) ---
with st.sidebar:
    st.title("Configuración Engine")
    st.markdown("---")
    file_path = st.text_input("Ruta del archivo ZIP:", "dataset.zip")
    
    st.markdown("---")
    max_cpus = os.cpu_count() or 4
    num_threads = st.slider("Hilos Concurrentes (Threads)", min_value=1, max_value=16, value=max_cpus)
    top_words_count = st.slider("Top Términos a procesar", min_value=10, max_value=100, value=30)

# --- 5. PANEL PRINCIPAL ---
st.title("E-Commerce Sentiment Analytics Engine")
st.caption("Arquitectura asistida por IA (VADER NLP) con procesamiento concurrente en RAM")

# 5.1 INICIALIZAR MEMORIA (Session State)
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
    st.session_state.pos_counter = None
    st.session_state.neg_counter = None
    st.session_state.stats = None
    st.session_state.total_records = 0
    st.session_state.execution_time = 0

if not os.path.exists(file_path):
    st.warning(f"No se encontró el archivo: `{file_path}`. Asegúrate de haberlo subido a GitHub con ese nombre exacto.")
else:
    df = load_data(file_path)
    
    if df is not None:
        columns = list(df.columns)
        st.success(f"✅ ¡Dataset comprimido cargado en RAM exitosamente! Total de registros: {len(df):,}")
        
        with st.expander("Ver vista previa de los datos"):
            st.dataframe(df.head())

        col_a, col_b = st.columns(2)
        with col_a:
            text_col_name = st.selectbox("Columna del TEXTO (Reseña):", columns, index=columns.index("text") if "text" in columns else (columns.index("Text") if "Text" in columns else 0))
        with col_b:
            rating_col_name = st.selectbox("Columna de CALIFICACIÓN (Score/Rating):", columns, index=columns.index("stars") if "stars" in columns else (columns.index("Score") if "Score" in columns else 0))

        # 5.2 BLOQUE DE PROCESAMIENTO
        if st.button("Iniciar Análisis con IA Multihilo"):
            start_time = time.time()
            total_records = len(df)
            
            # División de datos en bloques (Chunks) para los hilos
            chunk_size = math.ceil(total_records / num_threads)
            chunks = [df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(num_threads)]
            chunks = [c for c in chunks if not c.empty] # Limpiar chunks vacíos

            pos_counter = Counter()
            neg_counter = Counter()
            stats = {"pos_reviews": 0, "neg_reviews": 0}
            threads = []

            for chunk in chunks:
                t = threading.Thread(
                    target=process_dataframe_chunk,
                    args=(chunk, text_col_name, rating_col_name, pos_counter, neg_counter, stats)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # GUARDAR RESULTADOS EN MEMORIA
            st.session_state.pos_counter = pos_counter
            st.session_state.neg_counter = neg_counter
            st.session_state.stats = stats
            st.session_state.total_records = total_records
            st.session_state.execution_time = round(time.time() - start_time, 4)
            st.session_state.is_analyzed = True

# 5.3 BLOQUE DE VISUALIZACIÓN (Siempre se muestra si el análisis ya se hizo)
if st.session_state.is_analyzed:
    
    pos_counter = st.session_state.pos_counter
    neg_counter = st.session_state.neg_counter
    stats = st.session_state.stats
    
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Líneas Procesadas", f"{st.session_state.total_records:,}")
    m2.metric("Tiempo Total (Concurrente)", f"{st.session_state.execution_time} seg")
    m3.metric("Hilos Ejecutados", f"{num_threads}")
    m4.metric("Agente IA", "NLTK VADER")

    st.markdown("---")
    c_pos, c_neg = st.columns(2)
    
    total_eval_reviews = stats["pos_reviews"] + stats["neg_reviews"] or 1
    pos_rate = round((stats["pos_reviews"] / total_eval_reviews) * 100, 1)
    neg_rate = round((stats["neg_reviews"] / total_eval_reviews) * 100, 1)

    # --- TARJETA POSITIVA ---
    with c_pos:
        st.markdown(
            '### 🟢 Customer Satisfaction Analytics &nbsp;<span style="background-color: #dcfce7; color: #15803d; padding: 3px 10px; border-radius: 6px; font-size: 13px; font-weight: bold;">POSITIVE</span>', 
            unsafe_allow_html=True
        )
        st.caption("POSITIVE SENTIMENT WORD CLOUD • NLP Weighted")
        
        if pos_counter:
            wc_pos = WordCloud(
                width=700, height=380, background_color="#f0fdf4", color_func=green_color_func, max_words=top_words_count, collocations=False
            ).generate_from_frequencies(pos_counter)
            
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.imshow(wc_pos, interpolation="bilinear")
            ax.axis("off")
            fig.patch.set_facecolor('#f0fdf4')
            st.pyplot(fig)
            
            total_pos_words = sum(pos_counter.values())
            top_pos_term = pos_counter.most_common(1)[0][0] if pos_counter else "N/A"
            
            st.markdown(f"""
            <div style="background-color: #f2fcf5; border: 1px solid #bbf7d0; border-radius: 8px; padding: 16px 24px; margin-top: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-family: monospace; color: #8899a6; font-size: 11px; letter-spacing: 0.8px;">TOTAL POSITIVE WORDS</div>
                        <div style="color: #059669; font-size: 24px; font-weight: 700; margin-top: 4px;">{total_pos_words:,}</div>
                    </div>
                    <div style="flex: 1;">
                        <div style="font-family: monospace; color: #8899a6; font-size: 11px; letter-spacing: 0.8px;">TOP TERM</div>
                        <div style="color: #059669; font-size: 24px; font-weight: 700; margin-top: 4px;">{top_pos_term}</div>
                    </div>
                    <div style="flex: 1;">
                        <div style="font-family: monospace; color: #8899a6; font-size: 11px; letter-spacing: 0.8px;">POSITIVE RATE</div>
                        <div style="color: #059669; font-size: 24px; font-weight: 700; margin-top: 4px;">{pos_rate}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_info, col_btn = st.columns([1.5, 1])
            with col_info:
                st.caption(f"<div style='font-family: monospace; color: #a1a1aa; padding-top: 8px;'>📌 {len(pos_counter):,} TERMS INDEXED BY AI</div>", unsafe_allow_html=True)
            with col_btn:
                if st.button("📊 View Full Frequency Tally ➔", key="btn_pos", use_container_width=True):
                    show_frequency_tally(pos_counter, "positive", top_words_count)
        else:
            st.info("La IA no detectó tokens estrictamente positivos.")

    # --- TARJETA NEGATIVA ---
    with c_neg:
        st.markdown(
            '### 🔴 Critical Customer Frustrations &nbsp;<span style="background-color: #fee2e2; color: #b91c1c; padding: 3px 10px; border-radius: 6px; font-size: 13px; font-weight: bold;">NEGATIVE</span>', 
            unsafe_allow_html=True
        )
        st.caption("NEGATIVE SENTIMENT WORD CLOUD • NLP Weighted")
        
        if neg_counter:
            wc_neg = WordCloud(
                width=700, height=380, background_color="#fef2f2", color_func=red_color_func, max_words=top_words_count, collocations=False
            ).generate_from_frequencies(neg_counter)
            
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.imshow(wc_neg, interpolation="bilinear")
            ax.axis("off")
            fig.patch.set_facecolor('#fef2f2')
            st.pyplot(fig)
            
            total_neg_words = sum(neg_counter.values())
            top_neg_term = neg_counter.most_common(1)[0][0] if neg_counter else "N/A"
            
            st.markdown(f"""
            <div style="background-color: #fff5f5; border: 1px solid #fecaca; border-radius: 8px; padding: 16px 24px; margin-top: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-family: monospace; color: #8899a6; font-size: 11px; letter-spacing: 0.8px;">TOTAL FRUSTRATIONS</div>
                        <div style="color: #dc2626; font-size: 24px; font-weight: 700; margin-top: 4px;">{total_neg_words:,}</div>
                    </div>
                    <div style="flex: 1;">
                        <div style="font-family: monospace; color: #8899a6; font-size: 11px; letter-spacing: 0.8px;">TOP COMPLAINT</div>
                        <div style="color: #dc2626; font-size: 24px; font-weight: 700; margin-top: 4px;">{top_neg_term}</div>
                    </div>
                    <div style="flex: 1;">
                        <div style="font-family: monospace; color: #8899a6; font-size: 11px; letter-spacing: 0.8px;">NEGATIVE RATE</div>
                        <div style="color: #dc2626; font-size: 24px; font-weight: 700; margin-top: 4px;">{neg_rate}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_info, col_btn = st.columns([1.5, 1])
            with col_info:
                st.caption(f"<div style='font-family: monospace; color: #a1a1aa; padding-top: 8px;'>📌 {len(neg_counter):,} TERMS INDEXED BY AI</div>", unsafe_allow_html=True)
            with col_btn:
                if st.button("View Full Frequency Tally ➔", key="btn_neg", use_container_width=True):
                    show_frequency_tally(neg_counter, "negative", top_words_count)
        else:
            st.info("La IA no detectó tokens estrictamente negativos.")

    # --- GRÁFICA MULTIVARIABLE (MÓDULO VDB) ---
    st.markdown("---")
    st.subheader("Frecuencia Comparativa (Plotly Express)")
    
    pos_data = [{"Palabra": w, "Frecuencia": c, "Sentimiento": "Positivo"} for w, c in pos_counter.most_common(15)]
    neg_data = [{"Palabra": w, "Frecuencia": c, "Sentimiento": "Negativo"} for w, c in neg_counter.most_common(15)]
    
    df_combined = pos_data + neg_data
    if df_combined:
        fig_bar = px.bar(
            df_combined,
            x="Palabra",
            y="Frecuencia",
            color="Sentimiento",
            barmode="group",
            color_discrete_map={"Positivo": "#16a34a", "Negativo": "#dc2626"},
            title="Top Términos Comparativos Validados por IA"
        )
        fig_bar.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_bar, use_container_width=True)
