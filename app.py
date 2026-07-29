import io
import re
import csv
import random
import threading
import time
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
    page_title="E-Commerce Sentiment Engine (Unlimited CSV)",
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

# --- FUNCIONES DE COLOR PARA WORDCLOUD ---
def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    greens = ["#15803d", "#16a34a", "#059669", "#047857", "#22c55e", "#14532d"]
    return random.choice(greens)

def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    reds = ["#b91c1c", "#dc2626", "#ef4444", "#991b1b", "#c2410c", "#7f1d1d"]
    return random.choice(reds)

# --- 2. MOTOR DE PROCESAMIENTO POR LOTES (RAM-SAFE EN LA NUBE) ---
def process_lines_chunk(lines: list, text_index: int, rating_index: int, pos_counter: Counter, neg_counter: Counter, stats: dict):
    local_pos_count = 0
    local_neg_count = 0
    
    for line in lines:
        try:
            row = list(csv.reader([line]))[0]
            if len(row) <= max(text_index, rating_index): 
                continue
            
            raw_text = row[text_index]
            rating_str = row[rating_index].strip()
            
            clean_text = re.sub(r'[^\w\s]', '', raw_text.lower())
            words = [w for w in clean_text.split() if w not in ALL_STOPWORDS and len(w) > 2 and not w.isdigit()]
            
            if not words: 
                continue

            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            all_terms = words + bigrams

            is_pos_review = False
            is_neg_review = False

            if rating_str.isdigit():
                val = int(rating_str)
                if val >= 4: is_pos_review = True
                elif val <= 2: is_neg_review = True
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

# --- 3. MODAL DE FRECUENCIA DETALLADA ---
@st.dialog("Full Frequency Tally", width="large")
def show_frequency_tally(counter: Counter, sentiment_type: str, top_n: int):
    top_terms = counter.most_common(top_n)
    total_occurrences = sum(count for _, count in top_terms)
    
    color = "#059669" if sentiment_type == "positive" else "#dc2626"
    bg_color = "#dcfce7" if sentiment_type == "positive" else "#fee2e2"
    tag = "POSITIVE SENTIMENT" if sentiment_type == "positive" else "NEGATIVE SENTIMENT"
        
    st.markdown(f"""
<div style="margin-bottom: 24px;">
    <span style="background-color: {bg_color}; color: {color}; padding: 4px 12px; border-radius: 6px; font-size: 11px; font-family: monospace; font-weight: bold;">{tag}</span>
    <div style="color: #64748b; font-size: 14px; margin-top: 12px; font-family: monospace;">Customer Satisfaction • {len(top_terms)} terms • {total_occurrences:,} occurrences</div>
</div>
""", unsafe_allow_html=True)
    
    html_rows = ""
    max_count = top_terms[0][1] if top_terms else 1
    
    for idx, (term, count) in enumerate(top_terms):
        share = (count / total_occurrences) * 100 if total_occurrences > 0 else 0
        bar_width = (count / max_count) * 100 
        
        html_rows += f"""
<div style="display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid #f8fafc;">
    <div style="width: 8%; color: #cbd5e1; font-family: monospace;">{idx+1:02d}</div>
    <div style="width: 42%;">
        <div style="font-weight: 500; color: #1e293b; font-size: 14px;">{term}</div>
        <div style="height: 4px; background-color: #f1f5f9; width: 85%; border-radius: 2px;">
            <div style="height: 100%; background-color: {color}; width: {bar_width}%; border-radius: 2px;"></div>
        </div>
    </div>
    <div style="width: 15%; text-align: right; color: #94a3b8; font-family: monospace;">{share:.1f}%</div>
    <div style="width: 35%; text-align: right; color: {color}; font-family: monospace; font-weight: bold;">{count:,}</div>
</div>
"""
    st.markdown(html_rows, unsafe_allow_html=True)

# --- 4. BARRA LATERAL (Sidebar UI) ---
with st.sidebar:
    st.title("⚡ Configuración Engine")
    st.markdown("---")
    
    # SELECCIONA EL CSV SIN LÍMETE (Hasta 50 GB con el config.toml)
    uploaded_file = st.file_uploader(
        "📂 Selecciona cualquier CSV de tu computadora (Sin límite de tamaño):", 
        type=["csv"],
        help="El archivo será analizado por partes en stream, garantizando 0 colapsos en la nube."
    )
    
    st.markdown("---")
    num_threads = st.slider("Hilos Concurrentes", min_value=1, max_value=8, value=4)
    top_words_count = st.slider("Top Términos a procesar", min_value=10, max_value=100, value=30)
    lines_batch = st.select_slider("Líneas en RAM por lote", options=[10000, 25000, 50000, 100000], value=25000)

# --- 5. PANEL PRINCIPAL ---
st.title("⚡ E-Commerce Sentiment Analytics Engine")
st.caption("Arquitectura asistida por IA (VADER NLP) - Procesamiento Ilimitado para Streamlit Cloud")

if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
    st.session_state.pos_counter = None
    st.session_state.neg_counter = None
    st.session_state.stats = None
    st.session_state.total_records = 0
    st.session_state.execution_time = 0

if uploaded_file is None:
    st.info("👈 Por favor, selecciona un archivo CSV desde la barra lateral. No importa qué tan pesado sea.")
else:
    try:
        # CONVERTIR EN FLUJO DE STREAM PURO (0% Uso adicional de RAM)
        file_stream = io.TextIOWrapper(uploaded_file, encoding='utf-8', errors='ignore')
        
        # Leemos solo la primera línea del flujo para obtener los encabezados al instante
        header_line = file_stream.readline()
        sample_reader = list(csv.reader([header_line]))
        columns = sample_reader[0] if sample_reader else []

        if not columns:
            st.error("No se pudieron detectar columnas en el CSV.")
        else:
            text_default_idx = 0
            for cand in ["Text", "text", "Review", "review", "Comment", "comment", "Summary"]:
                if cand in columns:
                    text_default_idx = columns.index(cand)
                    break

            rating_default_idx = min(1, len(columns) - 1)
            for cand in ["Score", "score", "Rating", "rating", "Stars", "stars"]:
                if cand in columns:
                    rating_default_idx = columns.index(cand)
                    break

            col_a, col_b = st.columns(2)
            with col_a:
                text_col_name = st.selectbox("Columna de RESEÑA (Texto):", columns, index=text_default_idx)
            with col_b:
                rating_col_name = st.selectbox("Columna de CALIFICACIÓN (Rating):", columns, index=rating_default_idx)

            if st.button("🚀 Iniciar Análisis de Gran Escala"):
                text_idx = columns.index(text_col_name)
                rating_idx = columns.index(rating_col_name)
                
                start_time = time.time()
                pos_counter = Counter()
                neg_counter = Counter()
                stats = {"pos_reviews": 0, "neg_reviews": 0}
                total_records = 0
                
                with st.spinner("Procesando archivo sin límite en Streamlit Cloud..."):
                    while True:
                        # Sacamos únicamente los lotes configurados sin colapsar la RAM
                        batch = [file_stream.readline() for _ in range(lines_batch)]
                        batch = [line for line in batch if line]
                        
                        if not batch:
                            break
                            
                        total_records += len(batch)
                        
                        # Dividir el lote y procesar con multihilo
                        chunk_size = max(1, len(batch) // num_threads)
                        chunks = [batch[i * chunk_size : (i + 1) * chunk_size] for i in range(num_threads)]
                        if len(batch) % num_threads != 0:
                            chunks[-1].extend(batch[num_threads * chunk_size:])
                            
                        threads = []
                        for i in range(len(chunks)):
                            t = threading.Thread(
                                target=process_lines_chunk,
                                args=(chunks[i], text_idx, rating_idx, pos_counter, neg_counter, stats)
                            )
                            threads.append(t)
                            t.start()

                        for t in threads:
                            t.join()

                st.session_state.pos_counter = pos_counter
                st.session_state.neg_counter = neg_counter
                st.session_state.stats = stats
                st.session_state.total_records = total_records
                st.session_state.execution_time = round(time.time() - start_time, 4)
                st.session_state.is_analyzed = True

    except Exception as e:
        st.error(f"Error procesando el archivo: {str(e)}")

# --- 6. VISUALIZACIÓN ---
if st.session_state.is_analyzed:
    pos_counter = st.session_state.pos_counter
    neg_counter = st.session_state.neg_counter
    stats = st.session_state.stats
    
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Líneas Procesadas", f"{st.session_state.total_records:,}")
    m2.metric("Tiempo Total", f"{st.session_state.execution_time} seg")
    m3.metric("Hilos Ejecutados", f"{num_threads}")
    m4.metric("Agente IA", "NLTK VADER")

    st.markdown("---")
    c_pos, c_neg = st.columns(2)
    
    total_eval = stats["pos_reviews"] + stats["neg_reviews"] or 1
    pos_rate = round((stats["pos_reviews"] / total_eval) * 100, 1)
    neg_rate = round((stats["neg_reviews"] / total_eval) * 100, 1)

    # --- TARJETA Y WORDCLOUD POSITIVO ---
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

    # --- TARJETA Y WORDCLOUD NEGATIVO ---
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
                if st.button("📊 View Full Frequency Tally ➔", key="btn_neg", use_container_width=True):
                    show_frequency_tally(neg_counter, "negative", top_words_count)
        else:
            st.info("La IA no detectó tokens estrictamente negativos.")

    # --- GRÁFICA MULTIVARIABLE (BAR CHART) ---
    st.markdown("---")
    st.subheader("📊 Frecuencia Comparativa (Plotly Express)")
    
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
