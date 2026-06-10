# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from dataset import BOOKS_DATASET

# Page Configuration
st.set_page_config(
    page_title="Semantik & Hibrit Kütüphane Kataloğu",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
        color: #c9d1d9;
    }
    
    /* Title and Header styling */
    .title-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Design */
    .book-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .book-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.15);
        background: rgba(33, 38, 45, 0.8);
    }
    .book-title {
        color: #58a6ff;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .book-author {
        color: #bc8cff;
        font-size: 0.95rem;
        font-style: italic;
        margin-bottom: 10px;
    }
    .book-desc {
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.4;
        margin-bottom: 12px;
    }
    .book-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        font-size: 0.8rem;
    }
    .meta-tag {
        background: #21262d;
        border: 1px solid #30363d;
        color: #8b949e;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .meta-tag-isbn {
        background: rgba(240, 136, 62, 0.15);
        border: 1px solid rgba(240, 136, 62, 0.4);
        color: #f0883e;
        padding: 2px 8px;
        border-radius: 20px;
    }
    
    /* Method badges */
    .badge-boolean {
        background-color: #1f6feb;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-semantic {
        background-color: #238636;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-hybrid {
        background-color: #8957e5;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & INITIALIZATION -----------------

# Initialize dataset in session state to allow adding books dynamically
if "dataset" not in st.session_state:
    st.session_state.dataset = list(BOOKS_DATASET)

# Cache SentenceTransformer model
@st.cache_resource
def load_embedding_model():
    # Load all-MiniLM-L6-v2, which is lightweight (approx 120MB) and very fast
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

# Helper to build indexed text for each book
def get_book_text(book):
    return f"{book['title']} {book['author']} {book['subjects']} {book['description']}"

# Build indexing data and setup FAISS + TF-IDF
@st.cache_data(show_spinner=False)
def rebuild_indices(dataset_state):
    # 1. Prepare texts for embedding and TF-IDF
    texts = [get_book_text(b) for b in dataset_state]
    
    # 2. Dense Embeddings (FAISS)
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    # L2-normalize embeddings for Cosine Similarity inside Inner Product index
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    
    # 3. Sparse Embeddings (TF-IDF Vectorizer for Lexical scoring)
    vectorizer = TfidfVectorizer(lowercase=True, analyzer='word')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # 4. Dimensionality Reduction (PCA) for 2D visualization
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)
    
    return {
        "embeddings": embeddings,
        "faiss_index": faiss_index,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "coords_2d": coords_2d,
        "pca": pca
    }

# Compute indices based on current state of dataset
indices = rebuild_indices(st.session_state.dataset)

# ----------------- SEARCH ALGORITHMS -----------------

# 1. Traditional Boolean Search Engine
def boolean_search(query, dataset):
    query = query.strip().lower()
    if not query:
        return []
    
    matches = []
    
    for book in dataset:
        text = get_book_text(book).lower()
        
        # Simulating standard boolean parser
        # Supports: AND, OR, NOT (case-insensitive)
        
        # 1. Parse OR first (highest level split)
        if " or " in query:
            parts = query.split(" or ")
            is_match = any(boolean_match_segment(text, part) for part in parts)
        else:
            is_match = boolean_match_segment(text, query)
            
        if is_match:
            matches.append(book)
            
    return matches

def boolean_match_segment(text, segment):
    segment = segment.strip()
    
    # Handle NOT
    if " not " in segment:
        parts = segment.split(" not ")
        left_match = boolean_match_segment(text, parts[0])
        right_match = boolean_match_segment(text, parts[1])
        return left_match and not right_match
        
    # Handle AND
    if " and " in segment:
        parts = segment.split(" and ")
        return all(p.strip() in text for p in parts if p.strip())
        
    # No operators: Implicit AND for all words (standard search behavior)
    words = segment.split()
    if not words:
        return False
    return all(w in text for w in words)


# 2. Semantic Search (Dense Retrieval)
def semantic_search(query, dataset, indices, k=5):
    query_vector = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_vector)
    
    # Search the FAISS Index
    # FlatIP index returns cosine similarities as distances (higher is better, max 1.0)
    scores, idxs = indices["faiss_index"].search(query_vector, k)
    
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx != -1 and idx < len(dataset):
            book = dataset[idx].copy()
            book["score"] = float(score)
            results.append(book)
    return results


# 3. Hybrid Search Engine (Lexical + Semantic)
def hybrid_search(query, dataset, indices, alpha=0.5, k=5):
    # Get Semantic Scores
    query_vector = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_vector)
    sem_scores, _ = indices["faiss_index"].search(query_vector, len(dataset))
    
    # Mapping indices back to ensure alignment
    semantic_scores_map = {i: score for i, score in enumerate(sem_scores[0])}
    
    # Get Lexical (TF-IDF) Scores
    tfidf_vector = indices["vectorizer"].transform([query])
    # Cosine similarity between query tfidf and dataset tfidf matrix
    lex_scores = (indices["tfidf_matrix"] * tfidf_vector.T).toarray().flatten()
    lexical_scores_map = {i: score for i, score in enumerate(lex_scores)}
    
    # Combine Scores
    combined_results = []
    for i, book in enumerate(dataset):
        sem_score = semantic_scores_map.get(i, 0.0)
        lex_score = lexical_scores_map.get(i, 0.0)
        
        # Min-Max Scaling or Direct combination since both are cosine similarities [0, 1]
        # Soften semantic scores to standard [0, 1] range
        sem_score_norm = max(0.0, min(1.0, (sem_score + 1) / 2)) 
        
        hybrid_score = (alpha * sem_score_norm) + ((1.0 - alpha) * lex_score)
        
        book_copy = book.copy()
        book_copy["semantic_score"] = float(sem_score_norm)
        book_copy["lexical_score"] = float(lex_score)
        book_copy["score"] = float(hybrid_score)
        combined_results.append(book_copy)
        
    # Sort by hybrid score descending
    combined_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)
    return combined_results[:k]

# ----------------- SIDEBAR: DATASET MANAGEMENT -----------------

with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/library.png", width=80)
    st.markdown("### 📚 Katalog Yönetimi")
    st.write(f"Katalogda şu an **{len(st.session_state.dataset)}** kitap kayıtlı.")
    
    st.markdown("---")
    st.markdown("### 🆕 Yeni Kitap Ekle")
    with st.form("add_book_form", clear_on_submit=True):
        new_title = st.text_input("Kitap Adı")
        new_author = st.text_input("Yazar(lar)")
        new_subjects = st.text_input("Konu Başlıkları (Virgülle ayırın)")
        new_desc = st.text_area("Özet / Açıklama")
        new_isbn = st.text_input("ISBN (Örn: 978-975-...)")
        new_year = st.number_input("Yayın Yılı", min_value=-2000, max_value=2026, value=2024)
        
        submitted = st.form_submit_form_button = st.form_submit_button("Kataloğa Ekle & İndeksle")
        
        if submitted:
            if new_title and new_author and new_desc:
                new_book = {
                    "id": len(st.session_state.dataset) + 1,
                    "title": new_title,
                    "author": new_author,
                    "subjects": new_subjects,
                    "description": new_desc,
                    "isbn": new_isbn if new_isbn else "Yok",
                    "publication_year": int(new_year)
                }
                # Update dataset and clear cache of index rebuild
                st.session_state.dataset.append(new_book)
                st.cache_data.clear()
                st.success(f"'{new_title}' başarıyla eklendi ve FAISS vektör indeksine dahil edildi!")
                st.rerun()
            else:
                st.error("Lütfen Kitap Adı, Yazar ve Açıklama alanlarını doldurun.")

    st.markdown("---")
    st.markdown("### 🎓 Proje Hakkında")
    st.info(
        "Bu çalışma, kütüphane bilgi erişim sistemlerinde 'Doğal Dil' sorgularının doğrudan "
        "makine öğrenmesi ve vektörel uzay modelleriyle işlenmesini gösteren bir final projesidir. "
        "Sorguları Boolean parametrelerine ('AND', 'OR') çevirmenin yarattığı anlamsal kayıpları engellemeyi amaçlar."
    )

# ----------------- MAIN UI -----------------

st.markdown('<div class="title-text">Semantik ve Hibrit Kütüphane Arama Sistemi</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Doğal dil sorgularını Boolean mantığı yerine vektör uzayında anlamsal olarak arama prototipi</div>', unsafe_allow_html=True)

# Search input
search_query = st.text_input(
    "🔍 Aramak istediğiniz konuyu doğal dilde yazın (Örn: 'yapay zeka kullanan kütüphaneler', 'evrenin başlangıcı ve fizik', 'felsefi ideal devlet teorileri')",
    value="akıllı kütüphaneler ve makine öğrenmesi uygulamaları",
    placeholder="Sorguyu buraya yazın..."
)

# Tabs
tab_search, tab_visualization, tab_dataset = st.tabs(["🔎 Arama Motorları Karşılaştırması", "📊 2B Embedding Uzayı Görselleştirmesi", "📁 Mevcut Katalog Listesi"])

# ----------------- TAB 1: SEARCH COMPARISON -----------------
with tab_search:
    if search_query:
        # User controls for weights
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            alpha = st.slider(
                "⚖️ Hibrit Arama Ağırlık Dengesi (α - Alfa)",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="1.0 = Tamamen Semantik Arama (Anlam); 0.0 = Tamamen Leksikal Arama (Kelime Eşleşmesi/TF-IDF)"
            )
        with col_ctrl2:
            top_k = st.slider("📋 Gösterilecek Sonuç Sayısı", min_value=1, max_value=10, value=4)
            
        st.markdown("---")
        
        # Perform Searches
        boolean_results = boolean_search(search_query, st.session_state.dataset)
        semantic_results = semantic_search(search_query, st.session_state.dataset, indices, k=top_k)
        hybrid_results = hybrid_search(search_query, st.session_state.dataset, indices, alpha=alpha, k=top_k)
        
        # Display 3 columns
        col_bool, col_sem, col_hyb = st.columns(3)
        
        # 1. Column: Boolean Search
        with col_bool:
            st.markdown('### 🟦 Klasik Boolean Arama <span class="badge-boolean">Geleneksel</span>', unsafe_allow_html=True)
            # Explain Boolean conversion
            parsed_bool = " AND ".join(search_query.split())
            st.caption(f"**Boolean Dönüşümü:** `{parsed_bool}`")
            
            if not boolean_results:
                st.warning("⚠️ Eşleşen kayıt bulunamadı. Boolean mantığı doğal dil sorgusunu bölerek tam eşleşme aradığı için sonuç üretemedi.")
            else:
                for book in boolean_results[:top_k]:
                    st.markdown(f"""
                    <div class="book-card">
                        <div class="book-title">{book['title']}</div>
                        <div class="book-author">{book['author']}</div>
                        <div class="book-desc">{book['description']}</div>
                        <div class="book-meta">
                            <span class="meta-tag-isbn">ISBN: {book['isbn']}</span>
                            <span class="meta-tag">Yıl: {book['publication_year']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # 2. Column: Semantic Search
        with col_sem:
            st.markdown('### 🟩 Semantik Vektör Arama <span class="badge-semantic">Modern</span>', unsafe_allow_html=True)
            st.caption("**Model:** `all-MiniLM-L6-v2` + FAISS Cosine Similarity")
            
            for book in semantic_results:
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-title">{book['title']}</div>
                    <div class="book-author">{book['author']}</div>
                    <div class="book-desc">{book['description']}</div>
                    <div class="book-meta">
                        <span class="meta-tag-isbn" style="background-color: rgba(35, 134, 54, 0.15); border-color: rgba(35, 134, 54, 0.4); color: #3fb950;">Semantik Skor: {book['score']:.4f}</span>
                        <span class="meta-tag">Yıl: {book['publication_year']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # 3. Column: Hybrid Search
        with col_hyb:
            st.markdown('### 🟪 Hibrit Arama <span class="badge-hybrid">Otomobil (Önerilen)</span>', unsafe_allow_html=True)
            st.caption(f"**Kombinasyon:** {alpha} * Semantik + {1-alpha:.2f} * TF-IDF")
            
            for book in hybrid_results:
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-title">{book['title']}</div>
                    <div class="book-author">{book['author']}</div>
                    <div class="book-desc">{book['description']}</div>
                    <div class="book-meta">
                        <span class="meta-tag-isbn" style="background-color: rgba(137, 87, 229, 0.15); border-color: rgba(137, 87, 229, 0.4); color: #a371f7;">Hibrit Skor: {book['score']:.4f}</span>
                        <span class="meta-tag" title="Semantik: {book['semantic_score']:.3f} | Kelime: {book['lexical_score']:.3f}">Detaylar ℹ️</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 2: VISUALIZATION -----------------
with tab_visualization:
    st.markdown("### 🗺️ Vektör Embedding Uzayı")
    st.write(
        "Aşağıdaki grafik, kütüphanedeki kitapların 384 boyutlu anlamsal vektörlerinin PCA ile 2 boyuta düşürülmüş halidir. "
        "Yakın noktalar, anlamsal olarak birbirine benzeyen kitapları temsil eder."
    )
    
    if search_query:
        # Get query coordinate
        query_vector = model.encode([search_query]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Project using the same PCA
        query_2d = indices["pca"].transform(query_vector)[0]
        
        # Create Dataframe for plotting
        plot_data = []
        for i, book in enumerate(st.session_state.dataset):
            plot_data.append({
                "X": indices["coords_2d"][i, 0],
                "Y": indices["coords_2d"][i, 1],
                "Tip": "Kitap",
                "Başlık": book["title"],
                "Yazar": book["author"]
            })
            
        plot_data.append({
            "X": query_2d[0],
            "Y": query_2d[1],
            "Tip": "Arama Sorgusu",
            "Başlık": f"Sorgu: '{search_query}'",
            "Yazar": "Siz"
        })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Plotly Scatter
        fig = px.scatter(
            df_plot,
            x="X",
            y="Y",
            color="Tip",
            hover_name="Başlık",
            hover_data=["Yazar"],
            color_discrete_map={"Kitap": "#58a6ff", "Arama Sorgusu": "#ff7b72"},
            symbol="Tip",
            symbol_map={"Kitap": "circle", "Arama Sorgusu": "star"}
        )
        
        # Draw lines to nearest semantic neighbors
        nearest = semantic_search(search_query, st.session_state.dataset, indices, k=3)
        for near_book in nearest:
            # find original index of the book
            orig_idx = next((idx for idx, b in enumerate(st.session_state.dataset) if b["id"] == near_book["id"]), None)
            if orig_idx is not None:
                book_x = indices["coords_2d"][orig_idx, 0]
                book_y = indices["coords_2d"][orig_idx, 1]
                fig.add_trace(go.Scatter(
                    x=[query_2d[0], book_x],
                    y=[query_2d[1], book_y],
                    mode="lines",
                    line=dict(color="rgba(139, 148, 158, 0.4)", width=1.5, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                
        fig.update_traces(marker=dict(size=12))
        fig.update_layout(
            plot_bgcolor="rgba(22, 27, 34, 0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Boyut 1"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Boyut 2"),
            height=600,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Embedding grafiğini görmek için lütfen bir arama yapın.")

# ----------------- TAB 3: DATASET LIST -----------------
with tab_dataset:
    st.markdown("### 📁 Kütüphane Veritabanı Girdileri")
    df_books = pd.DataFrame(st.session_state.dataset)
    st.dataframe(
        df_books[["id", "title", "author", "isbn", "publication_year", "subjects"]],
        use_container_width=True,
        column_config={
            "id": "ID",
            "title": "Kitap Adı",
            "author": "Yazar",
            "isbn": "ISBN",
            "publication_year": "Yayın Yılı",
            "subjects": "Konular"
        }
    )
