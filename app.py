import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Sistem Rekomendasi Komik",
    page_icon="📚",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "goodreads_comics_cleaned.xlsx")
GENERIC_TAG_WORDS = {"comics", "cómics", "graphic", "novels", "novel", "comic", "books", "book"}


def format_tags(shelves_text: str) -> str:
    seen = []
    for w in str(shelves_text).lower().split():
        if w not in GENERIC_TAG_WORDS and w not in seen and len(w) > 1:
            seen.append(w)
    return ", ".join(seen) if seen else "-"


@st.cache_data(show_spinner="Memuat dataset komik...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["combined_features"] = df["combined_features"].fillna("")
    df["description"] = df["description"].fillna("Tidak ada deskripsi.")
    df["shelves"] = df["shelves"].fillna("-")
    if "image_url" in df.columns:
        df["image_url"] = df["image_url"].fillna("")
    else:
        df["image_url"] = ""
    return df


@st.cache_resource(show_spinner="Membangun model TF-IDF...")
def build_tfidf(texts: pd.Series):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix


def get_recommendations(df: pd.DataFrame, tfidf_matrix, selected_title: str, top_n: int = 10) -> pd.DataFrame:
    idx = df.index[df["title"] == selected_title][0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = sim_scores.argsort()[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results["similarity"] = (sim_scores[top_indices] * 100).round(2)
    return results


st.title("📚 Sistem Rekomendasi Komik (Content-Based Filtering)")
st.caption("Menggunakan **TF-IDF** & **Cosine Similarity** pada dataset Goodreads Comics & Graphic Novels.")

if not os.path.exists(DATA_PATH):
    st.error(f"File dataset tidak ditemukan di: `{DATA_PATH}`")
    st.stop()

df = load_data(DATA_PATH)
tfidf_matrix = build_tfidf(df["combined_features"])

with st.sidebar:
    st.header("⚙️ Pengaturan")
    top_n = st.slider("Jumlah rekomendasi", min_value=5, max_value=20, value=10, step=1)
    st.markdown("---")
    st.metric("Total Komik di Dataset", f"{len(df):,}")

selected_title = st.selectbox(
    "Pilih atau ketik judul komik yang kamu suka:",
    options=df["title"].tolist(),
    index=0,
)

if selected_title:
    selected_row = df[df["title"] == selected_title].iloc[0]
    with st.expander("📖 Detail Komik Pilihan", expanded=True):
        col_img, col1, col2 = st.columns([1, 3, 1])
        with col_img:
            if selected_row["image_url"]:
                st.image(selected_row["image_url"], width=140)
        with col1:
            st.subheader(selected_row["title"])
            st.write(selected_row["description"])
        with col2:
            st.metric("Rating Rata-rata", f"⭐ {selected_row['average_rating']:.2f}")
            st.metric("Jumlah Rating", f"{int(selected_row['ratings_count']):,}")
            st.caption(f"**Genre/Tag:** {format_tags(selected_row['shelves'])}")
            st.link_button(
                "Lihat di Goodreads ↗",
                f"https://www.goodreads.com/book/show/{selected_row['book_id']}",
                use_container_width=True,
            )

    if st.button("🔍 Cari Rekomendasi", type="primary"):
        recs = get_recommendations(df, tfidf_matrix, selected_title, top_n=top_n)
        st.subheader(f"✨ {top_n} Komik Mirip dengan *{selected_title}*")

        for rank, (_, row) in enumerate(recs.iterrows(), start=1):
            with st.container(border=True):
                c_img, c1, c2 = st.columns([1, 4, 1])
                with c_img:
                    if row["image_url"]:
                        st.image(row["image_url"], width=100)
                with c1:
                    st.markdown(f"**{rank}. {row['title']}**")
                    st.caption(f"Genre/Tag: {format_tags(row['shelves'])}")
                    with st.expander("Lihat Sinopsis"):
                        st.write(row["description"])
                with c2:
                    st.metric("Kemiripan", f"{row['similarity']}%")
                    st.caption(f"⭐ {row['average_rating']:.2f} ({int(row['ratings_count']):,} rating)")
                    st.link_button(
                        "Goodreads ↗",
                        f"https://www.goodreads.com/book/show/{row['book_id']}",
                        use_container_width=True,
                    )
