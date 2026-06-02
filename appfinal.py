import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Balkan Cinema AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# STYLING (CSS)
# ==========================================
st.markdown("""
    <style>
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }

    /* Styling στον τίτλο */
    .main-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    
    div.stInfo {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        color: #334155;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    
   
    div.stInfo i {
        display: none;
    }

    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #4682b4;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 400;
        color: #64748b;
    }

    /* Sidebar styling */
    .css-1639199 {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Styling Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4682b4;
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #36648b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# BLACKLIST & DATA LOGIC (UNCHANGED)
# ==========================================
FALSE_POSITIVES = [
    'The Power of the Dog', 'The Chronicles of Narnia: Prince Caspian', 'Triangle of Sadness',
    'Mirrors', 'The Lost Daughter', 'Child 44', 'Autómata', 'Watcher',
    'Vampire Academy: Blood Sisters', "Kelly's Heroes", 'Hellraiser', 'Seed of Chucky',
    'Once Upon a Time in Anatolia', 'The Zero Theorem', 'Crimes of the Future',
    'Joyeux Noël', 'Them', 'The Necessary Death of Charlie Countryman', 'Beckett',
    'Voyagers', 'The Innocents', 'An American Haunting', 'Inside', 'Loving Pablo',
    'How to Have Sex', 'Flee', 'Blood & Chocolate', 'Dara of Jasenovac', 'All Girls Weekend'
]

BALKAN_COUNTRIES = [
    'Albania', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Greece', 
    'Kosovo', 'Montenegro', 'North Macedonia', 'Romania', 'Serbia', 
    'Slovenia', 'Yugoslavia'
]

NAME_MAP = {
    "People's Republic of Bulgaria": "Bulgaria",
    "Macedonia": "North Macedonia",
    "Serbia and Montenegro": "Serbia",
}

def is_strictly_single_balkan(val):
    if pd.isna(val): return False
    val_str = str(val).replace('|', ',')
    parts = [p.strip() for p in val_str.split(',')]
    if len(parts) > 1: return False
    mapped_country = NAME_MAP.get(parts[0], parts[0])
    return mapped_country in BALKAN_COUNTRIES

def primary_country(val):
    if pd.isna(val): return None
    for sep in ['|', ',']:
        if sep in val: return val.split(sep)[0].strip()
    return val.strip()

@st.cache_data
def fetch_balkan_data():
    df = pd.read_csv("data/balkan_movies_confirmed.csv")
    df = df[~df['title_final'].isin(FALSE_POSITIVES)].copy()
    df = df[df['country'].apply(is_strictly_single_balkan)].copy()
    df = df.dropna(subset=['year_final'])
    df['year'] = df['year_final'].astype(int)
    df = df[(df['year'] >= 1945) & (df['year'] <= 2024)].copy()
    df['primary_country'] = df['country'].apply(primary_country).replace(NAME_MAP)
    df_balkan = df.drop_duplicates(subset=['title_final', 'primary_country']).copy()
    df_balkan['imdb_votes'] = pd.to_numeric(df_balkan['imdb_votes'], errors='coerce').fillna(0)
    df_balkan['imdb_rating'] = pd.to_numeric(df_balkan['imdb_rating'], errors='coerce')
    
    df_rec = df_balkan.dropna(subset=['title_final']).copy()
    df_rec = df_rec.reset_index(drop=True)
    df_rec['genres_final'] = df_rec['genres_final'].fillna('').str.replace(',', ' ')
    df_rec['director_names'] = df_rec['director_names'].fillna('').str.replace(',', ' ')
    df_rec['plot_overview'] = df_rec['plot_overview'].fillna('')
    df_rec['combined_features'] = (
        (df_rec['genres_final'] + " ") * 3 + (df_rec['director_names'] + " ") * 2 + df_rec['plot_overview']
    )
    return df_balkan, df_rec

df, df_rec = fetch_balkan_data()

# ==========================================
# NAVIGATION
# ==========================================
if 'app_view' not in st.session_state: st.session_state['app_view'] = 'map'
if 'selected_country' not in st.session_state: st.session_state['selected_country'] = 'Greece'

st.sidebar.markdown("### Balkan Cinema AI")
page = st.sidebar.radio("Navigation", ["Map & Insights", "Movie Recommender"])

# ==========================================
# PAGE 1: MAP & INSIGHTS
# ==========================================
# PAGE 1: MAP & INSIGHTS
# ==========================================
if page == "Map & Insights":

    if st.session_state['app_view'] == 'map':
        st.markdown('<h1 class="main-title">Geographical Insights</h1>', unsafe_allow_html=True)
        st.write("Explore the production volume of Balkan nations. Click the map to dive into country details.")
        
        country_dropdown = st.selectbox("Select a country:", ["Select..."] + BALKAN_COUNTRIES)
        if country_dropdown != "Select...":
            st.session_state['selected_country'] = country_dropdown
            st.session_state['app_view'] = 'details'
            st.rerun()

        map_data = df['primary_country'].value_counts().reset_index()
        map_data.columns = ['Country', 'Total']
        fig = px.choropleth(map_data, locations="Country", locationmode="country names", color="Total", color_continuous_scale="Blues")
        fig.update_geos(center=dict(lon=20.0, lat=42.5), projection_scale=4.5, showcountries=True)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=550, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        map_selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        if map_selection and "points" in map_selection.selection and len(map_selection.selection["points"]) > 0:
            loc = map_selection.selection["points"][0].get("location")
            if loc in BALKAN_COUNTRIES:
                st.session_state['selected_country'] = loc
                st.session_state['app_view'] = 'details'
                st.rerun()

    elif st.session_state['app_view'] == 'details':
        curr = st.session_state['selected_country']
        if st.button("⬅ Back to Map"):
            st.session_state['app_view'] = 'map'
            st.rerun()

        st.markdown(f'<h1 class="main-title">{curr} Cinema</h1>', unsafe_allow_html=True)
        c_df = df[df['primary_country'] == curr].copy()

        s1, s2, s3 = st.columns(3)
        s1.metric("Productions", len(c_df))
        s2.metric("Avg Rating", f"{round(c_df['imdb_rating'].mean(), 1)}/10")
        s3.metric("Total Votes", f"{int(c_df['imdb_votes'].sum()):,}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Most Popular")
            data = c_df.sort_values('imdb_votes', ascending=False).head(5)
            for _, row in data.iterrows():
                st.info(f"**{row['title_final']}** ({int(row['year'])})  \nRating: {row['imdb_rating']} | Votes: {int(row['imdb_votes']):,}")

        with col2:
            st.markdown("### Hidden Gems")
            gems = c_df[(c_df['imdb_rating'] >= 7.2) & (c_df['imdb_votes'].between(20, 400))].sort_values('imdb_rating', ascending=False).head(5)
            if not gems.empty:
                for _, row in gems.iterrows():
                    st.info(f"**{row['title_final']}** ({int(row['year'])})  \nRating: {row['imdb_rating']} | Rare Discovery")
            else:
                st.write("No gems found with current criteria.")

# ==========================================
# PAGE 2: RECOMMENDER
# ==========================================
else:
    st.markdown('<h1 class="main-title">Movie Recommender</h1>', unsafe_allow_html=True)
    st.write("Find similar movies based on plot, director, and genre similarity using AI.")
    
    choice = st.selectbox("Search a movie:", sorted(df_rec['title_final'].unique()))
    
    if st.button("Generate Recommendations"):
        with st.spinner("Analyzing cinematic features..."):

            # IMPORTANT: reset index to align with TF-IDF rows
            df_rec_clean = df_rec.reset_index(drop=True)

            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_mat = vectorizer.fit_transform(df_rec_clean['combined_features'])

            # SAFE mapping title → index
            title_to_idx = pd.Series(df_rec_clean.index, index=df_rec_clean['title_final'])

            if choice not in title_to_idx:
                st.error("Movie not found in dataset.")
                st.stop()

            idx = int(title_to_idx[choice])

            sim = cosine_similarity(tfidf_mat[idx], tfidf_mat).flatten()

            # Rating boost
            ratings = df_rec_clean['imdb_rating'].fillna(5.0).values
            scores = sim + (sim * (ratings / 10.0) * 0.15)

            scores[idx] = -1

            top_idx = scores.argsort()[-6:-1][::-1]

            st.markdown("#### Recommendations for you:")

            for i in top_idx:
                r = df_rec_clean.iloc[i]
                st.info(
                    f"**{r['title_final']}** ({int(r['year'])})  \n"
                    f"Dir: {r['director_names']} | Country: {r['primary_country']}  \n"
                    f"Rating: {r['imdb_rating']}/10"
                )
