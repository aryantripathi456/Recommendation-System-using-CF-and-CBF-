import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(page_title="Movie Recommender", layout="wide")       

st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
        color: #ffffff;
    }
    
    /* Movie Titles */
    .stMarkdown p strong {
        color: #e2e8f0;
        font-size: 1.05rem;
    }
    
    /* Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #1e293b;
        border-color: #334155;
        color: #f8fafc;
        border-radius: 8px;
    }
    
    /* Images with hover effect */
    div[data-testid="stImage"] img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: transform 0.3s ease-in-out;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_data():
    with open('data/movies_metadata.pkl', 'rb') as f:
        movies = pickle.load(f)
    print("Loaded movies metadata")
    
    with open('data/vector.pkl', 'rb') as f:
        vector = pickle.load(f)
    print("Loaded content vectors")

    with open('data/cf_preds.pkl', 'rb') as f:
        cf_preds = pickle.load(f)
    print("Loaded CF predictions")

    with open('data/cf_user_indices.pkl', 'rb') as f:
        cf_user_indices = pickle.load(f)
    print("Loaded CF user indices")

    return movies, vector, cf_preds, cf_user_indices

movies, vector, cf_preds, cf_user_indices = load_data()

# function to fetch poster from TMDB API
def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return f"https://placehold.co/500x750/1a1a2e/ffffff?text={movie_id}"        # We will just show a placeholder image if TMDB API is not set up

    try:
        url = f"https://api.tmdb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('poster_path'):
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except Exception as e:
        print(f"Error fetching poster for {movie_id}: {e}")
        
    return f"https://placehold.co/500x750/1a1a2e/ffffff?text={movie_id}"

def get_content_recommendations(movie_title, top_n=10):
    if movie_title not in movies['title'].values:
        return []
    
    movie_idx = movies[movies['title'] == movie_title].index[0]
    movie_vector = vector[movie_idx].reshape(1, -1)
    
    # Compute similarity with all movies
    similarity = cosine_similarity(movie_vector, vector).flatten()
    
    # Sort indices
    similar_indices = similarity.argsort()[::-1][1:top_n+1]
    
    # Get movies
    recommended_movies = []
    for idx in similar_indices:
        recommended_movies.append({
            'title': movies.iloc[idx]['title'],
            'tmdbId': movies.iloc[idx]['id'],
            'content_score': similarity[idx]
        })
    return recommended_movies

def hybrid_recommendations(user_id, movie_title, top_n=10):
    """
    Hybrid recommendation:
    1. Get content-based recommendations for the given movie.
    2. Weight them by the user's predicted rating for those recommended movies from Collaborative Filtering.
    """
    content_recs = get_content_recommendations(movie_title, top_n=50) # get a broader pool
    
    if not content_recs:
        return []
        
    hybrid_recs = []
    
    if user_id in cf_user_indices:
        user_row_idx = cf_user_indices.index(user_id)
        # cf_preds has users as rows and tmdbIds as columns
        # columns are string or int representations of tmdb_id depending on pivot table.
        
        user_predictions = cf_preds.iloc[user_row_idx]
        
        for rec in content_recs:
            tmdb_id = rec['tmdbId']
            # Normalize content score to some baseline (0-1)
            content_score = rec['content_score']
            
            cf_score = 0
            if tmdb_id in user_predictions.index:
                cf_score = user_predictions[tmdb_id]
                
            # Combine scores (e.g. 50-50 weight)
            # content is 0-1 and CF is usually 0-5, we will normalize CF.
            # for simplicity, we just add them or use CF to reorder
            hybrid_score = (content_score * 5) * 0.5 + cf_score * 0.5
            
            hybrid_recs.append({
                'title': rec['title'],
                'tmdbId': tmdb_id,
                'hybrid_score': hybrid_score,
                'content_score': content_score,
                'cf_score': cf_score
            })
    else:
        # Fallback to purely content-based if user is not in CF model
        for rec in content_recs:
            hybrid_recs.append({
                'title': rec['title'],
                'tmdbId': rec['tmdbId'],
                'hybrid_score': rec['content_score'],
                'content_score': rec['content_score'],
                'cf_score': 0
            })
            
    # Sort by hybrid score
    hybrid_recs = sorted(hybrid_recs, key=lambda x: x['hybrid_score'], reverse=True)
    return hybrid_recs[:top_n]


# UI

st.title("Hybrid Movie Recommender System")
st.markdown("This system uses a **Hybrid Algorithm** combining Content-Based Filtering (similar plot, genres, cast) and Collaborative Filtering (user rating predictions via SVD).")

col1, col2 = st.columns(2)
with col1:
    selected_movie = st.selectbox("Type or select a movie you like:", movies['title'].values)
with col2:
    selected_user = st.selectbox("Select a User ID (for Collaborative Filtering weighting):", cf_user_indices)

if st.button("Recommend"):
    with st.spinner("Finding the best movies for you..."):
        recs = hybrid_recommendations(selected_user, selected_movie, top_n=10)
        
        if recs:
            st.subheader(f"Because you liked '{selected_movie}', we recommend:")
            
            for i in range(0, len(recs), 5):
                cols = st.columns(5)
                for col, rec in zip(cols, recs[i:i+5]):
                    with col:
                        poster_url = fetch_poster(rec['tmdbId'])
                        st.image(poster_url, width='stretch')
                        st.markdown(f"**{rec['title']}**")
                        st.caption(f"Score: {rec['hybrid_score']:.2f}")
        else:
            st.error("No recommendations found. Try another movie.")

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
1. **Content-Based Model**: Finds 50 most similar movies based on plot, genres, and cast using TF-IDF and Cosine Similarity.
2. **Collaborative Filtering Model**: Uses Singular Value Decomposition (SVD) on user ratings to predict what you would rate those 50 similar movies.
3. **Hybrid combination**: Combines the Content Similarity Score and predicted CF Rating to present the final top 5 recommendations.
""")
