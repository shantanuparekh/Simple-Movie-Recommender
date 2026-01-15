import streamlit as st
import pickle
import pandas as pd
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_movies():
    movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
    return pd.DataFrame(movies_dict)

movies = load_movies()

# -------------------------------------------------
# COMPUTE SIMILARITY (NO PICKLE FILE)
# -------------------------------------------------
@st.cache_resource
def compute_similarity(movies_df):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies_df["tags"]).toarray()
    return cosine_similarity(vectors)

similarity = compute_similarity(movies)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "all_recommendations" not in st.session_state:
    st.session_state.all_recommendations = None

if "visible_count" not in st.session_state:
    st.session_state.visible_count = 5

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={TMDB_API_KEY}&language=en-US"
    )
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    poster_path = data.get("poster_path")

    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path
    return None


def recommend(movie_title):
    movie_index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:21]  # generate top 20 once

    results = []
    for i in movie_list:
        results.append({
            "title": movies.iloc[i[0]]["title"],
            "poster": fetch_poster(movies.iloc[i[0]]["id"])
        })

    return results

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

if st.button("Recommend"):
    st.session_state.all_recommendations = recommend(selected_movie)
    st.session_state.visible_count = 5

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------
if st.session_state.all_recommendations:
    st.subheader("Recommended Movies")

    recs = st.session_state.all_recommendations
    count = st.session_state.visible_count

    cols = st.columns(5)
    for idx in range(min(count, len(recs))):
        with cols[idx % 5]:
            if recs[idx]["poster"]:
                st.image(recs[idx]["poster"])
            st.caption(recs[idx]["title"])

    if count < len(recs):
        if st.button("Load More"):
            st.session_state.visible_count += 5
