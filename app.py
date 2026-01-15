import streamlit as st
import pickle
import pandas as pd
import requests

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
TMDB_API_KEY = "6e7b26793831f96a52595f6539e84908"

# -------------------------------
# LOAD DATA
# -------------------------------
similarity = pickle.load(open("similarity.pkl", "rb"))
movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

# -------------------------------
# SESSION STATE
# -------------------------------
if "all_recommendations" not in st.session_state:
    st.session_state.all_recommendations = None

if "visible_count" not in st.session_state:
    st.session_state.visible_count = 5

# -------------------------------
# FUNCTIONS
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={TMDB_API_KEY}&language=en-US"
    )
    data = requests.get(url).json()
    poster_path = data.get("poster_path")

    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path
    return None


def recommend(movie_title, top_n=50):
    movie_index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:21]   # 🔹 generate 20 once

    results = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]]["id"]
        results.append({
            "title": movies.iloc[i[0]]["title"],
            "poster": fetch_poster(movie_id)
        })

    return results

# -------------------------------
# UI
# -------------------------------
st.title("🎬 Movie Recommender System")

selected_movie_name = st.selectbox(
    "Select a movie",
    movies["title"].values
)

if st.button("recommend"):
    with st.spinner("Fetching recommendations..."):
        st.session_state.all_recommendations = recommend(
            selected_movie_name,
            top_n=50
        )
        st.session_state.shown_count = 10


# -------------------------------
# DISPLAY RESULTS
# -------------------------------
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

    # -------------------------------
    # LOAD MORE BUTTON
    # -------------------------------
    if count < len(recs):
        if st.button("Load More"):
            st.session_state.visible_count += 5
