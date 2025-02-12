import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # Added PCA import
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

# Initialize NLTK
nltk.download('wordnet')
nltk.download('stopwords')

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from the environment variable
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# TMDB API configuration (replace with your key)

BASE_URL = "https://api.themoviedb.org/3"


# Improved data fetching with better genre handling
@st.cache_data
def fetch_tmdb_data():
    movies = []
    # Fetch more movies for better data distribution
    for page in [1, 2, 3, 4]:  # Increased to 4 pages (80 movies)
        response = requests.get(
            f"{BASE_URL}/movie/popular",
            params={"api_key": TMDB_API_KEY, "page": page}
        )
        if response.status_code == 200:
            movies.extend(response.json().get('results', []))

    # Get genre list
    genres = requests.get(
        f"{BASE_URL}/genre/movie/list",
        params={"api_key": TMDB_API_KEY}
    ).json().get('genres', [])

    # Create dataframe with better genre handling
    data = []
    for m in movies:
        if m.get('overview'):
            genre_ids = m.get('genre_ids', [])
            primary_genre = genre_ids[0] if genre_ids else None
            data.append({
                'title': m.get('title', 'Untitled'),
                'overview': m.get('overview'),
                'genre_id': primary_genre,
                'poster_path': m.get('poster_path', '')
            })

    df = pd.DataFrame(data)

    # Map genre IDs to names with fallback
    genre_map = {g['id']: g['name'] for g in genres}
    df['genre'] = df['genre_id'].apply(
        lambda x: genre_map.get(x, 'Unknown')
    )

    # Filter to keep only top 5 genres for better balance
    top_genres = df['genre'].value_counts().head(5).index
    df = df[df['genre'].isin(top_genres)]

    return df[['title', 'overview', 'genre', 'poster_path']].dropna()


# Enhanced text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special chars
    tokens = text.split()
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops and len(t) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


@st.cache_data
def train_models(df):
    df['clean_text'] = df['overview'].apply(preprocess_text)

    # Improved TF-IDF with n-grams
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['clean_text'])
    y = df['genre']

    # Stratified split for better class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Class-weighted Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # Clustering with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    return tfidf, nb, lr, clusters, X_pca, (X_test, y_test, nb, lr)


def main():
    st.title("Movie Genre Analyzer 🍿")
    df = fetch_tmdb_data()

    if df.empty:
        st.error("Failed to fetch movies from TMDB")
        return

    tfidf, nb, lr, clusters, X_pca, (X_test, y_test, nb_model, lr_model) = train_models(df)

    # Sidebar controls
    st.sidebar.header("Controls")
    model_choice = st.sidebar.selectbox("Select Model", ["Naive Bayes", "Logistic Regression"])
    movie_title = st.sidebar.text_input("Enter Movie Title")

    # Main interface
    col1, col2 = st.columns(2)

    with col1:
        st.header("Movie Prediction")
        if st.sidebar.button("Predict Genre"):
            if movie_title:
                try:
                    response = requests.get(
                        f"{BASE_URL}/search/movie",
                        params={"api_key": TMDB_API_KEY, "query": movie_title}
                    ).json()

                    if response['results']:
                        movie = response['results'][0]
                        overview = movie.get('overview', '')

                        if overview:
                            clean_text = preprocess_text(overview)
                            vector = tfidf.transform([clean_text])

                            model = nb if model_choice == "Naive Bayes" else lr
                            prediction = model.predict(vector)[0]

                            st.subheader(f"Predicted Genre: {prediction}")
                            accuracy = accuracy_score(y_test, model.predict(X_test))
                            st.metric("Test Accuracy", f"{accuracy:.2%}")

                            if movie.get('poster_path'):
                                st.image(
                                    f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                                    caption=movie['title'],
                                    width=300
                                )
                            else:
                                st.warning("No poster available")
                        else:
                            st.error("No overview available for prediction")
                    else:
                        st.error("Movie not found in TMDB")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
            else:
                st.warning("Please enter a movie title")

    with col2:
        st.header("Data Insights")

        st.subheader("Genre Distribution")
        genre_counts = df['genre'].value_counts()
        st.bar_chart(genre_counts)

        st.subheader("Text Clustering")
        pca_df = pd.DataFrame(X_pca, columns=['X', 'Y'])
        pca_df['Cluster'] = clusters
        pca_df['Genre'] = df['genre'].values
        st.scatter_chart(pca_df, x='X', y='Y', color='Cluster')


if __name__ == "__main__":
    main()