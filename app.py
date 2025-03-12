from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load dataset
DATASET_PATH = "imdb_top_1000.csv"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
movies_df = pd.read_csv(DATASET_PATH)

# Fill missing values
movies_df.fillna("", inplace=True)

# Combine important text features, including movie overview and IMDb rating
movies_df["combined_features"] = (
    movies_df["Genre"] + " " +
    movies_df["Director"] + " " +
    movies_df["Star1"] + " " +
    movies_df["Star2"] + " " +
    movies_df["Overview"] + " " +
    movies_df["IMDB_Rating"].astype(str)  # Convert numerical rating to string for text processing
)

# TF-IDF Vectorizer to convert text to feature vectors
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df["combined_features"])

# Compute cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/recommendations')
def recommendations():
    movie_title = request.args.get('movie_title', default='Inception', type=str)
    print(f"Received request for movie title: {movie_title}")  # Debugging information
    
    # Find the index of the movie in the dataset
    indices = movies_df[movies_df["Series_Title"].str.lower() == movie_title.lower()].index
    print(f"Found indices: {indices}")  # Debugging information
    
    if indices.empty:
        print("Movie not found in dataset")  # Debugging information
        return {"error": "Movie not found in dataset"}, 404
    
    movie_idx = indices[0]

    # Get similarity scores for this movie with all other movies
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    print(f"Similarity scores: {sim_scores}")  # Debugging information

    # Sort movies based on similarity scores (highest to lowest)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]  # Top 3 similar movies
    print(f"Sorted similarity scores: {sim_scores}")  # Debugging information

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    print(f"Movie indices: {movie_indices}")  # Debugging information

    # Get recommended movies
    recommended_movies = movies_df.iloc[movie_indices][["Series_Title", "Released_Year", "Genre", "IMDB_Rating", "Director", "Overview", "Poster_Link"]].to_dict(orient="records")
    print(f"Recommended movies: {recommended_movies}")  # Debugging information

    return render_template('recommendations.html', movies=recommended_movies), 200

if __name__ == '__main__':
    app.run(debug=True)
