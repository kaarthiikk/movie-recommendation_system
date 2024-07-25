import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies_data = pd.read_csv("moviesdataset.csv")
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Preprocess content-based recommendation data
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# Preprocess collaborative-based recommendation data
ratings = pd.merge(movies, ratings).drop(['genres', 'timestamp'], axis=1)
user_ratings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
user_ratings = user_ratings.dropna(thresh=10, axis=1).fillna(0, axis=1)
item_similarity_df = user_ratings.corr(method='pearson')

# Functions for content-based recommendations
def get_content_based_recommendations(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        return ["No match found."]
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    return [movies_data[movies_data.index == movie[0]]['title'].values[0] for movie in sorted_similar_movies[:30]]

# Functions for collaborative-based recommendations
def get_similar_movies(movie_name, user_rating):
    if movie_name not in item_similarity_df.columns:
        return pd.Series(dtype=float)  # Return an empty series if movie not found
    similar_score = item_similarity_df[movie_name] * (user_rating - 2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

def get_collaborative_based_recommendations(movies_and_ratings):
    similar_movies = pd.DataFrame()
    for movie, rating in movies_and_ratings.items():
        similar_movies = similar_movies.append(get_similar_movies(movie, rating), ignore_index=True)
    return similar_movies.sum().sort_values(ascending=False)

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Home", "Prediction"])

    if page == "Home":
        show_home()
    elif page == "Prediction":
        show_prediction()

def show_home():
    st.title("Movie Recommendation System")
    st.write("""
        Welcome to the Movie Recommendation System!

        This app provides movie recommendations based on two different approaches:
        - **Content-Based Recommendations**: Uses movie descriptions and features.
        - **Collaborative Recommendations**: Uses user ratings and preferences.

        Use the 'Prediction' page to enter movie information and get recommendations from both approaches.
    """)

def show_prediction():
    st.title("Movie Recommendations")

    # Content-based recommendation section
    st.subheader("Content-Based Recommendations")
    movie_name = st.text_input("Enter a movie name for content-based recommendations")

    if st.button("Get Content-Based Recommendations"):
        if movie_name:
            recommendations = get_content_based_recommendations(movie_name)
            st.write("Content-Based Recommendations:")
            st.write(recommendations)
        else:
            st.write("Please enter a movie name.")

    # Collaborative-based recommendation section
    st.subheader("Collaborative-Based Recommendations")
    st.write("Enter multiple movies and their ratings (e.g., Movie1:5, Movie2:4).")

    input_data = st.text_area("Enter movies and ratings in the format 'Movie1:rating, Movie2:rating'")
    
    if st.button("Get Collaborative-Based Recommendations"):
        if input_data:
            movies_and_ratings = parse_input(input_data)
            if movies_and_ratings:
                recommendations = get_collaborative_based_recommendations(movies_and_ratings)
                st.write("Collaborative-Based Recommendations:")
                st.write(recommendations)
            else:
                st.write("Error parsing input.")
        else:
            st.write("Please enter movie names and ratings.")

def parse_input(input_data):
    try:
        movies_and_ratings = [tuple(x.split(':')) for x in input_data.split(',')]
        return {movie.strip(): float(rating.strip()) for movie, rating in movies_and_ratings}
    except Exception as e:
        st.write(f"Error parsing input: {e}")
        return {}

if __name__ == "__main__":
    main()
