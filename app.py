import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies_data = pd.read_csv('moviesdataset.csv')
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
    recommended_movies = [(movies_data[movies_data.index == movie[0]]['title'].values[0],
                           movies_data[movies_data.index == movie[0]]['tagline'].values[0]) 
                          for movie in sorted_similar_movies[:10]]
    return recommended_movies

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
    return similar_movies.sum().sort_values(ascending=False).head(10)

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Home", "Info", "Prediction"])

    if page == "Home":
        show_home()
    elif page == "Info":
        show_info_graphics()
    elif page == "Prediction":
        show_prediction()

def show_home():
    st.title("🎬 Movie Recommendation System")
    st.write("""
        Welcome to the Movie Recommendation System!

        This app provides movie recommendations based on two different approaches:
        - **Content-Based Recommendations**: Uses movie descriptions and features.
        - **Collaborative Recommendations**: Uses user ratings and preferences.

        Use the 'Prediction' page to enter movie information and get recommendations from both approaches.
    """)

def show_info_graphics():
  
    st.header("Content-Based Recommendations")
    st.write("""
        **Content-Based Recommendations** use movie descriptions and features to recommend similar movies. 
        This approach analyzes the characteristics of the movie, such as genres, keywords, tagline, cast, and director, 
        and finds other movies with similar characteristics.
    """)

    st.header("Collaborative-Based Recommendations")
    st.write("""
        **Collaborative-Based Recommendations** rely on user ratings and preferences. 
        This approach identifies movies that users with similar tastes have liked. 
        By comparing the ratings given by different users, it predicts which movies might be of interest to a given user based on their preferences.
    """)

    st.header("Cosine Similarity")
    st.write("""
        **Cosine Similarity** is a metric used to measure how similar two vectors are. 
        In the context of movie recommendations, it is used to determine how similar the feature vectors of different movies are. 
        The cosine similarity score ranges from -1 to 1, where 1 indicates that the vectors are identical, 
        and 0 indicates no similarity.
    """)

    st.header("Pearson Correlation")
    st.write("""
        **Pearson Correlation** is a measure of the linear relationship between two variables. 
        In collaborative filtering, it is used to calculate the similarity between movies based on user ratings. 
        The Pearson correlation coefficient ranges from -1 to 1, where 1 indicates a perfect positive linear relationship, 
        -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.
    """)

    st.header("Dataset")
    st.write("""
        The dataset consists of three main files:
        - **moviesdataset.csv**: Contains information about movies, including features like genres, keywords, tagline, cast, and director.
        - **ratings.csv**: Contains user ratings for movies, including user IDs, movie IDs, and ratings.
        - **movies.csv**: Contains movie IDs and titles.

        The dataset is used to perform both content-based and collaborative-based recommendations, 
        leveraging movie features and user ratings to suggest relevant movies.
    """)

def show_prediction():
    st.title("Movie Recommendations")

    # Content-based recommendation section
    st.subheader("Content-Based Recommendations")
    movie_name = st.text_input("Enter a movie name for content-based recommendations")

    if st.button("Get Content-Based Recommendations"):
        if movie_name:
            recommendations = get_content_based_recommendations(movie_name)
            st.markdown("### Content-Based Recommendations")
            if recommendations == ["No match found."]:
                st.markdown("<span style='color:red; font-size:20px;'>No match found.</span>", unsafe_allow_html=True)
            else:
                for idx, (movie, tagline) in enumerate(recommendations, 1):
                    st.markdown(f"<div style='background-color:#f0f0f0; padding:10px; margin:10px 0; border-radius:5px;'>"
                                f"<span style='color:blue; font-size:20px;'>{idx}. <b>{movie}</b></span><br>"
                                f"<span style='color:gray; font-size:16px;'>{tagline}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red; font-size:20px;'>Please enter a movie name.</span>", unsafe_allow_html=True)

    # Collaborative-based recommendation section
    st.subheader("Collaborative-Based Recommendations")
    st.write("Enter up to 5 movies and their ratings ")

    movie_columns = [f"Movie {i}" for i in range(1, 6)]
    rating_columns = [f"Rating {i}" for i in range(1, 6)]

    col1, col2 = st.columns(2)
    with col1:
        movies_input = [st.text_input(f"Movie {i}") for i in range(1, 6)]
    with col2:
        ratings_input = [st.number_input(f"Rating {i}", min_value=0.0, max_value=5.0, step=0.5) for i in range(1, 6)]

    if st.button("Get Collaborative-Based Recommendations"):
        movies_and_ratings = {movie: rating for movie, rating in zip(movies_input, ratings_input) if movie}
        if movies_and_ratings:
            recommendations = get_collaborative_based_recommendations(movies_and_ratings)
            st.markdown("### Collaborative-Based Recommendations")
            for idx, movie in enumerate(recommendations.index, 1):
                st.markdown(f"<div style='background-color:#e0ffe0; padding:10px; margin:10px 0; border-radius:5px;'>"
                            f"<span style='color:green; font-size:20px;'>{idx}. <b>{movie}</b></span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red; font-size:20px;'>Please enter at least one movie and its rating.</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
