# Importing relevant libraries
import pandas as pd #For data manipulation and analysis
from sklearn.feature_extraction.text import CountVectorizer # To convert text to numerical feature vectors
from sklearn.metrics.pairwise import cosine_similarity # To compute similarity between feature vectors
import streamlit as st # To build an interactive web app UI

# -------------------------------
# Load and prepare dataset
# -------------------------------

df = pd.read_csv("imdb-movies-dataset.csv")

# Keeping the relevant columns for similarity calculation
relevant_columns = ['title','genre','director','cast']
df = df[relevant_columns]

# Fills missing values with empty string to avoid errors
df = df.fillna({"cast":"", "genre":"", "director":""})

# ----------------------------------------
# Helper Functions
# ----------------------------------------

def validate_movie_title(title):
    """
    Ensures the entered movie title exists in the dataset.
    Returns True if it exists, and False otherwise with a message.  
    """
    matches = df[df.title.str.lower() == title.lower()]
    if matches.empty:
        st.write("Sorry, that movie is not in the dataset. Please check your spelling or try another title.")
        return False
    return True 

def get_title_from_index(index):
    """
    Returns the title of a movie given the corresponding index.
    """
    return df.iloc[index]["title"]

def get_index_from_title(title):
    """
    Returns the index of a movie given the corresponding title.
    """
    return df[df.title.str.lower() == title.lower()].index[0]


def combine_features(row, cast_weight, genre_weight, director_weight):
    """
    Combines selected movie features into a single string.
    Then applies customizable weights from user to emphasize certain features more than others.
    """
    try:
        # Takes the top 3 cast members and replaces space with underscore to combine first and last names
        cast = " ".join(c.strip().replace(" ","_") for c in row["cast"].split(",")[:3]) 
        # Takes the top 2 genres 
        genres = " ".join(g.strip() for g in row['genre'].split(",")[:2])
        # Combines director's first and last name into one string
        director = row['director'].replace(" ", "_")
        # Applies weights to each feature
        return (cast + " ")*cast_weight*2 + (genres + " ")*genre_weight + (director + " ")*director_weight
    except:
        # Returns empty string if there is an error
        return ""

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Movie Recomendation System")

movie_user_likes = st.text_input("Enter your favourite movie name: ").lower()

st.write("Rate the importance of each feature on a scale from 1(least) to 5(most).")
cast_weight = st.slider("How important is the cast? ", 1,5,1)
genre_weight = st.slider("How important is the genre? ", 1,5,1)
director_weight = st.slider("How important is the director? ", 1,5,1)

# -------------------------------
# Process goes through only if the movie is entered and valid
# -------------------------------

if movie_user_likes:
    if validate_movie_title(movie_user_likes):
        # Gets index of selected movie
        movie_index = get_index_from_title(movie_user_likes)
        # Combines all features for each movie using specified weights
        df["combined_features"] = df.apply(lambda row: combine_features(row, cast_weight, genre_weight, director_weight),axis=1)

        # Converts combined text features into numerical vectors
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df["combined_features"])

        # Computes cosine similarity between all movies
        cosine_sim = cosine_similarity(count_matrix)

        # Creates a list of movies with similarity scores
        similar_movies = list(enumerate(cosine_sim[movie_index]))

        # Sorts movies by order (most similar first)
        sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)


        # -------------------------------
        # Display recommendations
        # -------------------------------

        selected_movie = get_title_from_index(movie_index)
        st.write(f"Featured Weights - Cast: {cast_weight}, Genre: {genre_weight}, Director: {director_weight}.\n")
        st.write(f"Here are the top 10 films similar to:")
        st.markdown(f"## 🎬 {selected_movie}")
        num = 1
        # Display top 10 similar movies
        for movie in sorted_similar_movies[num:11]:
            st.write(f"{num}. {get_title_from_index(movie[0])}")
            num += 1

