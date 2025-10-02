This is a content-based movie recommendation project built in Python, showcasing data analysis, feature engineering, and interactive UI skills.




-----PROJECT OVERVIEW-----

This project uses a CSV dataset of over 10,000 movies to build a content-based movie recommendation system.

  - Recommends movies similar to a given title from the user, based on cast, genre, and director
  - Uses cosine similarity on the vectorized features 
  - Includes customizable feature weightings for a more personalized recommendation
  - Features a simple interactive UI using Streamlit, which allows for an easy run and demonstration of results

-----FEATURES DEMONSTRATED-----

  - Data cleaning, preprocessing, and feature extraction with Pandas
  - Text vectorization and similarity computation using scikit-learn
  - Implementation of feature weighting for customizable recommendations
  - Interactive front-end prototype with Streamlit

-----TECHNOLOGIES USED-----

  - Python (language)
  - Pandas (data manipulation and cleaning)
  - scikit-learn (CountVectorizer and cosine similarity)
  - Streamlit (UI prototyping)


-----INSTALLATION AND USAGE-----

1. Clone the repository:
```bash
git clone https://github.com/kai-eren/movie-recommendation.git
cd movie-recommendation
```

2. Install dependencies:
pip install pandas scikit-learn streamlit

3. Run the app:
streamlit run movierec_main.py

4. Enter a movie title and adjust the features using sliders to see recommended movies.



