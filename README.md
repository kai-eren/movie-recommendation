# Content-Based Movie Recommendation System

A Python project demonstrating data analysis, feature engineering, and interactive UI development.

## Project Overview
Builds a content-based movie recommendation system using a CSV of 10,000+ movies:  

- Recommends similar movies based on cast, genre, and director.  
- Uses cosine similarity on vectorized features.  
- Customizable feature weightings for personalized recommendations.  
- Simple interactive UI with Streamlit for demonstration.

## Skills Demonstrated
- Data cleaning, preprocessing, and feature extraction with Pandas.  
- Text vectorization and similarity computation using scikit-learn.  
- Implementing feature weighting for enhanced recommendations.  
- Front-end prototype using Streamlit.

- ## Technologies
Python | Pandas | scikit-learn | Streamlit

## Key Insights / Results
- Recommendations successfully highlight similar movies based on user-selected features.  
- Interactive UI allows quick exploration of movie similarities.


## Installation & Usage
1.
```bash
git clone https://github.com/kai-eren/movie-recommendation.git
cd movie-recommendation
pip install pandas scikit-learn streamlit
streamlit run movierec_main.py
```
2. Enter a movie title.

3. Adjust feature sliders to see recommended movies.
