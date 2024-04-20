import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example DataFrames
data = {'words': ['apple', 'banana', 'cherry']}
dataframe1 = pd.DataFrame(data)

data2 = {'words': ['fruit', 'car', 'banana']}
dataframe2 = pd.DataFrame(data2)

# Load TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Function to find the most similar word
def find_most_similar(input_word):
    # Combine the words for vectorization
    combined_words = list(dataframe1['words']) + list(dataframe2['words'])
    
    # Fit the vectorizer and transform the data
    word_vectors = vectorizer.fit_transform(combined_words)
    
    # Get the vector for the input word
    input_vec = word_vectors[dataframe1[dataframe1['words'] == input_word].index[0]]
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(input_vec, word_vectors[len(dataframe1):])
    
    # Get the index of the most similar word from dataframe2
    most_similar_index = similarity_scores.argsort()[0][-1]
    
    # Return the most similar word
    return dataframe2.iloc[most_similar_index]['words']

# Streamlit UI
st.title('Word Similarity Search')
selected_word = st.selectbox('Select a word from DataFrame 1:', dataframe1['words'])

if st.button('Find Similar Word in DataFrame 2'):
    most_similar_word = find_most_similar(selected_word)
    st.write(f'The most similar word to "{selected_word}" in DataFrame 2 is "{most_similar_word}".')
