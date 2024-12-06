import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
import string
import numpy as np
import nltk
import os


nltk.download("stopwords")
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("stopwords")
path = os.path.join(os.path.dirname(__file__), 'IndianFoodDatasetCSV.csv')
output_dir = os.path.abspath("output_files")
os.makedirs(output_dir, exist_ok=True)

# path = "./IndianFoodDatasetCSV.csv"
sampled_data = pd.read_csv(path)

stemmer_instance = nltk.stem.PorterStemmer()
stop_words = stopwords.words("english")


def tokenize_recipe(text):
    text = str(text)
    for punctuation_char in string.punctuation:
        text = text.replace(punctuation_char, "").lower()

    words = text.split(" ")
    stemmed_words = []

    for word in words:
        if (word not in stop_words) and (word != ""):
            stemmed_words.append(stemmer_instance.stem(word))

    return stemmed_words


@st.cache_data
def load_combined_data_and_model():
    file_dir = "output_files"  
    
    # Construct file paths
    embeddings_path = os.path.join(file_dir, "final_embeddings.pkl")
    tfidf_model_path = os.path.join(file_dir, "tfidf_model.pkl")
    
    with open(embeddings_path, "rb") as file:
        combined_data = pickle.load(file)
    with open(tfidf_model_path, "rb") as file:
        tfidf_model = pickle.load(file)
    return combined_data, tfidf_model


def generate_word_embeddings(dataframe, text_column):
    tokenized_text = dataframe[text_column].apply(tokenize_recipe)
    word2vec_model = Word2Vec(
        tokenized_text, vector_size=100, window=5, min_count=1, workers=4
    )
    word_vectors = {
        key: word2vec_model.wv[key] for key in word2vec_model.wv.index_to_key
    }

    return word_vectors


def compute_and_store_embeddings(data_frame):
    ingredient_embeddings_model = generate_word_embeddings(
        data_frame, "TranslatedIngredients"
    )
    data_frame["combined_text"] = (
        data_frame[["TranslatedRecipeName", "Diet", "TranslatedInstructions"]]
        .astype(str)
        .agg(" ".join, axis=1)
    )
    data_frame["combined_text"] = data_frame["combined_text"].str.lower()

    tfidf_model = TfidfVectorizer(min_df=5, tokenizer=tokenize_recipe)
    vectorized_text = tfidf_model.fit_transform(data_frame["combined_text"])

    ingredient_vectors = [
        np.mean(
            [
                ingredient_embeddings_model[word]
                for word in tokenize_recipe(ingredient_list)
                if word in ingredient_embeddings_model
            ]
            or [np.zeros(100)],
            axis=0,
        )
        for ingredient_list in data_frame["TranslatedIngredients"]
    ]

    final_embeddings = np.concatenate(
        [vectorized_text.toarray(), np.array(ingredient_vectors)], axis=1
    )

    with open(os.path.join(output_dir, "final_embeddings.pkl"), "wb") as embedding_file:
        pickle.dump(final_embeddings, embedding_file)

    with open(os.path.join(output_dir, "tfidf_model.pkl"), "wb") as tfidf_file:
        pickle.dump(tfidf_model, tfidf_file)

    print("Embeddings and TF-IDF vectorizer successfully saved!")


def search_similar_recipes(data_frame, user_query, top_n=5):

    try:
        combined_data, tfidf_model = load_combined_data_and_model()
    except FileNotFoundError:
        compute_and_store_embeddings(data_frame)
        combined_data, tfidf_model = load_combined_data_and_model()

    query_frame = pd.DataFrame({"combined_text": [str(user_query)]})
    query_frame["combined_text"] = query_frame["combined_text"].str.lower()
    query_vectorized = tfidf_model.transform(query_frame["combined_text"])

    feature_gap = combined_data.shape[1] - query_vectorized.shape[1]
    if feature_gap > 0:
        query_vectorized = np.pad(
            query_vectorized.toarray(), ((0, 0), (0, feature_gap))
        )

    similarity_scores = cosine_similarity(query_vectorized, combined_data)
    top_matches = similarity_scores[0].argsort()[::-1][:top_n]
    matched_recipe_names = data_frame.iloc[top_matches]["TranslatedRecipeName"].tolist()
    return matched_recipe_names


# Streamlit UI

# Sidebar with description
st.sidebar.title("Smart Recipe Recommender")
st.sidebar.write(
    """
This application will help you find recipe recommendations based on your input.
Enter a recipe name and get similar recipe suggestions.
"""
)

# Main header and user input
st.title("Smart Recipe Recommender")
st.subheader("Enter a recipe name to get recommendations")

# Input box for recipe name
recipe_input = st.text_input("Recipe Name", "")


# When the user enters a recipe name
if recipe_input:
    # Get recommendations based on the recipe name
    recipes = search_similar_recipes(sampled_data, recipe_input)

    recommendations = sampled_data.loc[
        sampled_data["TranslatedRecipeName"].isin(recipes),
        [
            "TranslatedRecipeName",
            "Diet",
            "Cuisine",
            "TranslatedInstructions",
            "CookTimeInMins",
        ],
    ]

    # Display recommendations
    st.subheader(f"Recommended recipes for '{recipe_input}':")

    # for rec in recommendations:
    st.write(recommendations)
else:
    st.write("Please enter a recipe name to get recommendations.")
