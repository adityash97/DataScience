import pandas as pd
from gensim.models import Word2Vec
import string
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
path = "../../resources/IndianFoodDatasetCSV.csv"
sampled_data = pd.read_csv(path)

stemmer_instance = nltk.stem.PorterStemmer()
stop_words = stopwords.words('english')
# print("****"*20,stopwords.words('english'),"****"*20)

stemmer_instance = nltk.stem.PorterStemmer()
stop_words = stopwords.words('english')

def tokenize_recipe(text):
    text = str(text)
    for punctuation_char in string.punctuation:
        text = text.replace(punctuation_char, '').lower()
    
    words = text.split(' ')
    stemmed_words = []
    
    for word in words:
        if (word not in stop_words) and (word != ''):
            stemmed_words.append(stemmer_instance.stem(word))
    
    return stemmed_words

def generate_word_embeddings(dataframe, text_column):
    tokenized_text = dataframe[text_column].apply(tokenize_recipe)
    word2vec_model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = {key: word2vec_model.wv[key] for key in word2vec_model.wv.index_to_key}
    
    return word_vectors

def compute_and_store_embeddings(data_frame):
    ingredient_embeddings_model = generate_word_embeddings(data_frame, 'TranslatedIngredients')
    data_frame['combined_text'] = data_frame[['TranslatedRecipeName', 'Diet', 'TranslatedInstructions']].astype(str).agg(' '.join, axis=1)
    data_frame['combined_text'] = data_frame['combined_text'].str.lower()

    tfidf_model = TfidfVectorizer(min_df=5, tokenizer=tokenize_recipe)
    vectorized_text = tfidf_model.fit_transform(data_frame['combined_text'])

    ingredient_vectors = [
        np.mean([ingredient_embeddings_model[word] for word in tokenize_recipe(ingredient_list) if word in ingredient_embeddings_model] 
                or [np.zeros(100)], axis=0) 
        for ingredient_list in data_frame['TranslatedIngredients']
    ]

    final_embeddings = np.concatenate([vectorized_text.toarray(), np.array(ingredient_vectors)], axis=1)

    with open('final_embeddings.pkl', 'wb') as embedding_file:
        pickle.dump(final_embeddings, embedding_file)
    
    with open('tfidf_model.pkl', 'wb') as tfidf_file:
        pickle.dump(tfidf_model, tfidf_file)
    
    print("Embeddings and TF-IDF vectorizer successfully saved!")
    
embeddings = generate_word_embeddings(sampled_data, 'TranslatedIngredients')



sampled_data['combined_text'] = sampled_data[['TranslatedRecipeName', 'Diet', 'TranslatedInstructions']].astype(str).agg(' '.join, axis=1)


sampled_data['combined_text'] = sampled_data['combined_text'].str.lower()

tfidf_vectorizer = TfidfVectorizer(min_df=5, tokenizer=tokenize_recipe)
vectorized_text_data = tfidf_vectorizer.fit_transform(sampled_data['combined_text'])

def tokenize_recipe(text):
    text = str(text)
    for punctuation_char in string.punctuation:
        text = text.replace(punctuation_char, '').lower()
    
    words = text.split(' ')
    stemmed_words = []
    
    for word in words:
        if (word not in stop_words) and (word != ''):
            stemmed_words.append(stemmer_instance.stem(word))
    
    return stemmed_words

def load_combined_data_and_model():
    with open('final_embeddings.pkl', 'rb') as file:
        combined_data = pickle.load(file)
    with open('tfidf_model.pkl', 'rb') as file:
        tfidf_model = pickle.load(file)
    return combined_data, tfidf_model


def search_similar_recipes(data_frame, user_query, top_n=5):
    compute_and_store_embeddings(data_frame)
    combined_data, tfidf_model = load_combined_data_and_model()
    # try:
    #     combined_data, tfidf_model = load_combined_data_and_model()
    # except FileNotFoundError:
    #     return "Sorry. Recommendation engine is missing."
    
    query_frame = pd.DataFrame({'combined_text': [str(user_query)]})
    query_frame['combined_text'] = query_frame['combined_text'].str.lower()
    query_vectorized = tfidf_model.transform(query_frame['combined_text'])

    feature_gap = combined_data.shape[1] - query_vectorized.shape[1]
    if feature_gap > 0:
        query_vectorized = np.pad(query_vectorized.toarray(), ((0, 0), (0, feature_gap)))

    similarity_scores = cosine_similarity(query_vectorized, combined_data)
    top_matches = similarity_scores[0].argsort()[::-1][:top_n]
    matched_recipe_names = data_frame.iloc[top_matches]['TranslatedRecipeName'].tolist()
    # import pdb;pdb.set_trace()
    return matched_recipe_names

recipes = search_similar_recipes(sampled_data,"spicy chicken in chinese style")

print("recipes : ",recipes)

recommendations = sampled_data.loc[sampled_data['TranslatedRecipeName'].isin(recipes), ['TranslatedRecipeName', 'Diet','Cuisine','TranslatedInstructions','CookTimeInMins']]