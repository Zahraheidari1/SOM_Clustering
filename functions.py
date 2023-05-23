from sklearn.metrics.pairwise import cosine_similarity
import hazm
from hazm.utils import stopwords_list
from sklearn.feature_extraction.text import TfidfVectorizer

def text_similarity(text1, text2):
    # Tokenize and lemmatize the texts
    tokenizer = hazm.WordTokenizer()
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)
    lemmatizer = hazm.Lemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords_list()
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Convert the token lists back to strings
    text1_clean = ' '.join(tokens1)
    text2_clean = ' '.join(tokens2)

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform([text1_clean, text2_clean])

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1)[0][1]

    return similarity
