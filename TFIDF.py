from sklearn.metrics.pairwise import cosine_similarity
import hazm
from hazm.utils import stopwords_list
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(documents):
    tokenizer = hazm.WordTokenizer()
    tokens = tokenizer.tokenize(documents)
    lemmatizer = hazm.Lemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stopwords
    stop_words = stopwords_list()
    tokens = [token for token in tokens if token not in stop_words]

    # Convert the token lists back to strings
    text_clean = ' '.join(tokens)

    return text_clean