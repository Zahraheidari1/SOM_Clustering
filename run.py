import numpy as np
from SOM import SOM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from hazm import word_tokenize
import hazm
from hazm.utils import stopwords_list

map_size = (10, 10)
df = pd.read_excel('Data.xlsx', sheet_name='Sheet') 
docs=df['متن'].to_list()
docs_test=input()

tokenizer = hazm.WordTokenizer()
tokens1 = tokenizer.tokenize(docs)
tokens2 = tokenizer.tokenize(docs_test)
lemmatizer = hazm.Lemmatizer()
tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
stop_words = stopwords_list()
tokens1 = [token for token in tokens1 if token not in stop_words]
tokens2 = [token for token in tokens2 if token not in stop_words]


    # Create the TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(tokens1).toarray()
# Create an instance of the SOM class
input_dim = tfidf_vectors.shape[1]
som = SOM(input_dim, map_size)

# Train the SOM on the document vectors
num_epochs = 10
som.train(tfidf_vectors, num_epochs)

test_tfidf_vectors = vectorizer.transform(tokens2).toarray()

cluster_labels = som.cluster(test_tfidf_vectors)
print(cluster_labels)

# Generate some test documents for clustering
#  # Replace this with your actual test document data

# Preprocess the test documents
#preprocessed_test_documents = [' '.join(word_tokenize(document)) for document in test_documents]

# Compute the document vectors for the test documents


