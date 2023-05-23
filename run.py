import numpy as np
from SOM import SOM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from TFIDF import tf_idf
from sklearn.decomposition import PCA

map_size = (10, 10)
df = pd.read_excel('Data.xlsx', sheet_name='Sheet') 
docs=df['متن'].to_list()
docs_test=input()

docs=[tf_idf(doc) for doc in docs]
vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(docs).toarray()

docs_vector=[]
#pca = PCA(n_components=2)
#docs_vector = pca.fit_transform(docs)
# Create an instance of the SOM class
input_dim = tfidf_vectors.shape[1]
som = SOM(input_dim, map_size)

# Train the SOM on the document vectors
num_epochs = 10
som.train(tfidf_vectors, num_epochs)

docs_test=[tf_idf(doc) for doc in docs_test]
test_tfidf_vectors = vectorizer.transform(docs_test).toarray()

cluster_labels = som.cluster(test_tfidf_vectors)
print(cluster_labels)

# Generate some test documents for clustering
#  # Replace this with your actual test document data

# Preprocess the test documents
#preprocessed_test_documents = [' '.join(word_tokenize(document)) for document in test_documents]

# Compute the document vectors for the test documents


