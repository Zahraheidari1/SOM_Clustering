import numpy as np
from SOM import SOM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from functions import preprocess_documents
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# Read the Excel file and extract the documents
df = pd.read_excel('Data.xlsx', sheet_name='Sheet') 
docs = df['متن'].to_list()


docs, docs_test = train_test_split(docs, test_size=0.2, random_state=42) 

docs = preprocess_documents(docs)
docs_test = preprocess_documents(docs_test)

vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(docs).toarray()

pca = PCA(n_components=2)
docs_vector = pca.fit_transform(tfidf_vectors)
# Create  SOM class
map_size = (10, 10)
input_dim = docs_vector.shape[1]
som = SOM(input_dim, map_size)

# Train the SOM on the document vectors
num_epochs = 10
som.train(docs_vector, num_epochs)

docs_test = preprocess_documents(docs_test)
test_tfidf_vectors = vectorizer.transform(docs_test).toarray()
test_docs_vector = pca.transform(test_tfidf_vectors)

cluster_labels = som.cluster(test_docs_vector)
print(cluster_labels)

#len(cluster_labels)