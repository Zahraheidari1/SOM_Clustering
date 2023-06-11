import numpy as np
import pandas as pd
from hazm.utils import stopwords_list
from hazm import WordTokenizer, Lemmatizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from SOM import SOM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_documents(documents):
    tokenizer = WordTokenizer()
    lemmatizer = Lemmatizer()
    stop_words = stopwords_list()
    preprocessed_docs = []

    batch_size = 100  # Number of documents to process in each batch

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]

        # Tokenize, lemmatize, and remove stopwords for each document in the batch
        batch_tokens = []
        for doc in batch_docs:
            tokens = tokenizer.tokenize(doc)
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            tokens = [token for token in tokens if token not in stop_words]
            text_clean = ' '.join(tokens)
            batch_tokens.append(text_clean)

        preprocessed_docs.extend(batch_tokens)

    return preprocessed_docs

def generate_cluster_labels(filename,num_clusters):
    # Read the Excel file and extract the documents
    df = pd.read_excel(filename, sheet_name='Sheet')
    docs = df['متن'].to_list()
    title = df['عنوان'].drop_duplicates().tolist()
    
    # Split the data into training and test sets
    docs_train, docs_test = train_test_split(docs, test_size=0.2, random_state=42)

    # Preprocess the training and test documents
    docs_train = preprocess_documents(docs_train)
    docs_test = preprocess_documents(docs_test)

    # Convert documents to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(docs_train).toarray()

    # Perform PCA on the TF-IDF vectors
    pca = PCA(n_components=2)
    docs_vector = pca.fit_transform(tfidf_vectors)

    # Create SOM instance
    map_size = (10,10)
    input_dim = docs_vector.shape[1]
    som = SOM(input_dim, map_size,num_clusters)

    # Train the SOM on the document vectors
    num_epochs = 10
    som.train(docs_vector, num_epochs)

    u_matrix = som.get_u_matrix()

    # Preprocess the test documents
    docs_test = preprocess_documents(docs_test)

    # Convert test documents to TF-IDF vectors
    test_tfidf_vectors = vectorizer.transform(docs_test).toarray()
    test_docs_vector = pca.transform(test_tfidf_vectors)
    similarity = cosine_similarity(test_docs_vector)[0][1]

    # Cluster the test documents using the trained SOM
    cluster_labels = som.cluster(test_docs_vector)

    # Write the cluster labels to the output text file
    output_file_path = "output.txt"
    with open(output_file_path, "w") as file:
        for label in cluster_labels:
            file.write(str(label) + "\n")

    print("Output has been saved to", output_file_path)

    # Save test_docs_vector to a text file
    np.savetxt("test_docs_vector.txt", test_docs_vector)

    return  test_docs_vector,  cluster_labels ,u_matrix
