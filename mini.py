import numpy as np
import pandas as pd
from hazm.utils import stopwords_list
from hazm import WordTokenizer, Lemmatizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def convert_to_base(number_list, num_clusters):
    converted_numbers = []
    for number in number_list:
        converted_number = number[0] * 10 + number[1] + 1
        converted_numbers.append((converted_number - 1) % num_clusters + 1)
    
    return converted_numbers

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

def calculate_cluster_accuracy(cluster_labels, ground_truth_labels):
    """
    Calculate the percentage of correctly recognized test data for each cluster and the overall accuracy.

    Args:
    cluster_labels (list): List of cluster labels assigned by the clustering algorithm.
    ground_truth_labels (list): List of ground truth labels.

    Returns:
    pandas.DataFrame: Table showing the percentage of correctly recognized test data for each cluster
                      and the overall accuracy.
    """
    data = {'Cluster': cluster_labels, 'Ground Truth': ground_truth_labels}
    df = pd.DataFrame(data)

    cluster_counts = df.groupby(['Cluster', 'Ground Truth']).size().unstack(fill_value=0)
    cluster_totals = cluster_counts.sum(axis=1)
    cluster_accuracy = cluster_counts.max(axis=1) / cluster_totals * 100

    total_correct = cluster_counts.max(axis=1).sum()
    total_samples = cluster_counts.sum().sum()
    total_accuracy = total_correct / total_samples * 100

    table = pd.DataFrame({'Correctly Recognized': cluster_counts.max(axis=1),
                          'Total': cluster_totals,
                          'Accuracy (%)': cluster_accuracy}).reset_index()

    overall_accuracy = pd.DataFrame({'Correctly Recognized': [total_correct],
                                     'Total': [total_samples],
                                     'Accuracy (%)': [total_accuracy],
                                     'Cluster': ['Overall']})

    table = pd.concat([table, overall_accuracy], ignore_index=True)

    return table



def generate_cluster_labels(filename):
    # Read the Excel file and extract the documents and ground truth labels
    df = pd.read_excel(filename, sheet_name='Sheet')
    docs = df['متن'].to_list()
    ground_truth_labels = df['عنوان'].to_list()
    
    # Split the data into training and test sets
    docs_train, docs_test, ground_truth_train, ground_truth_test = train_test_split(
        docs, ground_truth_labels, test_size=0.2, random_state=42)

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
    map_size = (10, 10)
    input_dim = docs_vector.shape[1]
    num_clusters = 12
    som = MiniSom(map_size[0], map_size[1], input_dim, sigma=0.3, learning_rate=0.7)

    # Initialize the weights using random values
    som.random_weights_init(docs_vector)

    # Train the SOM on the document vectors
    num_epochs = 10
    for _ in range(num_epochs):
        som.train_random(docs_vector,num_iteration=500)

    # Get the U-matrix
    u_matrix = som.distance_map()

    # Preprocess the test documents
    docs_test = preprocess_documents(docs_test)

    # Convert test documents to TF-IDF vectors
    test_tfidf_vectors = vectorizer.transform(docs_test).toarray()
    test_docs_vector = pca.transform(test_tfidf_vectors)

    # Cluster the test documents using the trained SOM
    cluster_labels = []
    for doc_vector in test_docs_vector:
        winner = som.winner(doc_vector)
        cluster_labels.append(winner)

    # Convert cluster indices to cluster labels between 1 and 12
    cluster_labels = convert_to_base(cluster_labels, num_clusters)

    # Write the cluster labels to the output text file
    output_file_path = "output.txt"
    with open(output_file_path, "w") as file:
        for label, doc in zip(cluster_labels, docs_test):
            file.write(f"Cluster {label}: {doc}\n")

    print("Output has been saved to", output_file_path)

    # Save test_docs_vector to a text file
    np.savetxt("test_docs_vector.txt", test_docs_vector)

    # Evaluate clustering accuracy
    table = calculate_cluster_accuracy(cluster_labels, ground_truth_test)
    print(table)
    
    return test_docs_vector, cluster_labels, u_matrix
