import pandas as pd
from hazm.utils import stopwords_list
from hazm import WordTokenizer, Lemmatizer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from SOM import SOM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, ParameterGrid
from bokeh.layouts import row, column
from bokeh.plotting import curdoc, figure
from main import generate_bokeh_plots

def convert_to_base(number_list, map_size):
    converted_numbers = []
    for number in number_list:
        converted_numbers.append(number[0] * map_size[1] + number[1])

    return converted_numbers

def preprocess_documents(documents):
    tokenizer = WordTokenizer()
    lemmatizer = Lemmatizer()
    stop_words = stopwords_list()
    preprocessed_docs = []

    batch_size = 1000  # Number of documents to process in each batch

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

def generate_cluster_labels(filename):
    # Read the Excel file and extract the documents
    df = pd.read_excel(filename, sheet_name='Sheet')
    docs = df['متن'].to_list()

    # Split the data into training and test sets
    docs_train, docs_test = train_test_split(docs, test_size=0.2, random_state=42)

    # Preprocess the training and test documents
    docs_train = preprocess_documents(docs_train)

    # Convert documents to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(docs_train).toarray()

    # Perform PCA on the TF-IDF vectors
    pca = PCA(n_components=2)
    docs_vector = pca.fit_transform(tfidf_vectors)

    # Create SOM instance
    input_dim = docs_vector.shape[1]
    som = SOM(input_dim, (1, 1))  # Initial map size is set to (1, 1)

    # Grid search for optimal map size and number of epochs
    param_grid = {
        'map_size': [(3, 3), (5, 5), (10, 10), (15, 15), (20, 20)],
        'num_epochs': [1, 10, 20, 50, 100]
    }

    results = {}

    for map_size in param_grid['map_size']:
        nmi_values = []
        ari_values = []
        silhouette_values = []
        quantization_errors = []
        topographic_errors = []

        for num_epochs in param_grid['num_epochs']:
            # Train the SOM on the document vectors
            som.__init__(input_dim, map_size)
            som.train(docs_vector, num_epochs)

            u_matrix = som.get_u_matrix()

            # Preprocess the test documents
            docs_test = preprocess_documents(docs_test)

            # Convert test documents to TF-IDF vectors
            test_tfidf_vectors = vectorizer.transform(docs_test).toarray()
            test_docs_vector = pca.transform(test_tfidf_vectors)

            # Cluster the test documents using the trained SOM
            cluster_labels = som.cluster(test_docs_vector)
            cluster_labels = convert_to_base(cluster_labels, map_size)

            # Calculate the evaluation metrics
            silhouette_coefficient = silhouette_score(test_docs_vector, cluster_labels)
            nmi = normalized_mutual_info_score(docs_test, cluster_labels)
            ari = adjusted_rand_score(docs_test, cluster_labels)

            # Calculate quantization error and topographic error
            quantization_err = som.quantization_error(test_docs_vector)
            topographic_err = som.topographic_error(test_docs_vector)

            # Append the evaluation metric values
            nmi_values.append(nmi)
            ari_values.append(ari)
            silhouette_values.append(silhouette_coefficient)
            quantization_errors.append(quantization_err)
            topographic_errors.append(topographic_err)

        # Save the evaluation metric values for the map size
        results[map_size] = {
            'nmi_values': nmi_values,
            'ari_values': ari_values,
            'silhouette_values': silhouette_values,
            'quantization_errors': quantization_errors,
            'topographic_errors': topographic_errors
        }

    generate_bokeh_plots(param_grid['num_epochs'], results)

    return test_docs_vector , cluster_labels



