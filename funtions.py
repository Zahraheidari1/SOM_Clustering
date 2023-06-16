import pandas as pd
from hazm.utils import stopwords_list
from hazm import WordTokenizer, Lemmatizer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from SOM import SOM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, ParameterGrid
from bokeh.layouts import row, column
from bokeh.models import Select
from bokeh.plotting import figure, show

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

    best_nmi = 0.0
    best_map_size = None
    best_num_epochs = None

    nmi_values = []
    ari_values = []
    silhouette_values = []

    for params in ParameterGrid(param_grid):
        map_size = params['map_size']
        num_epochs = params['num_epochs']

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

        # Append the evaluation metric values
        nmi_values.append(nmi)
        ari_values.append(ari)
        silhouette_values.append(silhouette_coefficient)

        # Print evaluation metrics for each parameter combination
        print(f"Map Size: {map_size}, Num Epochs: {num_epochs}")
        print(f"Silhouette Coefficient: {silhouette_coefficient}")
        print(f"NMI: {nmi}")
        print(f"ARI: {ari}")
        print("")

        # Update best parameters if NMI is higher
        if nmi > best_nmi:
            best_nmi = nmi
            best_map_size = map_size
            best_num_epochs = num_epochs

    print(f"Best Map Size: {best_map_size}")
    print(f"Best Num Epochs: {best_num_epochs}")
    print(f"Best NMI: {best_nmi}")

    generate_bokeh_plots(param_grid['num_epochs'], nmi_values, ari_values, silhouette_values, best_map_size,
                         best_num_epochs)


    return test_docs_vector, cluster_labels, u_matrix

def generate_bokeh_plots(num_epochs, nmi_values, ari_values, silhouette_values, best_map_size, best_num_epochs):
    # Generate Bokeh plot for NMI
    p_nmi = figure(title='NMI', x_axis_label='num_epochs', y_axis_label='NMI')
    p_nmi.line(num_epochs, nmi_values, color="blue", alpha=0.3)

    # Create select menus for map size and num_epochs
    map_size_select = Select(title="Map Size:", value=str(best_map_size),
                             options=[str(x) for x in param_grid['map_size']])
    num_epochs_select = Select(title="Num Epochs:", value=str(best_num_epochs),
                               options=[str(x) for x in param_grid['num_epochs']])

    # Define update_selections function
    def update_selections(attr, old, new):
        map_size = tuple(map(int, map_size_select.value.split(',')))
        num_epochs = int(num_epochs_select.value)

        # Update SOM with selected parameters
        SOM.__init__(input_dim, map_size)
        SOM.train(docs_vector, num_epochs)

        # Update NMI values
        cluster_labels = som.cluster(test_docs_vector)
        cluster_labels = convert_to_base(cluster_labels, map_size)
        nmi = normalized_mutual_info_score(docs_test, cluster_labels)
        nmi_values[param_grid['num_epochs'].index(num_epochs)] = nmi

        # Update NMI plot
        p_nmi.line(num_epochs, nmi_values, color="blue", alpha=0.3)

    # Attach update_selections function to the select menus' on_change event
    map_size_select.on_change('value', update_selections)
    num_epochs_select.on_change('value', update_selections)

    # Create layout
    layout = column(row(map_size_select, num_epochs_select), p_nmi)

    # Show the Bokeh plot
    show(layout)


