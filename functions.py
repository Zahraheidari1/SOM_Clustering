import pandas as pd
from hazm.utils import stopwords_list
from hazm import WordTokenizer, Lemmatizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from SOM import SOM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def get_mapsize(cluster):
    a = 1
    b = cluster
    for i in range(1,cluster+1):
        for j in range(cluster,0,-1):
            if i*j==cluster and abs(i-j)<=abs(a-b):
                a=i
                b=j
    return (a,b)

def convert_to_base(number_list, map_size):
    converted_numbers = []
    for number in number_list:
        converted_numbers.append((number[0] * map_size[1] + number[1])+1)

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

def run_multiple(mincluster,maxcluster,filename,sourceSilhouette,sourceCalinski,sourceDavies):
    AllClusterLabel = {}
    train,test=pre_procces(filename)
    for i in range(mincluster,maxcluster+1):
        # Call the generate_cluster_labels function with the selected file
        cluster_labels = clustering(train,test,i)
        scores = scoring(test,cluster_labels)

        sourceSilhouette.stream(
            {"Cluster":[i] ,
            "Score":[scores['silhouette']]})
        sourceCalinski.stream(
            {"Cluster":[i] ,
            "Score":[scores['calinski']]})
        sourceDavies.stream(
            {"Cluster":[i] ,
            "Score":[scores['davies']]})

        AllClusterLabel.update({i:cluster_labels})

        print(f"Function executed successfully for cluster : {i}\t with scores:\n\tSilhouette: {scores['silhouette']}\tCalinski: {scores['calinski']}\tDavies: {scores['davies']}")
    return test,AllClusterLabel

def pre_procces(filename):
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
    # Preprocess the test documents
    docs_test = preprocess_documents(docs_test)

    # Convert test documents to TF-IDF vectors
    test_tfidf_vectors = vectorizer.transform(docs_test).toarray()
    test_docs_vector = pca.transform(test_tfidf_vectors)
    return docs_vector,test_docs_vector

def clustering(train,test,cluster=10,epochs=50):
    map_size = get_mapsize(cluster)
    # Create SOM instance
    input_dim = train.shape[1]

    som = SOM(input_dim, map_size)  
    som.train(train, 50)

    cluster_labels = som.cluster(test)
    cluster_labels = convert_to_base(cluster_labels, map_size)
    return cluster_labels

def scoring(test,label):
    # Calculate the evaluation metrics
    silhouette_coefficient = silhouette_score(test, label)
    calinski_harabasz = calinski_harabasz_score(test, label)
    davies_bouldin = davies_bouldin_score(test, label)

    return {"silhouette":silhouette_coefficient,"calinski":calinski_harabasz,"davies":davies_bouldin}

def generate_cluster_labels(filename,cluster=10,epochs=50):
    map_size = get_mapsize(cluster)
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

    som = SOM(input_dim, map_size)  
    som.train(docs_vector, 50)

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
    calinski_harabasz = calinski_harabasz_score(test_docs_vector, cluster_labels)
    davies_bouldin = davies_bouldin_score(test_docs_vector, cluster_labels)

    return test_docs_vector , cluster_labels , {"silhouette":silhouette_coefficient,"calinski":calinski_harabasz,"davies":davies_bouldin}
