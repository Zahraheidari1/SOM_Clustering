import pandas as pd
from hazm.utils import stopwords_list
from hazm import WordTokenizer, Lemmatizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from SOM import SOM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from minisom import MiniSom

def dunn_index(data, labels):
    centroids = [np.mean(data[labels == i], axis=0) for i in np.unique(labels)]
    centroid_distances = cdist(centroids, centroids, metric='euclidean')
    
    min_intercluster_distance = np.min(centroid_distances[centroid_distances > 0])
    max_intracluster_distance = np.max([np.max(cdist(data[labels == i], data[labels == i], metric='euclidean')) for i in np.unique(labels)])
    
    dunn_index = min_intercluster_distance / max_intracluster_distance
    return dunn_index

def topographic_error(data, som):
    hit_map = np.zeros((som.map_size[0], som.map_size[1]))

    for i, x in enumerate(data):
        bmu_idx = som.find_bmu(x)[1]
        hit_map[bmu_idx] += 1

    topographic_error = 1 - (np.sum(hit_map > 0) / len(data))
    return topographic_error

def quantization_error(data, som):
    quantization_error = np.mean(np.linalg.norm(data - som.weights[som.cluster(data)], axis=1))
    return quantization_error

def get_mapsize(cluster):
    a = 1
    b = cluster
    return (b,a)

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

def run_multiple(mincluster,maxcluster,filename,rep):
    AllClusterLabel = {}
    tests = []
    docs=pre_procces(filename)
    for j in range(rep):
        train,test = train_test_split(docs, test_size=0.2, random_state=42)
        tests.append(test)
        for i in range(mincluster,maxcluster+1):
            # Call the generate_cluster_labels function with the selected file
            cluster_labels,som = clustering(train,test,i)
            scores = scoring(test,cluster_labels,som)
            
            if j != 0:
                AllClusterLabel[i][1].append(cluster_labels)
                AllClusterLabel[i][0]["silhouette"]+=scores["silhouette"]/rep
                AllClusterLabel[i][0]["calinski"]+=scores["calinski"]/rep
                AllClusterLabel[i][0]["dunn"]+=scores["dunn"]/rep
                AllClusterLabel[i][0]["topographic"]+=scores["topographic"]/rep
                AllClusterLabel[i][0]["quantization"]+=scores["quantization"]/rep
            else:
                AllClusterLabel.update({i:(scores,[cluster_labels])})
                AllClusterLabel[i][0]["silhouette"]/=rep
                AllClusterLabel[i][0]["calinski"]/=rep
                AllClusterLabel[i][0]["dunn"]/=rep
                AllClusterLabel[i][0]["topographic"]/=rep
                AllClusterLabel[i][0]["quantization"]/=rep
    
    return tests,AllClusterLabel

def run_multiple_compare(n_clusters,repeat,filename,rep):
    AllRepeatCompare = {}
    docs=pre_procces(filename)
    for i in range(repeat):
        Scores = {
            "SOM":[0,0,0],
            "Kmeans":[0,0,0],
            "Agglomerative":[0,0,0],
            "MiniSom":[0,0,0],
        }
        
        for j in range(rep):
            train,test = train_test_split(docs, test_size=0.2, random_state=42)
            # Call the generate_cluster_labels function with the selected file
            cluster_labels,som = clustering(train,test,n_clusters)
            scores = scoring_another(test,cluster_labels)
            Scores["SOM"][0] += scores["silhouette"]
            Scores["SOM"][1] += scores["calinski"]
            Scores["SOM"][2] += scores["dunn"]

            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train)
            kmeans_labels = kmeans.predict(test)
            scoreskmeans = scoring_another(test,kmeans_labels)
            Scores["Kmeans"][0] += scoreskmeans["silhouette"]
            Scores["Kmeans"][1] += scoreskmeans["calinski"]
            Scores["Kmeans"][2] += scoreskmeans["dunn"]

            agg_cluster = AgglomerativeClustering(n_clusters=n_clusters).fit(train)
            agg_labels = agg_cluster.fit_predict(test)
            scoresagg = scoring_another(test,agg_labels)
            Scores["Agglomerative"][0] += scoresagg["silhouette"]
            Scores["Agglomerative"][1] += scoresagg["calinski"]
            Scores["Agglomerative"][2] += scoresagg["dunn"]

            minisom = MiniSom(1, n_clusters, train.shape[1], sigma=1.0, learning_rate=0.5)
            minisom.train(train, 100)
            minisom_bmu_indices = np.array([minisom.winner(x)[1] for x in test])
            scoresminisom = scoring_another(test,minisom_bmu_indices)
            Scores["MiniSom"][0] += scoresminisom["silhouette"]
            Scores["MiniSom"][1] += scoresminisom["calinski"]
            Scores["MiniSom"][2] += scoresminisom["dunn"]

        AllRepeatCompare.update({i:{"SOM":{ "silhouette":Scores["SOM"][0]/rep,
                                            "calinski":Scores["SOM"][1]/rep,
                                            "dunn":Scores["SOM"][2]/rep},
                                    "Kmeans":{ "silhouette":Scores["Kmeans"][0]/rep,
                                            "calinski":Scores["Kmeans"][1]/rep,
                                            "dunn":Scores["Kmeans"][2]/rep},
                                    "Agglomerative":{ "silhouette":Scores["Agglomerative"][0]/rep,
                                            "calinski":Scores["Agglomerative"][1]/rep,
                                            "dunn":Scores["Agglomerative"][2]/rep},
                                    "MiniSom":{ "silhouette":Scores["MiniSom"][0]/rep,
                                            "calinski":Scores["MiniSom"][1]/rep,
                                            "dunn":Scores["MiniSom"][2]/rep},
                                    }})
        
    return AllRepeatCompare

def pre_procces(filename):
    # Read the Excel file and extract the documents
    df = pd.read_excel(filename, sheet_name='Sheet')
    docs = df['متن'].to_list()

    # Preprocess the training and test documents
    docs = preprocess_documents(docs)

    # Convert documents to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(docs).toarray()

    # Perform PCA on the TF-IDF vectors
    pca = PCA(n_components=2)
    docs_vector = pca.fit_transform(tfidf_vectors)
    return docs_vector

def clustering(train,test,cluster=10,epochs=50):
    map_size = get_mapsize(cluster)
    # Create SOM instance
    input_dim = train.shape[1]

    som = SOM(input_dim, map_size)  
    som.train(train, 50)

    cluster_labels = som.cluster(test)
    cluster_labels = convert_to_base(cluster_labels, map_size)
    return cluster_labels,som

def scoring(test,label,som):
    # Calculate the evaluation metrics
    silhouette_coefficient = silhouette_score(test, label)
    calinski_harabasz = calinski_harabasz_score(test, label)
    dunn = dunn_index(test, label)
    topographic_err = topographic_error(test, som)
    quantization_err = quantization_error(test, som)
    return {"silhouette":silhouette_coefficient,"calinski":calinski_harabasz,"dunn":dunn,
            "topographic":topographic_err,"quantization":quantization_err}

def scoring_another(test,label):
    silhouette_coefficient = silhouette_score(test, label)
    calinski_harabasz = calinski_harabasz_score(test, label)
    dunn = dunn_index(test, label)
    return {"silhouette":silhouette_coefficient,"calinski":calinski_harabasz,"dunn":dunn}
