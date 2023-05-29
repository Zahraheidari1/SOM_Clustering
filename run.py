import numpy as np
from SOM import SOM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from functions import generate_cluster_labels
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


output_file = generate_cluster_labels('Data.xlsx')
print("Output file:", output_file)
