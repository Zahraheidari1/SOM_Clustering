import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hazm
import sys
from openpyxl import load_workbook
import xlwings as xw
from  text_similarity import text_similarity

df = pd.read_excel('Data.xlsx', sheet_name='Sheet') 
docs=df['متن'].to_list()
text2=input()
max_score = 0
max_text = ""

for doc in docs:
    similarity_score=text_similarity(doc, text2)
    if similarity_score > max_score:
        max_score = similarity_score
        max_text = doc

print(max_text)
print(max_score)      
'''normalizer = hazm.Normalizer()
tokenizer = hazm.WordTokenizer()
stemmer = hazm.Stemmer()
docs = [normalizer.character_refinement(doc) for doc in docs if doc is not docs]
docs = [' '.join(tokenizer.tokenize(doc)) for doc in docs if doc is not docs]
docs=[stemmer.stem(doc) for doc in docs]
vectorizer = CountVectorizer(stop_words=hazm.stopwords_list())
text=['انهدام كامل مجتمع توليد سلاحهاي ميكروبي ']
print(docs)
doc_vectors = vectorizer.fit_transform(docs)


similarity_matrix = cosine_similarity(doc_vectors)


doc_index = 0
similar_docs = pd.DataFrame(similarity_matrix[doc_index], columns=['similarity'])
similar_docs['document'] = docs
similar_docs = similar_docs.sort_values(by='similarity', ascending=False)
print(similar_docs)'''
