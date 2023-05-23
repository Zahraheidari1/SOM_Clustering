from sklearn.metrics.pairwise import cosine_similarity
from hazm import *
from hazm.utils import stopwords_list
from sklearn.feature_extraction.text import TfidfVectorizer
from re import UNICODE, sub,compile
import string
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from hazm import SnowballStemmer
import numpy as np
def cleaner_persian(doc):
    doc=doc.lower()
    p=compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    doc=sub(p,'',doc)
    p=compile("["
    u"\U0001F600-\U0001FFFF"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U00002780"
    u"\U000024C2-\U0001F251"
    "]+",flags=UNICODE)
    doc=sub(p,'',doc)
    p = compile('[۰-۹]+')
    doc = p.sub('', doc)
    doc=sub(r'<lb>',' ',doc)
    doc=sub(r"[,.\"!'`@#$%^(){}\[\]/;~<>+=-_]",'',doc)
    tokenz=word_tokenize(doc)
    table=str.maketrans('','',string.punctuation)
    stripped=[w.translate(table) for w in tokenz]
    words=[w for w in stripped if w.isalpha()]
    stop=set(stopwords.words('persian'))
    words=[w for w in words if not w in stop]
    s=SnowballStemmer(language='persian')
    words=[s.stem(w) for w in words]
    return tuple(words)

def create_vocab(docs):
    vocab=set()
    for i in docs:
        for j in i:
            vocab.add(j)
    return tuple(vocab)

def normilaze_dict(d:dict):
    maximum=1e-7
    for i in d.values():
        if i>maximum:
            maximum=i
    for i in d.keys():
        d[i]=log(1+d[i])/log(1+maximum)

def vectorizer(docs):
    tokens=list()
    for i in docs:
        tokens.append(cleaner_persian(i))
    vocab=create_vocab(tokens)
    print(vocab)
    results=list()
    norm_tf=list()
    dictionary={i:0 for i in vocab}
    for i in tokens:
        temp=dictionary.copy()
        for j in i:
            temp[j]+=1
        results.append(temp)
    for i in results:
        normilaze_dict(i)
    for i in results:
        temp=list()
        for j in vocab:
            temp.append(i[j])
        norm_tf.append(temp)
    norm_tf=np.array(norm_tf)
    norm_tf=norm_tf.reshape((len(norm_tf),1,len(norm_tf[0])))
    return norm_tf
