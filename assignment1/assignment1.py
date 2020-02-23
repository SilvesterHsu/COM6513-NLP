# %%
import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# fixing random seed for reproducibility
random.seed(123)
np.random.seed(123)

# %%
data_tr = pd.read_csv('data_sentiment/train.csv')
data_tr.head()
text_tr = list(data_tr.to_numpy()[:,0])
label_tr = data_tr.to_numpy()[:,1]

# %%
stop_words = ['a','in','on','at','and','or',
              'to', 'the', 'of', 'an', 'by',
              'as', 'is', 'was', 'were', 'been', 'be',
              'are','for', 'this', 'that', 'these', 'those', 'you', 'i',
             'it', 'he', 'she', 'we', 'they' 'will', 'have', 'has',
              'do', 'did', 'can', 'could', 'who', 'which', 'what',
             'his', 'her', 'they', 'them', 'from', 'with', 'its']

# %%
def extract_ngrams(x_raw, ngram_range=(1,3), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', stop_words=[], vocab=set()):

    if len(vocab)>0:
        return [*vocab]
    def finder(s,n):
        if n == 1:
            return s
        tmp = list()
        for i in range(len(s)):
            if i+n>len(s):
                break
            tmp.append(tuple(s[i:i+n]))
        return tmp

    pattern = re.compile(token_pattern)
    bagWords = list()
    for term in pattern.findall(x_raw):
        if term not in stop_words:
            bagWords.append(term)

    x = list()
    for n in range(ngram_range[0],ngram_range[1]+1):
        x += finder(bagWords,n)

    return x
# %%
extract_ngrams("this is a great movie to watch",
               ngram_range=(1,3),
               stop_words=stop_words)

# %%
extract_ngrams("this is a great movie to watch",
               ngram_range=(1,2),
               stop_words=stop_words,
               vocab=set(['great',  ('great','movie')]))


# %%
import re
pattern = re.compile(r'\b[A-Za-z][A-Za-z]+\b')
x_raw = "this is a great movie to watch"
bagWords = list()
for term in pattern.findall(x_raw):
    if term not in stop_words:
        bagWords.append(term)
x = list()
for n in range(ngram_range[0],ngram_range[1]+1):
    x += finder(bagWords,n)
x


bagWords +finder(bagWords,ngram_range)
ngram_range = (1,3)

finder(bagWords,1)


def finder(s,n):
    if n == 1:
        return s
    tmp = list()
    for i in range(len(s)):
        if i+n>len(s):
            break
        tmp.append(tuple(s[i:i+n]))
    return tmp
