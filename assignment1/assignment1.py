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
data_tr = pd.read_csv('assignment1/data_sentiment/train.csv')
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
    """N-gram extraction from a document.

    Args:
        x_raw: a string corresponding to the raw text of a document
        ngram_range: a tuple of two integers denoting the type of ngrams you want
                to extract, e.g. (1,2) denotes extracting unigrams and bigrams.
        token_pattern: a string to be used within a regular expression to extract
                all tokens. Note that data is already tokenised so you could opt
                for a simple white space tokenisation.
        stop_words: a list of stop words
        vocab: a given vocabulary. It should be used to extract specific features.

    Returns:
        A list of terms to the corresponding N-gram. Each part fits one N-gram,
        except 1-gramrow. For example:

        ['great','movie','watch',
        ('great', 'movie'),('movie', 'watch'),
        ('great', 'movie', 'watch')]

    Raises:
        x_raw is empty?
    """

    def finder(s,n,features = set()):
        '''Find different combinations of n-gram from the list.

        Args:
            s: a term list
            n: refers to the n in N-gram
            features: only select combinations which has `feature term`

        Returns:
            A list of terms to the corresponding N-gram. For example:
            for n=2,
            [('great', 'movie'),('movie', 'watch')]
        '''
        if n <= 0:
            raise Exception('n = {}, but n can not be smaller than 1 in N-gram'.format(n))
        if len(features) == 0:
            if n == 1:
                return s
            tmp = list()
            for i in range(len(s)):
                if i+n>len(s):
                    break
                tmp.append(tuple(s[i:i+n]))
            return tmp
        else:
            tmp = list()
            for i in range(len(s)):
                if i+n>len(s):
                    break
                present = tuple(s[i:i+n]) if n != 1 else s[i]
                if present in features:
                    tmp.append(present)
            return tmp

    # Find all words by condition
    pattern = re.compile(token_pattern)
    term_eligible = list()
    for term in pattern.findall(x_raw):
        if term not in stop_words:
            term_eligible.append(term)

    # Find combinations of different N-grams
    x = list()
    for n in range(ngram_range[0],ngram_range[1]+1):
        x += finder(term_eligible,n,vocab)

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
