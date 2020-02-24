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
data_tr = pd.read_csv('data_sentiment/train.csv',header=None)
data_tr.head()
text_tr = X_tr_raw = list(data_tr.to_numpy()[:,0])
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
            all tokens. Note that data is already tokenised so you could opt for
            a simple white space tokenisation.
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
            return tmp'''
        if n <= 0 or n > len(s):
            raise Exception('n is out of range')
        if len(features) == 0:
            if n == 1:
                return s
            else:
                return map(lambda i: tuple(s[i:i+n]),range(len(s)-n+1))
        else:
            if n == 1:
                return [t for t in s if t in features]
            else:
                return [tuple(s[i:i+n]) for i in range(0,len(s)-n+1) if tuple(s[i:i+n]) in features]

    # Find all words by condition
    pattern = re.compile(token_pattern)
    term_eligible = [term for term in pattern.findall(x_raw) if term not in stop_words]

    # Find combinations of different N-grams
    x = [term for n in range(ngram_range[0],ngram_range[1]+1) for term in finder(term_eligible,n,vocab)]
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
def get_vocab(X_raw, ngram_range=(1,3), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', min_df=0, keep_topN=0, stop_words=[]):
    '''1. create a vocabulary of ngrams
       2. count the document frequencies of ngrams
       3. their raw frequency

    Args:
        X_raw: a list of strings each corresponding to the raw text of a document
        ngram_range: a tuple of two integers denoting the type of ngrams you want
            to extract, e.g. (1,2) denotes extracting unigrams and bigrams.
        token_pattern: a string to be used within a regular expression to extract
            all tokens. Note that data is already tokenised so you could opt for
            a simple white space tokenisation.
        stop_words: a list of stop words
        min_df: keep ngrams with a minimum document frequency.
        keep_topN: keep top-N more frequent ngrams.

    Returns:
        vocab: a set of the n-grams that will be used as features.
        df: a Counter (or dict) that contains ngrams as keys and their corresponding
            document frequency as values.
        ngram_counts: counts of each ngram in vocab
        For example,


    '''

    tf = ngram_counts = list()
    df = list()
    for line in X_tr_raw:
        features = extract_ngrams(line,ngram_range=(1,3),stop_words=stop_words)
        tf += features
        df += list(set(features))
    tf = ngram_counts = Counter(tf)
    df = Counter(df)
    vocab = [items[0] for items in tf.most_common()[:5000]]

    return vocab, df, ngram_counts

#%%
vocab, df, ngram_counts = get_vocab(X_tr_raw, ngram_range=(1,3), keep_topN=5000, stop_words=stop_words)
print(len(vocab))
print()
print(list(vocab)[:100])
print()
print(df.most_common()[:10])

#%%
reference_dict = dict(enumerate(vocab))
reference_dict

#%%
X_ngram = [Counter(extract_ngrams(line,ngram_range=(1,3),stop_words=stop_words)) for line in X_tr_raw]
X_ngram[:5]

#%%
def vectorise(X_ngram, vocab):
    '''1. select the features of vocab from X_ngram.
       2. convert X_ngram into matrix

    Args:
        X_ngram: a list of texts (documents) features(Bag-of-ngram)
        vocab: a set of selected features(n-grams)

    Returns:
        X_vec: an array shapes #document x #vocab, where document is a single line
            in dataset.
    '''

    X_vec = np.zeros([len(X_ngram),len(vocab)])
    for docs_index in range(len(X_ngram)):
        for feature_index in range(len(vocab)):
            X_vec[docs_index,feature_index] = X_ngram[docs_index].get(reference_dict[feature_index],0)
    return X_vec

#%% Count vectors
X_tr_count = vectorise(X_ngram, vocab)
print("The shape of X_vec is {}".format(X_tr_count.shape))
X_tr_count[:2,:50]

#%%
#idf = {term: np.log(len(X_ngram)/frequency) for term,frequency in df.items()}
#idf
#%% compute idfs
idfs = np.zeros((1,len(vocab)))
for i in range(len(vocab)):
    idfs[0,i] = np.log(len(X_ngram)/df[reference_dict[i]])

idfs

#%% transform count vectors to tf.idf vectors
X_tr_tfidf = X_tr_count*idfs

#%%
X_tr_tfidf[1,:50]














#%%
t = ['manages', 'questions', 'covered', 'body', 'ron', 'flair', 'drunken', 'approach', 'etc', 'allowing', 'lebowski', 'strong', 'model', 'category', 'family', 'couldn', 'argento', 'why', 'shown', ('doesn', 'work'), 'ocean', ('lot', 'more'), 'lou', 'attorney', 'kick', 'thinking', 'worth', 'larger', ('waste', 'time'), ('back', 'forth'), 'roles', 'adventures', ('million', 'dollars'), 'critics', 'according', ('ghost', 'dog'), 'outside', 'protect', ('last', 'time'), ('but', 'so'), 'creative', 'sell', 'pile', 'needless', 'immediately', 'screens', 'cards', 'blonde', 'meets', 'place', 'needs', 'needed', 'teacher', 'conceived', 'competition', 'powerful', 'expected', ('first', 'movie'), ('but', 'least'), 'gave', 'pleasures', 'spectacular', 'safe', 'wishes', 'stuff', ('there', 'something'), 'robert', 'kid', 'latest', ('bad', 'guy'), 'comet', 'street', 'intelligent', 'allow', ('tim', 'roth'), ('production', 'design'), 'living', 'abyss', 'clean', ('makes', 'him'), 'aware', 'footage', 'vicious', 'sharon', 'genuinely', 'south', 'draw', 'wall', ('will', 'smith'), 'romeo', ('scenes', 'but'), 'sometimes', 'friend', 'millionaire', 'families', 'technique', 'spirit', ('not', 'going'), 'horrifying', 'national']
