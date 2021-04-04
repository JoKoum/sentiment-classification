import numpy as np
import pandas as pd
from collections import Counter

import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import spacy

import requests
from bs4 import BeautifulSoup

def clean_text(text):
    """Accepts a string a input and applies a cleaning pipeline,
       converting to lowercase, removing special characters, punctuation
       and stopwords and performing lemmatization.
       The output is the cleaned input string.
    """
    punctuation = set(string.punctuation)
    greek = re.compile(r'[α-ωά-ώϊϋΐΰ]+') #Isolate greek characters
    special_chars = re.compile('[u"\U0001F300-\U0001F5FF"]+',flags=re.UNICODE) #Locate special characters like emojis
    nlp = spacy.load('el_core_news_sm')
    # Convert the text into lowercase
    text = text.lower()
    # Remove special characters
    text = special_chars.sub(r'',text)
    # Remove Latin characters
    wordList = greek.findall(text)
    # Remove punctuation
    wordList = ["".join(x for x in word if (x=="'")|(x not in punctuation)) for word in wordList]
    # Remove stopwords
    wordList = [word for word in wordList if word not in stopwords.words('greek')]
    # Lemmatisation
    cleaned_text = nlp(" ".join(wordList))
    cleaned_text = [token.lemma_ for token in cleaned_text]
    return " ".join(cleaned_text).strip()

def read_and_preprocess_data(min_count = 10, polarity_cutoff = 0.1):
    """Read the artificially translated cleaned dataset.
       Create the vocabulary from the given reviews, by
       reducing noise, leaving out rare words that may get 
       attributed with much sentiment. Then create a dictionary 
       of words in the vocabulary mapped to index positions
       Outputs: 
       greek_reviews (Pandas Dataframe)
       labels (Pandas Dataframe)
       vocab (list)
       word2index (dictionary)
    """
    reviews_dataset = pd.read_csv('https://drive.google.com/uc?id=1I9_zVYFblWO4uYP3-WsIkV-RPdsUMx2o&export=download',
                              index_col=0)

    nans = np.where(pd.isnull(reviews_dataset.cleaned_text))
    reviews_dataset.drop(list(nans)[0],inplace=True)
    
    sentiment_dict = {1:0, 2:0, 3:0, 4:1, 5:1}
    reviews_dataset['ratings'].replace(sentiment_dict,inplace=True)
    
    reviews = pd.DataFrame({'reviews':reviews_dataset.cleaned_text.values, 'ratings':reviews_dataset.ratings.values})

    greek_reviews = reviews.reviews
    labels = reviews.ratings

    positive_counts = Counter()
    negative_counts = Counter()
    total_counts = Counter()
    
    for i in range(len(reviews)):
        try:
            for word in greek_reviews[i].split(" "):
                total_counts[word] += 1
                if labels[i] == 1:
                    positive_counts[word] += 1 
                else:
                    negative_counts[word] += 1
        except KeyError:
            pass

    pos_neg_ratios = Counter()

    for term, cnt in list(total_counts.most_common()):
        if (cnt >= 100):
            pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
            pos_neg_ratios[term] = pos_neg_ratio
    
    for word, ratio in pos_neg_ratios.most_common():
        if (ratio > 1):
            pos_neg_ratios[word] = np.log(ratio)
        else:
            pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))


    vocab = set()
    for review in greek_reviews:
        for word in review.split(" "):
            if (total_counts[word] > min_count):
                if (word in pos_neg_ratios.keys()):
                    if ((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                        vocab.add(word)
                else:
                    vocab.add(word)

    vocab = list(vocab)
    
    # Create a dictionary of words in the vocabulary mapped to index positions
    # (to be used in layer_0)
    word2index = {}
    
    for i,word in enumerate(vocab):
        word2index[word] = i
    
    return greek_reviews, labels, vocab, word2index



def data_extraction(url,reviews_selector,ratings_selector):
    """Accepts a URL and the respective CSS selectors
       of the reviews text and rating.
       Returns the extracted data in a Pandas dataframe.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text,'html.parser')
    reviews = soup.select(reviews_selector)
    ratings = soup.select(ratings_selector)
    rlist = []
    for i in range(len(reviews)):
        rlist.append([reviews[i].text,ratings[i].text])
    data = pd.DataFrame(rlist,columns=['reviews','sentiment'])
    sentiment_dict = {'1':0, '2':0, '3':0, '4':1, '5':1}
    data['sentiment'].replace(sentiment_dict,inplace=True)
    data['reviews'] = data['reviews'].apply(clean_text)
    return data