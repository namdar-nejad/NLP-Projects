import pandas as pd
import re

import numpy as np
import nltk
import sklearn

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import wordnet

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

posFilePath = "./data/rt-polarity.pos"
negFilePath = "./data/rt-polarity.neg"

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


# Data Prep

# import data
pos_input = np.loadtxt(posFilePath, dtype=str, delimiter='\n', encoding="latin-1")
neg_input = np.loadtxt(negFilePath, dtype=str, delimiter='\n', encoding="latin-1")

# create dataframes
pos_df = pd.DataFrame({'Review' : pos_input, 'Sentiment' : 'positive'})
neg_df = pd.DataFrame({'Review' : neg_input, 'Sentiment' : 'negative'})
all_df = pos_df.append(neg_df).reset_index()
del all_df['index']
all_df = all_df.sample(frac = 1)

# encoding Sentiment column
le = LabelEncoder()
all_df["Sentiment"] = le.fit_transform(all_df["Sentiment"])


# Pre-processing Functions

# POS tagging
def get_pos(word):
    if word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('N'):
        return wordnet.NOUN
    elif word.startswith('R'):
        return wordnet.ADV
    else:
        return None

# lemmatization with POS
def lemmatize(words):
    
    pos_list = nltk.pos_tag(nltk.word_tokenize(words))
    
    rtn = []
    for word in pos_list:
        if get_pos(word[1]) is None:
            rtn.append(word[0])
        else:
            rtn.append(WordNetLemmatizer().lemmatize(word[0], get_pos(word[1])))
            
    rtn = " ".join(rtn)
        
    return rtn

# stemming
def stem(words):
    
    rtn = []
    split = words.split()
    
    for word in split:
        rtn.append(PorterStemmer().stem(word))
        
    rtn = " ".join(rtn)
    return rtn

# remove stop words
def rm_stop(words):
    
    split = words.split()
    rtn = []
    
    for word in split:
        if word not in stopword_list:
            rtn.append(word)
            
    rtn = " ".join(rtn)
        
    return rtn

# main preprocessing function
def preprocess_corpus(corpus, clean=False, stop=False, lemmatization=False,
                     stemming=False, special=False):
    
    normalized_corpus = []
    
    for line in corpus:
            
        # clean text
        if clean:
            
            # remove HTML tags
            line = re.sub('<[^>]*>','',line)
            
            # lowercase the text    
            line = line.lower()
            
            # remove extra newlines
            line = re.sub(r'[\r|\n|\r\n]+', ' ', line)
            
        #remove special chars
        if special:         
            line = re.sub('[^a-zA-z0-9\s]', '', line)
            
        # remove stop words
        if stop:
            line = rm_stop(line)
        
        # lemmatize text
        if lemmatization:
            line = lemmatize(line)
        
        # stem text
        if stemming:
            line = stem(line)
            
        # clean text
        if clean:
            # remove extra whitespace
            line = re.sub(' +', ' ', line)
        
        normalized_corpus.append(line)
        
    return normalized_corpus

# function to slip data and create testing and traning set
def split_data(train_percent):

    testSpltIndx=int(((len(pos_df) + len(neg_df))*(train_percent/100)))
    
    reviews = np.array(all_df['Review'])
    sentiments = np.array(all_df['Sentiment'])

    train_reviews = reviews[:testSpltIndx]
    train_sentiments = sentiments[:testSpltIndx]

    test_reviews = reviews[testSpltIndx:]
    test_sentiments = sentiments[testSpltIndx:]
    
    return train_reviews, train_sentiments, test_reviews, test_sentiments


# Features Engineering
def build_feature(processed_train_reviews, processed_test_reviews):

    # BOW approach
    vec = CountVectorizer()
    # build train features
    train_features = vec.fit_transform(processed_train_reviews)

    # build test features
    test_features = vec.transform(processed_test_reviews)
    
    return train_features, test_features


# Build Model
def build_model(train_features, train_sentiments, test_features):

    # Logistic Regression
    lr = LogisticRegression(max_iter=200)
    # Logistic Regression model on BOW features
    lr.fit(train_features,train_sentiments)

    # predict using model
    predictions = lr.predict(test_features)
    
    return predictions


# Train Model
def train(processed_train_reviews, processed_test_reviews):
    
    # build features
    train_features, test_features = build_feature(processed_train_reviews, processed_test_reviews)
    
    # build_model
    predictions = build_model(train_features, train_sentiments, test_features)
    
    return test_sentiments, predictions


# Funtion to print results 
def print_res(predictions, test_sentiments):
    print('Accuracy:  {:2.5%} '.format(metrics.accuracy_score(test_sentiments, predictions)))
    print('Precision: {:2.5%} '.format(metrics.precision_score(test_sentiments, predictions, average='weighted')))
    print('Recall:    {:2.5%} '.format(metrics.recall_score(test_sentiments, predictions, average='weighted')))
    print('F1 Score:  {:2.5%} '.format(metrics.f1_score(test_sentiments, predictions, average='weighted')))
    
    return (metrics.accuracy_score(test_sentiments, predictions)), (metrics.precision_score(test_sentiments, predictions, average='weighted')), (metrics.recall_score(test_sentiments, predictions, average='weighted'))


## HERE IS WHERE WE ARE RUNNING SOME EXAMPLES ##

# this arry will contain the percentage of the traning set we are going to have
# to test on other traning splits add the traning set percentage to the array
traning_set_splits = [90]

# this array contains the preprocessing tags we are going to use
# in order the tasgs refer to [clean, stop, lemmatization, stemming, special]
preprcessing_configs = [
	    [[False, False, False, False, False]],
	        
	    [[True, False, False, False, False],
	    [False, True, False, False, False],
	    [False, False, True, False, False], 
	    [False, False, False, True, False],
	    [False, False, False, False, True]],
	              
	    [[True, False, True, False, True],
	    [True, False, False, True, True],
	    [False, True, False, True, True]]
            ]

# Run and evaluate
results = []

for i in traning_set_splits:
    for m in preprcessing_configs:
        
        config = []
        percent = []
        accuracy = []

        for j in m:

            # slip data and create testing and traning set
            train_reviews, train_sentiments, test_reviews, test_sentiments = split_data(i)

            # preprocess datasets
            processed_train_reviews = preprocess_corpus(train_reviews, j[0], j[1], j[2], j[3], j[4])
            processed_test_reviews = preprocess_corpus(test_reviews, j[0], j[1], j[2], j[3], j[4])
            
            test_sentiments, predictions = train(processed_train_reviews, processed_test_reviews)
            
            # print results and data split info
            print(" ")
            print(str(i) + "% Train " + str(100-i) +"% Test" )
            print("clean=%s, stop=%s, lemmatization=%s, stemming=%s, special=%s" % (j[0], j[1], j[2], j[3], j[4]))

            acc, prec, rec = print_res(predictions, test_sentiments)

            # store results for later on
            config.append("clean=%s, stop=%s, lemmatization=%s, stemming=%s, special=%s" % (j[0], j[1], j[2], j[3], j[4]))
            accuracy.append(round(acc*100,2))
            percent.append((i,100-i))
        
            output = {'Config':config, 'Percent':percent, 'Accuracy':accuracy}
            output_df = pd.DataFrame(output)
            
        results.append(output_df)


# print all the results orderd by 'Accuracy'
for i in results:
    pd.set_option('max_colwidth', 500)

    output_df.style.set_properties(subset=['Config'], **{'width': '600px'})

    display(i.sort_values('Accuracy'))



