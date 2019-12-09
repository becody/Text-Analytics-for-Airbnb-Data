# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:37:06 2019

@author: 17046
"""

#import modules
import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import tqdm
from sklearn.model_selection import train_test_split
import re
import string
import nltk
nltk.download('punkt')

import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.summarization import summarize, keywords
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models import LsiModel

#############################################################################
# Import Data 
#import AirBnB data 
reviews = pd.read_csv (r'C:\Users\17046\OneDrive\Documents\MSA 20\Text Analytics\reviews.csv')
reviews.comments = reviews.comments.astype(str)
reviews.head()

#############################################################################
# Begin cleaning data and tokenizing documents
#Get rid of missing values
reviews.dropna(inplace=True)
reviews.head()

# Remove punctuation, then tokenize documents
punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
term_vec = [ ]

for d in reviews['comments']:
    d = d.lower()
    d = punc.sub( '', d )
    term_vec.append( nltk.word_tokenize( d ) )

# Print resulting term vectors
#for vec in term_vec:
 #   print(vec)

# Remove stop words from term vectors
stop_words = nltk.corpus.stopwords.words( 'english' )

for i in range( 0, len( term_vec ) ):
    term_list = [ ]

    for term in term_vec[ i ]:
        if term not in stop_words:
            term_list.append( term )

    term_vec[ i ] = term_list

# Print term vectors with stop words removed and porter stems
for vec in term_vec:
    print(vec)

porter = nltk.stem.porter.PorterStemmer()

for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
        term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )

# Print term vectors with everything above removed
#for vec in term_vec:
 #   print(vec)
#############################################################################
#############################################################################
 
#############################################################################
# Document Similarity Attempt 1
#############################################################################
 
#  Convert term vectors into gensim dictionary
dict = gensim.corpora.Dictionary( term_vec )

corp = [ ]
for i in range( 0, len( term_vec ) ):
    corp.append( dict.doc2bow( term_vec[ i ] ) )

#  Create TFIDF vectors based on term vectors bag-of-word corpora
tfidf_model = gensim.models.TfidfModel( corp )

tfidf = [ ]
for i in range( 0, len( corp ) ):
    tfidf.append( tfidf_model[ corp[ i ] ] )

#  Create pairwise document similarity index
    #Currently does not work well 
n = len( dict )
index = gensim.similarities.SparseMatrixSimilarity( tfidf_model[ corp ], num_features = n )

#  Print TFIDF vectors and pairwise similarity per document
for i in range( 0, len( tfidf ) ):
    s = 'Doc ' + str( i + 1 ) + ' TFIDF:'

    for j in range( 0, len( tfidf[ i ] ) ):
        s = s + ' (' + dict.get( tfidf[ i ][ j ][ 0 ] ) + ','
        s = s + ( '%.3f' % tfidf[ i ][ j ][ 1 ] ) + ')'

    print(s)

for i in range( 0, len( corp ) ):
    print ('Doc', ( i + 1 ), 'sim: [ ',)

    sim = index[ tfidf_model[ corp[ i ] ] ]
    
    for j in range(0, len( sim ) ):
        print('%.3f ' % sim[ j ], ']')

#####################################################
# Latent Semantic Analysis on the dictionary above
#####################################################
    
# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
def prepare_corpus(term_vec):
    dict = gensim.corpora.Dictionary( term_vec )
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.

    doc_term_matrix = [dict.doc2bow(doc) for doc in term_vec]

# generate LDA model
    return dict, doc_term_matrix

#Create LSA model using Gensim
def create_gensim_lsa_model(term_vec,number_of_topics,words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dict,doc_term_matrix=prepare_corpus(term_vec)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dict)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel

#determine number of optimum topics using coherence sores
def compute_coherence_values(dict, doc_term_matrix, term_vec, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for number_of_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dict)  # train model 
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=term_vec, dictionary=dict, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

#plotting coherence score values
def plot_graph(doc_clean,start, stop, step):
    dict,doc_term_matrix=prepare_corpus(term_vec)
    model_list, coherence_values = compute_coherence_values(dict, doc_term_matrix,term_vec,
                                                            stop, start, step)
    # Show graph
    stop=40; start=2; step=6;
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

start,stop,step=2,12,1
plot_graph(term_vec, start,stop,step)
import pyLDAvis.gensim

#running all of the above functions 
number_of_topics=7
words=10
model=create_gensim_lsa_model(term_vec,number_of_topics,words)

prepare_corpus(term_vec)

#############################################################################
# Latent Sentiment Analysis Attemot 2 (Topic Modeling)
#############################################################################
 
#Creating duplicate dictionary as dict above 
dictionary = corpora.Dictionary(term_vec)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in term_vec]

#Creating trained model to evaluate coherence scores
Lda = models.LdaMulticore
coherenceList_umass = []
coherenceList_cv = []
num_topics_list = np.arange(3,26)
for num_topics in (num_topics_list):
    lda= Lda(doc_term_matrix, num_topics=num_topics,id2word = dictionary, 
             passes=20,chunksize=4000,random_state=43)
    cm = CoherenceModel(model=lda, corpus=doc_term_matrix, 
                        dictionary=dictionary, coherence='u_mass')
    coherenceList_umass.append(cm.get_coherence())
    cm_cv = CoherenceModel(model=lda, corpus=doc_term_matrix,
                           texts=term_vec, dictionary=dictionary, coherence='c_v')
    coherenceList_cv.append(cm_cv.get_coherence())
    vis = pyLDAvis.gensim.prepare(lda, doc_term_matrix, dictionary)
    pyLDAvis.save_html(vis,f'pyLDAvis_{num_topics}.html')

#Plotting coherence scores to find optimum number of topics contained in our docs
plotData = pd.DataFrame({'Number of topics': num_topics_list,
                        'CoherenceScore':coherenceList_umass})
f, ax = plt.subplots(figsize=(10,6))
sns.set.style("darkgrid")
sns.pointplot(x='Number of topics', y=' CoherenceScore', data=plotData)
plt.axhline(y= -3.9)
plt.title('Topic coherence')
plt.savefig('Topic coherence plot.png')























