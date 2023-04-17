import pandas as pd
import re
import csv
import gensim
from gensim import corpora
import pickle

#read file
df = pd.read_csv('interaction_subset.csv', usecols=['recipe_id','review'],quoting=csv.QUOTE_NONE)
df = df.dropna(how='all')
df['puncts_removed_review'] = df['review'].map(lambda x: re.sub(r'[^\w\s]', '', str(x))) #remove punctuations 
df['lowercase_review'] = df['puncts_removed_review'].map(lambda x: x.lower()) #convert to lowercase

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
  
#split reviews into lists of words 
reviews = df.lowercase_review.values.tolist()
ids = df.recipe_id.values.tolist()
for i in range(len(reviews)):
  reviews[i] = word_tokenize(reviews[i])
  
#perform lemmatization
#reference https://www.holisticseo.digital/python-seo/nltk/lemmatize
w_n_lemmatizer = WordNetLemmatizer()

def nltk_pos(tag):
  if tag.startswith('J'):
    return wordnet.ADJ
  elif tag.startswith('V'):
    return wordnet.VERB
  elif tag.startswith('N'):
    return wordnet.NOUN
  elif tag.startswith('R'):
    return wordnet.ADV
  else:
    return None

def lemmatize_review(rev):
  tagged_rev = nltk.pos_tag(rev)
  wordnet_tagged = map(lambda x: (x[0], nltk_pos(x[1])), tagged_rev)
  lemmatized_review = []
  for w, tag in wordnet_tagged:
    if tag is None:
      lemmatized_review.append(w)
    elif tag is wordnet.VERB:
      lemmatized_review = lemmatized_review
    else:        
      lemmatized_review.append(w_n_lemmatizer.lemmatize(w, tag))
  return lemmatized_review

for i in range(len(reviews)):
  reviews[i] = lemmatize_review(reviews[i])

#remove stop words
stop_words = stopwords.words('english')
irrelevant_lst = ['recipe', 'ingredient', 'would', 'also', 'make', 'cook', 'really', 'like', 'good', 'well', 'great', 'thanks', 'wonderful']
stop_words.extend(irrelevant_lst)
for lst in reviews:
  for w in lst:
    lst[:] = [w for w in lst if w not in stop_words]
  
#len(dict) = num of unique tokens
dict = corpora.Dictionary(reviews)
#filter out words that occur in fewer than 50 reviews, or more than 70% of all reviews
dict.filter_extremes(no_below=100, no_above=0.7)
#bag-of-words representation of reviews, len(corpus) = num of documents
corpus = [dict.doc2bow(rev) for rev in reviews]
#entry corresponds to tuple -- (word, # of occurences)

LDA = gensim.models.ldamodel.LdaModel

lda_model = LDA(corpus=corpus, id2word=dict, num_topics=2, random_state=50, chunksize=100, alpha='auto', eta='auto',passes=50,iterations=100) 
topics = lda_model.print_topics(num_words=5)

#use LDA topics to assign reviews to corresponding topics
#get_term_topics returns the probability that a word belongs to a particular topic
#similarly, we also have get_document_topics
def get_label(review):
  if type(review) == str:
    bow_format = dict.doc2bow(review.split())
  else:
    bow_format = review
  topics_prob = lda_model.get_document_topics(bow_format)
  if max(topics_prob[0][1], topics_prob[1][1]) == topics_prob[0][1]:
    label = "simple and tasty"
  else:
    label = "recipes you'll use over and over again"
  return label

#dictionary of the format {id:(# of 1st-topic reviews, # of 2nd-topic reviews)}
review_labels = {}
for i in range(len(corpus)):
  label = get_label(corpus[i])
  review_labels[ids[i]] = [0,0]
  if label == "simple and tasty":
    review_labels[ids[i]][0] += 1 
  else: 
    review_labels[ids[i]][1] += 1

for k in review_labels:
  if max(review_labels[k][0], review_labels[k][1]) == review_labels[k][0]:
    review_labels[k] == "simple and tasty"
  else:
    review_labels[k] == "recipes you'll use over and over again"

with open('pickled_dict.pickle', 'wb') as handle:
    pickle.dump(review_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
