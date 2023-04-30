import pandas as pd
import re
import csv
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import pickle

#read file
df = pd.read_csv('/Users/cheyenne/Desktop/4300/final_project/4300-Template-Spring-2023/backend/match.csv', usecols=['recipe_id','review'], engine='python')
df = df.dropna(how='all')
df['empty_space_removed'] = df['review'].map(lambda x: " ".join(str(x).split()))
df['puncts_removed_review'] = df['empty_space_removed'].map(lambda x: re.sub(r'[^\w\s]', '', str(x))) #remove punctuations 
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
#reference: https://www.holisticseo.digital/python-seo/nltk/lemmatize
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


#find the optimal number of topics using coherence score
#input:t_num(list of topic numbers to choose from)
def c_score(t_num):
  lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dict, num_topics=t_num, random_state=50, chunksize=100, alpha='auto', eta='auto',passes=50) 
  coherence_model_lda = CoherenceModel(model=lda_model, texts=reviews, dictionary=dict, coherence='c_v')
  coherence_score = coherence_model_lda.get_coherence()
  return coherence_score


possible_topic_nums = [2,3,4,5]

def optimal_find(num_lst):
  c_score_lst = []
  for n in num_lst:
    score = c_score(n)
    c_score_lst.append(score)
  best_topic_num = possible_topic_nums[c_score_lst.index(max(c_score_lst))]
  return best_topic_num

if __name__ == "__main__":
  best_num = optimal_find(possible_topic_nums)
  lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dict, num_topics=best_num, random_state=50, chunksize=100, alpha='auto', eta='auto',passes=50,iterations=100) 
  topics = lda_model.print_topics(num_words=5)
  

  best_topics = ["time", "easy", "delicious"]


#use LDA topics to assign reviews to corresponding topics
#get_term_topics returns the probability that a word belongs to a particular topic
#similarly, we also have get_document_topics
  def get_label(review):
    if type(review) == str:
      bow_format = dict.doc2bow(review.split())
    else:
      bow_format = review
    topics_prob = lda_model.get_document_topics(bow_format)
    if max(topics_prob[0][1], topics_prob[1][1], topics_prob[2][1]) == topics_prob[0][1]:
      label = "People who've tried out this recipe love it so much that they're already planning on customizing it for their family cookbook!"
    elif max(topics_prob[0][1], topics_prob[1][1], topics_prob[2][1]) == topics_prob[1][1]:
      label = "Not much kitchen experience? Check out this super easy recipe!"
    else:
      label = "Simply delicious!"
    return label

#dictionary of the format {id:(# of 1st-topic reviews, # of 2nd-topic reviews)}
  review_labels = {}
  for i in range(len(corpus)):
    label = get_label(corpus[i])
    if ids[i] not in review_labels:
      review_labels[ids[i]] = [0,0,0]
    if label == "People who've tried out this recipe love it so much that they're already planning on customizing it for their family cookbook!":
      review_labels[ids[i]][0] += 1 
    elif label == "No kitchen experience? Check out this super easy recipe!": 
      review_labels[ids[i]][1] += 1
    else:
      review_labels[ids[i]][2] += 1
      
  #assign topics to recipes
  for k in review_labels:
    if max(review_labels[k][0], review_labels[k][1], review_labels[k][2]) == review_labels[k][0]:
      review_labels[k] = "People who've tried out this recipe love it so much that they're already planning on customizing it for their family cookbook!"
    elif max(review_labels[k][0], review_labels[k][1], review_labels[k][2]) == review_labels[k][1]:
      review_labels[k] = "No kitchen experience? Check out this super easy recipe!"
    else:
      review_labels[k] = "Simply delicious!"
  
  print(review_labels)

  with open('pickled_dict.pickle', 'wb') as handle:
    pickle.dump(review_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
