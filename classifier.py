# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:15:16 2020

@author: Gautam_Pai (Kaggler)
"""

""" 
Import the packages
"""
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
%matplotlib inline

"""
Reading all the datasets into dataframes trainining, validation and dictionary
"""
training = pd.read_excel("GlobalHackathonFY20_BeginnerProblem_MasterData.xlsx", sheet_name="Training")
validation = pd.read_excel("GlobalHackathonFY20_BeginnerProblem_MasterData.xlsx", sheet_name="Validation")
dictionary = pd.read_excel("GlobalHackathonFY20_BeginnerProblem_MasterData.xlsx", sheet_name="DictionarySet")

"""
Exploratory Data Analysis
"""
training.shape
validation.shape
dictionary.shape


def get_missing_values(dataset):
    """
    Method to get missing values for a given dataset
    """
    missing_values =[]
    for col in dataset.columns:
        missing_values.append(dataset[col].isnull().sum())
        print("The number of NAs in {0} is {1}".format(col,dataset[col].isnull().sum()))
    return missing_values

missing_values = get_missing_values(training)
missing_series = pd.Series(missing_values, index=training.columns)
missing_series.plot(kind='barh')

get_missing_values(validation)
get_missing_values(dictionary)

def get_duplicate_records(dataset, subset):
    """
    Method to return the duplicate records within the dataset

    Parameters
    ----------
    dataset : dataframe object
        The dataframe object where duplicated records need to be identified.
    subset : list of columns
        columns to search for duplicate records.

    Returns
    -------
    Duplicated records.

    """
    return dataset[dataset.duplicated(subset=subset)]

get_duplicate_records(training, subset=["title","description"])
get_duplicate_records(training, subset=["title","body"])
get_duplicate_records(training, subset=["title","body","description"])

get_duplicate_records(dictionary, subset=["Title","Description"])

training = training.drop_duplicates(subset=["title","description","body"])
dictionary = dictionary.drop_duplicates(subset=["Title","Description"])

training["Category"].value_counts()
#Visualizaing the Category column distribution
sns.catplot(data=training, x="Category", kind='count', height=4,aspect=5)

def preprocess_text(column, df):
    """
    Method to preprocess texts within the given column and it contains lowercasing, removing stopwords and lemmatization

    Parameters
    ----------
    column : string
        The column of the dataframe.
    df : DataFrame object
        The dataframe object where the preprocessing is to be done.

    Returns
    -------
    Dataframe object with the preprocessed feature.

    """
    for index in range(0,df.shape[0]):
        try:
            title = re.sub("[^a-zA-Z]",' ', df[column][index])
            title = [str.lower() for str in word_tokenize(title)]
            #removing stopwords
            title = [str for str in title if not str in set(stopwords.words('english')) and len(str) > 2]
            lemmatizer = WordNetLemmatizer()
            title = [lemmatizer.lemmatize(str) for str in title]
            title = ' '.join(title)
            df[column][index] = title
        except:
            pass

"""
Implement Preprocessing of all fields in Dictionary dataset
"""
preprocess_text('Title', dictionary)
dictionary['Description'].fillna(' ', inplace=True)
preprocess_text('Description', dictionary)
dictionary.to_csv("Dictionary_preprocessed.csv", index=False)

"""
Implement Preprocessing of all fields in Training dataset
"""
preprocess_text('title', training)
training['description'].fillna(' ', inplace=True)
preprocess_text('description', training)
training['body'].fillna(' ', inplace=True)
preprocess_text('body', training)

"""
Eliminating records which are missing in title and body after doing preprocessing
"""
training = training[~training['title'].isnull()]
training = training[~training['body'].isnull()]
training.to_csv("Training_preprocessed.csv", index=False)


def lemmatize_stemming(text):
    """
    Transform the text to its base form

    Parameters
    ----------
    text : string
        The text to transform.

    Returns
    -------
    string
        The transformed text.

    """
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    """
    Preprocess text using gensim utils

    Parameters
    ----------
    text : string
        The text to preprocess.

    Returns
    -------
    result : string
        Preprocessed text after eliminating stopwords and transforming the text to its base type.

    """
    #print(text)
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

def get_processeddocs(data):
    """
    Gets the processed doc i.e. lowercase, tokenize(using gensim), removal of stopwords

    Parameters
    ----------
    data : string
        columns vector containing texts.

    Returns
    -------
    processed_docs : string
        column vector containing texts preprocessed.

    """
    processed_docs = []

    for doc in data:
        processed_docs.append(preprocess(doc))
        
    return processed_docs

"""
Preprocess the title + description fields of the dictionary dataset
"""
dictionary["title_description"] = dictionary['Title']+' '+dictionary['Description']
dict_processed_docs = get_processeddocs(dictionary['title_description'])
training["title_description"] = training['title']+' '+training['description']
train_processed_docs = get_processeddocs(training['title_description'])
docs = dict_processed_docs + train_processed_docs
dict_corpus = gensim.corpora.Dictionary(docs)

#Checkpoint
# dict_corpus_copy = dict_corpus
# dictionary.filter_extremes(no_below=15, no_above=0.1)

bow_corpus_dict = [dict_corpus.doc2bow(doc) for doc in dict_processed_docs]

#Check sample output
[[(dict_corpus.id2token[id], freq) for id, freq in cp] for cp in bow_corpus_dict[:10]]

lda_model =  gensim.models.LdaMulticore(bow_corpus_dict, 
                                   num_topics = 15, 
                                   id2word = dict_corpus,                                    
                                   passes = 10,
                                   eval_every=1,
                                   workers = None)

lda_model.print_topics()

for idx, topic in lda_model.print_topics(-1):
    print("Category: {} \nWords: {}".format(idx, topic ))
    print("\n")

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(bow_corpus_dict)) 
#Perplexity:  -6.83

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dict_corpus, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Coherence Score:  0.4264283394676994

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model =  gensim.models.LdaMulticore(corpus, 
                                   num_topics = num_topics, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   eval_every=1,
                                   workers = None)
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dict_corpus, corpus=bow_corpus_dict, texts=docs, start=3, limit=21, step=3)

limit=21; start=3; step=3;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

#Save the model
lda_model.save("lda_model.model")

lda_model.show_topics(formatted=False)

#Visualization of the keywords in a specific topic is only enabled in jupyter notebook, we shall pickle the vis object and unpickle it in jupyter notebook
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus_dict, dict_corpus)
f = open('pyldavis.pkl', 'wb')
pkl.dump(vis, f)
f.close()

#Find Bag of words for training dataset
bow_corpus_train = [dict_corpus.doc2bow(doc) for doc in train_processed_docs]

def format_topics_sentences(ldamodel, corpus, texts):
    """
    Find the dominant topic in each sentence

    Parameters
    ----------
    ldamodel : gensim.models.LdaMulticore
        Latent Dirichlet Allocation model for classifying into topics.
    corpus : list
        bag of words corpus.
    texts : Series object of strings
        column vector containing strings.

    Returns
    -------
    sent_topics_df : DataFrame
        DataFrame containing Dominant Topic, Percentage contribution of the terms and Keywords.

    """
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0: #=> dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)                
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts).reset_index()
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)

#Label the training dataset into relevant topics based on the terms/keywords
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=bow_corpus_train, texts=training['title_description'])

#Get the topic and its keywords in the training dataset
training[['Topic','Topic_Keywords']] = df_topic_sents_keywords[["Dominant_Topic","Topic_Keywords"]]

preprocess_text("Topic_Keywords", training) #for removing comma delimiters
#Save the dataset
training.to_csv("Training_Preprocessed_Topic.csv", index=False)



#**********************************Modelling **************************************************
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD

#training.dropna(subset=["body","Topic_Keywords"], inplace=True)
training['content'] = training['Topic_Keywords']+ ' '+training['body']

X_train,X_test,y_train,y_test = train_test_split(training['content'],training['Category'],test_size=0.25, random_state=18)

#Load PRetrained RF classifier
# f = open("model.pkl","rb")
# classifier = pkl.load(f)
# f.close()

count_vectorizer = CountVectorizer(stop_words='english', lowercase=True, max_features=1300) #1275 0.5636363636363636
train_count = count_vectorizer.fit_transform(X_train)
test_count = count_vectorizer.transform(X_test)
# tf_idf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=1275) #1225 .55883
# train_count = tf_idf_vectorizer.fit_transform(X_train)
# test_count = tf_idf_vectorizer.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 700, 
                                    class_weight = 'balanced_subsample', 
                                    #oob_score=True,
                                    max_features = "sqrt", 
                                    n_jobs=-1,
                                    bootstrap=False,
                                    # min_samples_leaf = 4, 
                                    # min_samples_split = 10,
                                    random_state=34)

classifier.fit(train_count,y_train)
pred = classifier.predict(test_count)

print("Accuracy {}".format(accuracy_score(y_test,pred)))
print("*************Confusion Matrix****************************")
print(confusion_matrix(y_test,pred, labels=y_test.unique()))
print(classification_report(y_test,pred))

#Save the model
f = open('model.pkl', 'wb')
pkl.dump(classifier, f)
f.close()

#Analyzing the misclassified samples
df = pd.DataFrame(np.column_stack([y_test,pred,X_test.values]))
#**********************************Validation Section/Submitting the prediction**********************************

validation['description'].fillna(' ', inplace=True)
validation['title_description'] = validation['title']+' '+validation['description']

validate_processed_docs = get_processeddocs(validation['title_description'])

bow_corpus_validate = [dict_corpus.doc2bow(doc) for doc in validate_processed_docs]

"""Generate topic and its keywords for validation dataset"""
df_validate_topic_keywords = format_topics_sentences(ldamodel=lda_model, corpus=bow_corpus_validate, texts=validation['title_description'])
validation[['Topic','Topic_Keywords']] = df_validate_topic_keywords[["Dominant_Topic","Topic_Keywords"]]

preprocess_text("Topic_Keywords", validation) #for removing comma delimiters
preprocess_text("body", validation) 
validation.to_csv("validate_topic.csv", index=False)

validation['content'] = validation['Topic_Keywords']+ ' '+validation['body']
validate_content = count_vectorizer.transform(validation['content'])

prediction_class = classifier.predict(validate_content)
validation['Category'] = prediction_class

scoring_file = pd.read_excel("Beginner_Submission_File.xlsx", sheet_name="Sheet1")
scoring_file['Prediction'] = validation['Category']
scoring_file.to_excel("Beginner_File.xlsx", index=False)





