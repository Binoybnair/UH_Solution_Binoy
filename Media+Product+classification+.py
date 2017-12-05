
# coding: utf-8

# ## Coding Assignment
# ## Media Product Classification
# 
# ### Multi-class classification for media products into 4 categories - movies, books, music and rest
# 

# In[1]:


import string #For String manipulation,will be used for data preparation


# In[2]:


from nltk.corpus import stopwords #To remove stopwords from the textdata before passing it for analysis


# In[3]:


import pandas as pd 
import numpy as np


# In[4]:


import seaborn as sns #For data visulaization
get_ipython().magic('matplotlib inline')


# In[6]:


df = pd.read_csv('train.csv') 
#Training data set


# In[7]:


df_t = pd.read_csv('evaluation.csv') #Evaluation data set 


# In[44]:


df.describe()


# In[45]:


df.info()


# In[46]:


df.head(5)


# In[68]:


sum(df['label'].isnull()) #Check if there are any missing values in column label


# In[69]:


sns.countplot(df['label']) #visualizing the distribution of label data


# In[70]:


sum(df['storeId'].isnull()) #Check missing values in storeId


# In[73]:


df[df['storeId'].isnull()]['label'].nunique() #Check number of unique labels for all the missing store ids


# In[47]:


df_t.head(5)


# In[11]:


df['textdata'] = df['additionalAttributes'].fillna('')+df['breadcrumbs'].fillna('') #Concatenating additional attributes and breadcrumbs column and forming a new column which would be used for analysis


# In[12]:


df_t['textdata'] = df_t['additionalAttributes'].fillna('')+df_t['breadcrumbs'].fillna('') # Performing the same step above for test data set


# In[50]:


df.head(5) #After adding new column


# In[51]:


df_t.head(5)


# In[52]:


df.groupby('label').describe()


# In[53]:


sum(df['textdata'].isnull()) #Check for missing values


# In[54]:


sum(df_t['textdata'].isnull()) #Check for missing values


# In[8]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[9]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[10]:


#We will use SciKit Learn's pipeline capabilities to store a pipeline of workflow. 
#This will allow us to set up all the transformations that we will do to the data for future use

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[13]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(df['textdata'], df['label'], test_size=0.4)


# In[20]:


pipeline.fit(df.textdata,df.label) #Train the algorithm 


# In[14]:


pipeline.fit(msg_train,label_train)


# In[15]:


pred_label = pipeline.predict(msg_test)


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(pred_label,label_test))


# In[22]:


predicted_label = pipeline.predict(df_t.textdata) #Predict data on test data based on the model obtained above


# In[55]:


df_t['PredictedLabel'] = predicted_label #Add the predicted values to the original test data set


# In[58]:


df_t.groupby('PredictedLabel').count()['textdata'] #Distribution of label data


# In[74]:


df_t.to_csv('submissions.csv',index=False) # Write dataframe to csv file and set index as False to avoid unnecessary column


# In[ ]:


#----------------------------------------------------------------------------------------------------------------------------

