#!/usr/bin/env python
# coding: utf-8

# In[58]:


pip install wordcloud 


# In[59]:


# import library, pandas to read the dataset, matplotlib to visualize data, and re to clean the text 
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import numpy as np


# In[60]:


# read data stored in file name Car-details.csv using pd.read_csv, then store the data in df dataframe
df = pd.read_csv('drugs.csv')


# In[61]:


#show first ten dataframe content
df.head(10)


# In[62]:


# Show last five rows
df.tail()


# In[63]:


df.shape


# In[64]:


# explore the data type at each series
df.info()


# In[65]:


df.describe()


# In[66]:


# check to null values 
df.isna().sum()


# In[67]:


df.dropna(subset=['condition'],inplace=True)


# In[68]:


# Assess if there are any duplicates.
sum(df.duplicated())


# In[69]:


#Check Outlier
sns.boxplot(data=df)
plt.xticks(rotation=25);


# In[70]:


# let's make a new column review sentiment 

df.loc[(df['rating'] >= 5), 'Sentiment'] = ('positive')
df.loc[(df['rating'] < 5), 'Sentiment'] = ('negative')


# In[71]:


df['Sentiment'].value_counts()


# In[72]:


condition_df = df.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)


# In[73]:


condition_df


# In[74]:


condition_df[0:25].plot(kind='bar', figsize=(20,8), fontsize=10, color='g')
plt.xlabel("Condition", fontsize=20)
plt.ylabel("Number of drugs", fontsize=20)
plt.title("Number of Drugs per Condition (Top 25 Conditions)", fontsize=25)
plt.show()


# In[75]:


ratings = df['rating'].value_counts().sort_values(ascending=False)


# In[76]:


ratings.plot(kind='bar', figsize=(8,5), fontsize=10, color='b')
plt.xlabel("Ratings", fontsize=15)
plt.xticks(rotation='horizontal')
plt.ylabel("Number of Reviews", fontsize=15)
plt.title("Bar Chart (Ratings vs. Number of Reviews)", fontsize=20)
plt.show()


# In[77]:


colors1 = ['whitesmoke','lightsalmon','lightgreen','moccasin','powderblue','violet','lavender','pink','beige','lightcyan']
explode = np.full(shape=10, fill_value=0.05, dtype='float64')
ratings.plot.pie(labels=None, colors=colors1, autopct='%1.0f%%', pctdistance=0.85, explode=explode, startangle=90, figsize=(6,6))
centre_circle = plt.Circle((0,0), 0.70, fc='white')
plt.gcf().gca().add_artist(centre_circle)
plt.legend(ratings.index, loc='center')
plt.xlabel("")
plt.ylabel("")
plt.title("Percentage of Ratings", fontsize=15)
plt.show()


# In[78]:


usefulDrugs = df.groupby(['drugName'])['usefulCount'].nunique().sort_values(ascending=False)


# In[79]:


usefulDrugs


# In[80]:


usefulDrugs[0:30].plot(kind='bar', figsize=(25,10), fontsize=10, color='slateblue')
plt.xlabel("Drug Name", fontsize=20)
plt.ylabel("Useful Count", fontsize=20)
plt.title("Useful Count of Top 30 Drugs", fontsize=25)
plt.show()


# In[81]:


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)


# In[82]:


from wordcloud import WordCloud


# In[83]:


import string
# Join the different processed reviews together.
long_string = ','.join(list(df['review'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# In[84]:


# let's make a new column review sentiment 

df.loc[(df['rating'] >= 5), 'Review_Sentiment'] = 1
df.loc[(df['rating'] < 5), 'Review_Sentiment'] = 0


# In[85]:


df['Review_Sentiment'].value_counts()


# In[86]:


# a pie chart to represent the sentiments of the patients

size = [161491, 53572]
colors = ['pink', 'lightblue']
labels = "Positive Sentiment","Negative Sentiment"
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, colors = colors, labels = labels, explode = explode, autopct = '%.2f%%')
plt.axis('off')
plt.title('Pie Chart Representation of Sentiments', fontsize = 25)
plt.legend()
plt.show()


# In[87]:


df.head()


# In[88]:


# This barplot show the top 10 conditions the people are suffering.
cond = dict(df['condition'].value_counts())
top_condition = list(cond.keys())[0:10]
values = list(cond.values())[0:10]
sns.set(style = 'darkgrid', font_scale = 1.3)
plt.rcParams['figure.figsize'] = [18, 7]


# In[89]:


sns_ = sns.barplot(x = top_condition, y = values, palette = 'winter')
sns_.set_title("Top 10 conditions")
sns_.set_xlabel("Conditions")
sns_.set_ylabel("Count");


# In[90]:


get_ipython().system('pip install nltk')


# In[91]:


import nltk
nltk.download('punkt')


# In[92]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt


# In[93]:


from nltk import word_tokenize,sent_tokenize


# In[94]:


#make a copy
df2 = df.copy(deep = True)


# In[95]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


# In[96]:


def ana_sentiment(text):
    sia = SIA()
    results = []
    text_s=tokenize.sent_tokenize(text)

    for s in text_s:
      pol_score = sia.polarity_scores(s)
      pol_score['headline'] = s
      results.append(pol_score)

    df = pd.DataFrame.from_records(results)
    sent_scores=list(df.mean(axis=0)) 
    return(sent_scores)


# In[97]:


neg_score=[];neu_score=[];pos_score=[]
for ind, row in df2.iterrows():
    s=ana_sentiment(row['review'])
    neg_score.append(s[1]); neu_score.append(s[2]); pos_score.append(s[3])

data_clean=df2.copy()
data_clean['neg_score']=neg_score;
data_clean['neu_score']=neu_score;
data_clean['pos_score']=pos_score
data_clean=data_clean.drop(['drugName','condition','review','rating','date'],axis=1)
data_clean.head(10)


# In[ ]:





# In[ ]:




