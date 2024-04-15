#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# In[3]:


df_train = pd.read_csv('/kaggle/input/twitter/train_tweet.csv')
df_test = pd.read_csv('/kaggle/input/twitter/test_tweets.csv')


# In[4]:


df_train.shape,df_test.shape


# In[5]:


df_train.head()


# In[6]:


df_test.head()


# In[7]:


df_train.isna().sum()
df_test.isna().sum()


# ## Check for positive and negative comments

# In[8]:


df_train[df_train['label'] == 1].head(10)


# In[9]:


df_train[df_train['label']==0].head(10)


# In[10]:


negative_count = df_train[df_train['label'] == 1].shape[0]
neutral_count = df_train[df_train['label'] == 0].shape[0]

labels = ['Neutral', 'Negative']
sizes = [neutral_count, negative_count]
colors = ['lightgreen', 'lightcoral']
explode = (0.5, 0) 

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Distribution of Neutral and Negative Labels')
plt.show()


# In[11]:


length_train = df_train['tweet'].str.len().plot.hist(color = 'black', figsize = (6, 4))
length_test = df_test['tweet'].str.len().plot.hist(color = 'orange', figsize = (6, 4))


# #### Add length of tweet

# In[12]:


df_train['len'] = df_train['tweet'].str.len()
df_test['len'] = df_test['tweet'].str.len()

print(df_train.head(10))
print('*'*150)
print(df_test.head(10))


# In[13]:


df_train.groupby('label').describe()


# In[14]:


mean_label_by_length = df_train.groupby(df_train['tweet'].str.len())['label'].mean()
plt.figure(figsize=(10, 6))
plt.hist(mean_label_by_length, color='black', bins=20) 
plt.title('Variation of Length vs Mean Label')
plt.xlabel('Mean Label')
plt.ylabel('Frequency')
plt.show()


# *   ###  using countvectorizer collect unique words and count of that word in each row

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = 'english')   # array of words not include is,and etc
words = cv.fit_transform(df_train.tweet)


# sum of each word count
sum_word = words.sum(axis=0)

# sort words according to frequency
words_freq = [(word, sum_word[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

freq = pd.DataFrame(words_freq, columns=['word', 'freq'])

freq.head(50).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'cyan')
plt.title("Most Frequently Occuring Words - Top 50")


# In[16]:


from wordcloud import WordCloud

word_freq_dict = dict(words_freq)

# Set the font family globally for the WordCloud library
plt.rcParams['font.family'] = 'Arial'

# Create a WordCloud object
wordcloud = WordCloud(background_color='white', width=1000, height=1000).generate_from_frequencies(word_freq_dict)

# Display the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ### Negative Words

# In[17]:


neg_words =' '.join([text for text in df_train['tweet'][df_train['label'] == 1]])

wordcloud = WordCloud(background_color = 'cyan', width=800, height=500, random_state = 0, max_font_size = 110).generate(neg_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Negative Words')
plt.show()


# ### Neutral Words

# In[18]:


neutral_words =' '.join([text for text in df_train['tweet'][df_train['label'] == 0]])

wordcloud = WordCloud(background_color = 'violet', width=800, height=500, random_state = 0, max_font_size = 110).generate(neutral_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Neutral Words')
plt.show()


# * ***#: This matches the literal character "#" in the text.
# (\w+): This is a capturing group denoted by the parentheses (). Within this group:
# \w+: This matches one or more word characters. \w represents any alphanumeric character (equivalent to [a-zA-Z0-9_]), and + indicates that the preceding pattern (in this case, \w) should occur one or more times. So, \w+ will match one or more consecutive alphanumeric characters.***

# In[19]:


import re
def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[20]:



#### non-racist/sexist tweets
ht_neutral = hashtag_extract(df_train['tweet'][df_train['label']==0])

#### racist/sexist tweets
ht_negative = hashtag_extract(df_train['tweet'][df_train['label']==1])

### flatten the list
ht_neutral = sum(ht_neutral,[])
ht_negative = sum(ht_negative,[])


# In[21]:


ht_neutral


# In[22]:


ht_negative


# #### count frequency of each flatten word

# In[23]:


import nltk
from nltk.tokenize import word_tokenize
c = nltk.FreqDist(ht_neutral)
df = pd.DataFrame({'Hashtag': list(c.keys()),
                  'Count': list(c.values())})

df = df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(10, 8))  
ax = sns.barplot(data=df.sort_values(by='Count', ascending=False), x="Count", y="Hashtag", palette="viridis")  
ax.set_xlabel('Frequency') 
ax.set_ylabel('Hashtag')   
ax.set_title('Top 50 Most Frequent Hashtags in Non-Racist/Sexist Tweets')  
plt.show()


# In[24]:


import nltk
from nltk.tokenize import word_tokenize
d = nltk.FreqDist(ht_negative)
df1 = pd.DataFrame({'Hashtag': list(d.keys()),
                  'Count': list(d.values())})

df1 = df1.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(10, 8))  
ax = sns.barplot(data=df1.sort_values(by='Count', ascending=False), x="Count", y="Hashtag", palette="viridis")  
ax.set_xlabel('Frequency') 
ax.set_ylabel('Hashtag')   
ax.set_title('Top 50 Most Frequent Hashtags in Racist/Sexist Tweets')  
plt.show()


# In[25]:


tok_tweet = df_train['tweet'].apply(lambda x: x.split()) 


# In[26]:


import gensim

### word to vector model

wv = gensim.models.Word2Vec(
            tok_tweet,
            vector_size=200,  # Desired number of features/independent variables 
            window=5,         # Context window size
            min_count=2,
            sg=1,             # 1 for skip-gram model
            hs=0,
            negative=10,      # For negative sampling
            workers=2,        # Number of cores
            seed=34
)


# In[27]:


wv.train(tok_tweet, total_examples= len(df_train['tweet']), epochs=20)


# In[28]:


wv.wv.most_similar(positive = "user")


# In[29]:


wv.wv.most_similar(negative = "racist")


# In[30]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer   # removes common morphological and inflectional endings from words


# In[31]:


train_corpus = []

for i in range(0, 31962):
    review = re.sub('[^a-zA-Z]', ' ', df_train['tweet'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    train_corpus.append(review)
    


# In[32]:


test_corpus = []

for i in range(0, 17197):
    review = re.sub('[^a-zA-Z]', ' ', df_test['tweet'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus.append(review)
    


# ### for training

# In[33]:


### creating a bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)
x = cv.fit_transform(train_corpus).toarray()
y = df_train.iloc[:, 1]


# In[34]:


print(x.shape)
print(y.shape)


# ### For testing

# In[35]:


cv = CountVectorizer(max_features = 2500)
x_test = cv.fit_transform(test_corpus).toarray()

print(x_test.shape)


# In[36]:


from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[37]:


print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)


# In[38]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)


# ## RandomForestClassifier

# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# In[40]:


rf = RandomForestClassifier()
rf.fit(x_train, y_train)

y_pred = rf.predict(x_valid)

print("Training Accuracy :", rf.score(x_train, y_train))
print("Validation Accuracy :", rf.score(x_valid, y_valid))
print("F1 score :", f1_score(y_valid, y_pred))
cm = confusion_matrix(y_valid, y_pred)
print(cm)


# ## Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_valid)

print("Training Accuracy :", lr.score(x_train, y_train))
print("Validation Accuracy :", lr.score(x_valid, y_valid))
print("F1 score :", f1_score(y_valid, y_pred))
cm = confusion_matrix(y_valid, y_pred)
print(cm)


# ## DecisionTree Classifier

# In[42]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_valid)

print("Training Accuracy :", dt.score(x_train, y_train))
print("Validation Accuracy :", dt.score(x_valid, y_valid))
print("F1 score :", f1_score(y_valid, y_pred))
cm = confusion_matrix(y_valid, y_pred)
print(cm)


# ## GradientBoost

# In[43]:


from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_valid)

print("Training Accuracy :", xgb.score(x_train, y_train))
print("Validation Accuracy :", xgb.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)


# In[ ]:




