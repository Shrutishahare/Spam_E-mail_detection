#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('spam.csv',encoding = "ISO-8859-1")


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


df.info


# In[6]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.sample(5)


# In[8]:


df.rename(columns={'v1':'target' , 'v2':'text'},inplace=True)
df.sample(5)


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[10]:


df['target']= encoder.fit_transform(df['target'])


# In[11]:


df.head()


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df=df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# In[16]:


df.shape


# In[17]:


df.head()


# In[18]:


df['target'].value_counts()


# In[19]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[20]:


import nltk


# In[21]:


get_ipython().system('pip install nltk')


# In[22]:


nltk.download('punkt')


# In[23]:


df['num_alphabets']=df['text'].apply(len)


# In[24]:


df.head()


# In[25]:


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[26]:


df.head()


# In[27]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df[['num_alphabets','num_words','num_sentences']].describe()


# In[30]:


df[df['target']==0][['num_alphabets','num_words','num_sentences']].describe()


# In[31]:


df[df['target']==1][['num_alphabets','num_words','num_sentences']].describe()


# In[32]:


import seaborn as sns


# In[33]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_alphabets'])
sns.histplot(df[df['target']==1]['num_alphabets'],color='red')


# In[34]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')


# In[35]:


sns.pairplot(df,hue='target')


# In[36]:


sns.heatmap(df.corr(),annot=True)


# In[37]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[44]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('loving')


# In[39]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[41]:


import string
string.punctuation


# In[46]:


transform_text('Did u like my prep on ML')


# In[47]:


transform_text('I loved the YT lectures?')


# In[48]:


df['text'][100]


# In[49]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('dancing')


# In[50]:


df['transformed_text']=df['text'].apply(transform_text)


# In[51]:


df.head()


# In[52]:


get_ipython().system('pip install wordcloud')


# In[53]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[54]:


spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[55]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[56]:


ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[57]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[58]:


df.head()


# In[59]:


spam_corpus =[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for words in msg.split():
        spam_corpus.append(words)


# In[60]:


len(spam_corpus)


# In[61]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show


# In[62]:


ham_corpus =[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for words in msg.split():
        ham_corpus.append(words)


# In[63]:


len(ham_corpus)


# In[64]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show


# In[65]:


from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
cv = CountVectorizer()
tfidf= TfidfVectorizer()


# In[66]:


X= tfidf.fit_transform(df['transformed_text']).toarray()


# In[67]:


X.shape


# In[68]:


y = df['target'].values


# In[69]:


y


# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[72]:


from sklearn.naive_bayes import GaussianNB , MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[73]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[74]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[75]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[76]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[90]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[96]:


svc=SVC(kernel='sigmoid',gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear',penalty='l1')
rfc=RandomForestClassifier(n_estimators=50,random_state=2)
abc=AdaBoostClassifier(n_estimators=50,random_state=2)
bc=BaggingClassifier(n_estimators=50,random_state=2)
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)
gbc=GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[97]:


clfs={
    'SVC': svc,
    'KN' : knc,
    'NB': mnb,
    'DT':dtc,
    'LR':lrc,
    'RF':rfc,
    'AB':abc,
    'BC':bc,
    'ETC':etc,
    'GBC':gbc
}


# In[98]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[99]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[100]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf,X_train,y_train,X_test,y_test)
    
    print("For",name)
    print("Accuracy -",current_accuracy)
    print("Precision -",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[101]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Accuracy',ascending=False)


# In[102]:


performance_df


# In[103]:


performance_df1=pd.melt(performance_df,id_vars="Algorithm")


# In[104]:


performance_df1


# In[107]:


sns.catplot(x = 'Algorithm', y='value',
           hue='variable', data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[111]:


temp_df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores})


# In[113]:


temp_df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max':accuracy_scores,'Precision_max':precision_scores})


# In[ ]:


new_df=performance_df.merge(temp_df,on='Algorithm')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




