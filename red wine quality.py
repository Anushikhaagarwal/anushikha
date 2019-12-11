#!/usr/bin/env python
# coding: utf-8

# # The task is to Predict the quality of Red Wine.
# As it is a multiclass problem we will use classification technique because the the target variable is in levels so its better to use classification technique.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#loading the dataset
redwine = pd.read_csv('winequality-red.csv')


# In[3]:


redwine.head()


# In[4]:


redwine.info()


# In[5]:


redwine.shape


# In[6]:


redwine.describe()


# In[7]:


redwine.dtypes


# In[8]:


redwine.nunique()


# In[38]:


redwine.isna().sum()


# In[10]:


sns.pairplot(redwine)


# In[11]:


sns.distplot(redwine['quality'])


# In[12]:


redwine.columns


# In[13]:


#Visualizing the distribution of the target variable
plt.figure(1,figsize=(12,4))
plt.subplot(121)
redwine['quality'].value_counts(normalize=True).plot.bar()
plt.subplot(122)
redwine['quality'].value_counts(normalize=False).plot.bar()


# In[14]:


redwine.skew()


# In[15]:


#Visulaize the distribution of all numerical variables
redwine.hist(figsize=(20,30),bins=50,xlabelsize=8,ylabelsize=8,grid=False)


# In[16]:


#Measure skewness of all variables
from scipy.stats import skew
redwine_skew = redwine.apply(lambda x: skew(x.dropna()))
redwine_skew[redwine_skew > .75]


# In[17]:


#Using log transformation to normalise the variables
change=['fixed acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','sulphates','alcohol']
redwine[change] = np.log1p(redwine[change])


# In[18]:


redwine.hist(figsize=(20,30),bins=50,xlabelsize=8,ylabelsize=8,grid=False)


# In[19]:


redwine.columns


# In[20]:


redwine.skew()


# In[21]:


redwine['quality'].value_counts()


# In[22]:


redwine.nunique()


# In[23]:


def isgood(quality):
    if quality == 5 & 6:
        return 1
    else:
        return 0


# In[24]:


redwine['good'] = redwine['quality'].apply(isgood)


# In[25]:


redwine.columns


# In[26]:


redwine['good'].value_counts()


# In[27]:


plt.figure(figsize=(12,10))
cor = redwine.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[28]:


x= redwine[['fixed acidity','volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH','sulphates', 'alcohol']]
y = redwine[['good']]


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=101)


# In[30]:


y_train.shape


# In[31]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[32]:


#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 10000,criterion = 'entropy',random_state = 0)
#classifier.fit(x_train, y_train)


# In[33]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)


# In[34]:


#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(x_train, y_train)


# In[35]:


y_pred = classifier.predict(x_test)


# In[36]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[37]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= x_train, y= y_train, cv=10)
#accuracies.mean()
#accuracies.min()
accuracies.max()


# The End.
