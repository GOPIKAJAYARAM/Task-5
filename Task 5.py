#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
# conda install -c conda-forge/label/gcc7 missing


# #Importing both tables 

# In[2]:


df_user= pd.read_csv('takehome_users.csv')
print(df_user.shape)
df_user.head()


# In[3]:


df_user.nunique()
type(df_user)


# In[4]:


df_usage= pd.read_csv('takehome_user_engagement.csv')
print(df_usage.shape)
df_usage.head()


# In[5]:


#unique values of each column 
df_usage.nunique()


# In[6]:


df_usage.visited.value_counts()


# In[7]:


#visted column is not necessary since every column shows the details of the person who has visited 
#hens dropping visited column 
df_usage.drop('visited',axis=1,inplace=True)
df_usage.head()


# In[8]:


df_usage.info()


# In[9]:


#converting the object type of time_stamp in to date time 
df_usage['time_stamp']= pd.to_datetime(df_usage['time_stamp'], format='%Y-%m-%d %H:%M:%S')
df_usage.info()


# In[10]:


df_usage=df_usage.sort_values(by="time_stamp")
df_usage


# In[11]:


from datetime import datetime
df_usage["time_stamp"]=df_usage["time_stamp"].dt.date
df_usage['time_stamp']= pd.to_datetime(df_usage['time_stamp'], format='%Y-%m-%d')
df_usage=df_usage.reset_index(drop=True)


# In[12]:


def find_if_adopted (df):
    from datetime import timedelta
    df1=df.drop_duplicates(subset='time_stamp')
    diff_3day=df1['time_stamp'].diff(periods=2)
    return any(diff_3day <= timedelta(days=7))   


# In[13]:


adopt=df_usage.groupby('user_id').apply(find_if_adopted)


# In[14]:


df_usage=df_usage.drop("time_stamp",axis=1)
df_usage=df_usage.drop_duplicates(subset='user_id')
df_usage=df_usage.reset_index(drop=True)
adopt=np.array(adopt)
df_2=pd.DataFrame({"adopted_user":adopt})
df_usage['adopted_user']=df_2['adopted_user']
df_usage['adopted_user']=df_usage['adopted_user'].astype(int)
df_usage


# In[15]:


df_usage['adopted_user'].value_counts()
#so we have 1656 adopted user form 8823 unique users


# In[16]:


# df_user=df_user.rename(columns={"object_id":"user_id"})
df_user=df_user.set_index('object_id')
df_user.index.name='user_id'


# In[17]:


df_usage=df_usage.set_index('user_id')
df_usage.tail(20)


# In[18]:


#first changed to index of both the dataset to user id so that they allign while concating 
df_data= pd.concat([df_user,df_usage], axis=1, join='inner')


# In[19]:


df_data.loc[11504]


# In[20]:


print(df_usage.shape)
print(df_user.shape)
print(df_data.shape)


# In[21]:


#The user id's which invited other adotped user ids .
df_data[df_data['adopted_user']==1].invited_by_user_id.value_counts().head()


# In[22]:


df_source = df_data.creation_source.value_counts()
df_source = df_source.reset_index()
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
explode = (0.05, 0.05, 0.05,0.05,0.05)
colors = ['gold', 'yellowgreen', 'lightcoral','lightblue','orange']
# Put parameter values
plt.pie(
    df_source['creation_source'],
    labels=df_source['index'],
    shadow=True,
    startangle=0,
    autopct='%1.1f%%',
    radius=2,
    explode=explode,
    colors=colors
    )

# Add title
plt.title('Distribution of creation source',fontsize=20)
plt.axis('equal')

# Display plot
plt.tight_layout()
plt.show()


# In[23]:


sns.set_theme(style='whitegrid',palette='inferno_r')
ax=sns.displot(df_data, x="creation_source", hue="adopted_user", multiple="dodge",shrink=.8)
ax=plt.xticks(rotation=75)


# In[24]:


df_optmail = df_data.opted_in_to_mailing_list.value_counts()
df_source = df_optmail.reset_index()
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
explode = (0.05,0.05)

# Put parameter values
plt.pie(
    df_source['opted_in_to_mailing_list'],
    labels=df_source['index'],
    shadow=True,
    startangle=0,
    autopct='%1.1f%%',
    radius=2,
    explode=explode
    )

# Add title
plt.title('Distribution of people who opted in to mailing list',fontsize=20)
plt.axis('equal')

# Display plot
plt.tight_layout()
plt.show()


# In[25]:


df_adopt = df_data.adopted_user.value_counts()
df_source = df_adopt.reset_index()
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
explode = (0.05,0.05)
labels =['Not adopted','Adopted user']
colors = ['#ff9999','#66b3ff']
# Put parameter values
plt.pie(
    df_source['adopted_user'],
    labels=labels,
    shadow=True,
    startangle=0,
    autopct='%1.1f%%',
    radius=2,
    explode=explode,
    colors=colors
    )

# Add title
plt.title('Adopted user or not',fontsize=20)
plt.axis('equal')

# Display plot
plt.tight_layout()
plt.show()


# In[26]:


df_drip = df_data.enabled_for_marketing_drip.value_counts()
df_drip = df_drip.reset_index()
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
explode = (0.05, 0.05)
colors = ['lightblue','pink']
labels =['Disabled','Enabled']

# Put parameter values
plt.pie(
    df_drip['enabled_for_marketing_drip'],
    labels=labels,
    shadow=True,
    startangle=0,
    autopct='%1.1f%%',
    radius=2,
    explode=explode,
    colors=colors
    )

# Add title
plt.title('Distribution which represents percentage enabled and disabled for marketing drip',fontsize=20)
plt.axis('equal')

# Display plot
plt.tight_layout()
plt.show()


# In[27]:


from datetime import datetime
# convert unix timestamp to datetime
df_data['last_session_creation_time'] = df_data['last_session_creation_time'].apply(
    lambda x: datetime.strptime(str(datetime.fromtimestamp(float(int(x)))), '%Y-%m-%d %H:%M:%S'))
df_data['creation_time'] = df_data['creation_time'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))


# In[28]:


df_data['history'] = df_data['last_session_creation_time'] - df_data['creation_time']
df_data['history'] = df_data['history'].dt.days
latest = max(df_data['creation_time'])
df_data['account_age'] = latest - df_data['creation_time']
df_data['account_age'] = df_data['account_age'].dt.days
df_data = df_data.drop(['last_session_creation_time','creation_time' ], axis=1)
df_data.head()


# In[29]:


data=df_data.drop(['name','email'],axis=1)
data.head()


# In[30]:


from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
data['creation_source']=le.fit_transform(data['creation_source'])


# In[31]:


data.creation_source.unique()


# In[32]:


data.head()


# In[33]:


#finding if feature history is related  to adopted user using t-test

import scipy.stats as stats 
history=data["history"]
adopted=data["adopted_user"]
stats.ttest_ind(a=history,b=adopted,equal_var=True)


# pvalue is less tha 0.05 which means I have enough evidance to reject null hypothysis 
# ie: we can conclude that history and adopted feature is depended 

# In[34]:


#finding dependency btwn creation source and adopted_user using chi-square test
import scipy.stats as st
tab = pd.crosstab(data['creation_source'],data['adopted_user'])
# tab = tab.T
print(tab)
 
'''chi2 : float
The test statistic.
 
p : float
The p-value of the test
 
dof : int
Degrees of freedom'''
 
st.chi2_contingency(tab)


# In[35]:


data.head()


# In[36]:


data.drop("invited_by_user_id",axis=1,inplace=True)


# In[37]:


data.shape


# In[38]:


import missingno 
missingno.matrix(data,figsize=(30,10))


# In[39]:


data.info()


# LogisticRegression

# In[40]:


X=data.iloc[:,[0,1,2,3,5,6]].values
Y=list(data["adopted_user"])


# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3, random_state=3)


# In[42]:


from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
model.fit(x_train,y_train)


# In[43]:


y_pred = model.predict(x_test)


# In[44]:


#this is for referance perpose only ,not mandatory to do for the model
df= pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
df.sample(5)


# In[45]:


from sklearn import metrics 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[46]:


model.coef_


# In[47]:


model.intercept_


# At the start of coding, the adopted_user is defined as a new column.
# Further, the visualisation of certain features with respect to the adopted user is carried out.
# This is shown with the help of matplotlib and seaborn.
# The logistic regression gave the coeficients, which represent the importance.
# of each corresponding feature.

# In[ ]:




