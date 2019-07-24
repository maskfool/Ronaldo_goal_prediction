#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[18]:


df=pd.read_csv("data.csv")
df.head()


# In[20]:


df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


# In[21]:


df.head()


# In[22]:


df.drop(['match_event_id', 'game_season','team_name','date_of_game','shot_id_number','match_id','team_id','area_of_shot','range_of_shot'], axis=1, inplace=True)
df


# In[23]:


new = df["lat/lng"].str.split(",", n = 1, expand = True) 
df["lat"]= new[0]
df["lng"]= new[1]
df.drop(columns =["lat/lng"], inplace = True) 
df


# In[24]:


nulls = df.isnull().sum()
nulls[nulls>0]


# In[25]:


df=df.fillna({
    'location_x': 0.0,
    'location_y':0.0,
    'remaining_min':0.0,
    'power_of_shot':3.0,
    'knockout_match':0.0,
    'remaining_sec':0.0,
    'distance_of_shot':20.0,
    'is_goal':0.0,
    #'area_of_shot':'Center(C)',
    'shot_basics':'Mid Range',
    #'range_of_shot':'Less Than 8 ft.',
    'home/away':'MANU @ SAS',
    'lat':42.982923,
    'lng':-71.446094,
    'type_of_shot':'shot - 39',
    'type_of_combined_shot':'shot - 3',
    'remaining_min.1':0.0000,
    'knockout_match.1':0.00000,
    'remaining_sec.1':0.0000,
    'distance_of_shot.1':20.000,
    'power_of_shot.1':3.00 
})


# In[26]:


X_all = df.drop(['is_goal'],1)
Y_all = df['is_goal']


# In[27]:


X_all.rename(columns = {'home/away':'home_away'},inplace=True)
X_all


# In[30]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()

#X_all.area_of_shot=le.fit_transform(X_all.area_of_shot)
X_all.shot_basics=le.fit_transform(X_all.shot_basics)
#X_all.range_of_shot=le.fit_transform(X_all.range_of_shot)
X_all.home_away=le.fit_transform(X_all.home_away)
#X_all.area_of_shot=le.fit_transform(X_all.area_of_shot)
X_all.type_of_shot=le.fit_transform(X_all.type_of_shot)
X_all.type_of_combined_shot=le.fit_transform(X_all.type_of_combined_shot)
X_all


# In[31]:


X_all['lat'] = X_all.lat.astype(float)
X_all['lng'] = X_all.lng.astype(float)


# In[33]:


X_all['shot_id_number'] = range(1, 1+len(df))
X_all


# In[34]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_all,Y_all,test_size=0.2,random_state=0)


# In[55]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, Y_train)


# In[56]:


model.score(X_test,Y_test)


# In[57]:


predictions=model.predict(X_test)
predictions


# In[58]:


submission = pd.DataFrame({'shot_id_number':X_test['shot_id_number'],'is_goal':predictions})


# In[59]:


submission.head()


# In[60]:


filename = 'submission_file.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)


# In[ ]:




