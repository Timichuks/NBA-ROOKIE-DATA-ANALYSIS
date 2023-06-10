#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# sklearn for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics


# In[2]:


# import file
df=pd.read_csv('nba_rookie_data.csv')
df.head()


# In[3]:


df.shape


# In[4]:


#missing value
df.info()
df.isnull().sum()


# In[5]:


# Data Exploration
# statistical summary


# In[6]:


df.head(10)


# In[7]:


df.tail(10)


# In[8]:


#remove the string column
df[df.columns[1:-1]]


# In[9]:


#copy of the column with empty datasets
new= df[('3 Point Percent')]
new.head()


# In[10]:


#filling the data with the mean
new= new.fillna(new.mean())


# In[11]:


#add mean toempty data set
df[('3 Point Percent')]=new


# In[12]:


#confirm missing column is gone
df.isna().sum()


# In[13]:


df.describe()


# In[14]:


df.corr()


# In[15]:


#grouping and counting the dataset based on the target variable
print(df.groupby('TARGET_5Yrs')['Games Played'].count())


# In[16]:


#defining input and output
x = df.iloc[:, 1].values #input
y = df.iloc[:, -1].values #output


# In[17]:


x = x.reshape(-1, 1)


# In[18]:


scaler = StandardScaler()
x_new = scaler.fit_transform(x)
x1 = pd.DataFrame(x_new, columns =['Games Played'])


# In[19]:


#training and test sets:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, 
random_state=0)


# In[20]:


#reshaping the dataset
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


# In[21]:


#logistic regression
logre = LogisticRegression(max_iter=1000)
logre.fit(x_train, y_train)


# In[22]:


y_pred = logre.predict(x_test)
print('Prediction:', y_pred)


# In[23]:


#accuracy of the model
print('Our Accuracy is %.3f' % logre.score(x_test, y_test))


# In[24]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != logre.predict(x_test)).sum()))


# In[25]:


#visualizing datasets
fig1, ax1= plt.subplots()
ax1.scatter(x_train, y_train, color='blue')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.plot(x_test, logre.predict_proba(x_test)[:,1], color='red')


# In[26]:


#evaluating the model performance
conf = metrics.confusion_matrix(y_test, y_pred)


# In[27]:


#where conf is confusion metrix
conf


# In[28]:


#confusion matrix visualization
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(10,5))
sns.heatmap(conf, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[29]:


#evaluating model performance using classification report
print(classification_report(y_test,y_pred))


# In[30]:


#Training the data set with Guassian Naive Bayes Model
Gnb = GaussianNB()
Gnb.fit(x_train, y_train)


# In[31]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != Gnb.predict(x_test)).sum()))


# In[32]:


#checking the accuracy of the model
print('Our Accuracy is %.3f' % Gnb.score(x_test, y_test))


# In[33]:


y_pred = Gnb.predict(x_test)
print('Prediction:', y_pred)


# In[34]:


#visualizing datasets
fig1, ax1= plt.subplots()
ax1.scatter(x_train, y_train, color='blue')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.plot(x_test,Gnb.predict_proba(x_test)[:,1], color='red')


# In[35]:


#model performance
conf = metrics.confusion_matrix(y_test, y_pred)


# In[36]:


conf


# In[37]:


#confusion matrix visualization
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(10,5))
sns.heatmap(conf, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[38]:


#evaluating model performance using classification report
print(classification_report(y_test,y_pred))


# In[39]:


#Neural Network
#Training the data set with MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", random_state=0, max_iter = 2000)
mlp.fit(x_train, y_train)


# In[40]:


#Testing the model
y_pred = mlp.predict(x_test)
print('Prediction:', y_pred)


# In[41]:


#Accuracy of the model
print('Our Accuracy is %.3f' % mlp.score(x_test, y_test))


# In[42]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != mlp.predict(x_test)).sum()))


# In[43]:


#visualizing datasets
fig1, ax1= plt.subplots()
ax1.scatter(x_train, y_train, color='blue')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.plot(x_test, mlp.predict_proba(x_test)[:,1], color='red')


# In[44]:


#evaluating the model performance
conf = metrics.confusion_matrix(y_test, y_pred)


# In[45]:


#confusion matrix visualization
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(conf, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# In[46]:


#evaluating model performance using classification report
print(classification_report(y_test,y_pred))


# In[47]:


#defining input and output
x = df.iloc[:, [1, 2, 3]].values #input
y = df.iloc[:, -1].values #output


# In[48]:


x


# In[49]:


x.shape


# In[50]:


y


# In[51]:


#training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, 
random_state=0)


# In[52]:


#Using logistic regression
logre = LogisticRegression(max_iter=1000)
logre.fit(x_train, y_train)


# In[53]:


y_pred = logre.predict(x_test)
print('Prediction:', y_pred)


# In[54]:


# accuracy of the model
print('Our Accuracy is %.2f' % logre.score(x_test, y_test))


# In[55]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != logre.predict(x_test)).sum()))


# In[56]:


#evaluating the model performance
conf= confusion_matrix(y_test, y_pred)


# In[57]:


conf #confusion matrix


# In[58]:


#confusion matrix visualization
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(conf, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[59]:


#evaluating model performance using classification report
print(classification_report(y_test,y_pred))


# In[60]:


#Training the data set with Guassian Naive Bayes Model
Gnb = GaussianNB()
Gnb.fit(x_train, y_train)


# In[61]:


y_pred =Gnb.predict(x_test)
print('Prediction:', y_pred)


# In[62]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != Gnb.predict(x_test)).sum()))


# In[63]:


#Accuracy of the model
print('Our Accuracy is %.2f' % Gnb.score(x_test, y_test))


# In[64]:


y_test.shape


# In[65]:


#evaluating the model performance using confusion matrix
conf1  = metrics.confusion_matrix(y_test, y_pred)


# In[66]:


conf1


# In[67]:


#Visualizing the confusion matrix based on actual and predicted
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(conf1, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[68]:


#evaluating the model performance using the classification report
print(classification_report(y_test,y_pred))


# In[69]:


#Neural Network
#Training the data set with MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", random_state=0, max_iter = 2000)
mlp.fit(x_train, y_train)


# In[70]:


#Testing the model
y_pred = mlp.predict(x_test)
print('Prediction:', y_pred)


# In[71]:


#Accuracy of the model
print('Our Accuracy is %.2f' % mlp.score(x_test, y_test))


# In[72]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y_test != mlp.predict(x_test)).sum()))


# In[73]:


#confusion matrix
conf2 = metrics.confusion_matrix(y_test, y_pred)


# In[74]:


conf2 #confusion matrix


# In[75]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(conf2, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[76]:


#classification report
print(classification_report(y_test,y_pred))


# In[77]:


#defining input and output
x1 = df.iloc[:, [1, 2, 3, 4, 5, 6]].values #input
y1 = df.iloc[:, -1].values #output


# In[78]:


x1.shape
x1


# In[79]:


# training and test sets:
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size= 1/3, 
random_state=0)


# In[80]:


#Logistic Regression
logre = LogisticRegression(max_iter=1000)
logre.fit(x1_train, y1_train)


# In[81]:


#predicting the dataset
y1_pred = logre.predict(x1_test)
print('Prediction:', y1_pred)


# In[82]:


#checking the accuracy of the model
print('Our Accuracy is %.3f' % logre.score(x1_test, y1_test))


# In[83]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x1_test.shape[0], (y1_test != logre.predict(x1_test)).sum()))


# In[84]:


#evaluating the model performance
conf3 = metrics.confusion_matrix(y1_test, y1_pred)


# In[85]:


conf3


# In[86]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(10,5))
sns.heatmap(conf3, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[87]:


#evaluating model performance using classification report
print(classification_report(y1_test,y1_pred))


# In[88]:


#Training the data set with Guassian Naive Bayes Model
Gnb = GaussianNB()
Gnb.fit(x1_train, y1_train)


# In[89]:


y_pred = Gnb.predict(x1_test)
print('Prediction:', y1_pred)


# In[90]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x_test.shape[0], (y1_test != Gnb.predict(x1_test)).sum()))


# In[91]:


#checking the accuracy of the model
print('Our Accuracy is %.3f' % Gnb.score(x1_test, y1_test))


# In[92]:


#Neural Network
#Training the data set with MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", random_state=0, max_iter = 2000)
mlp.fit(x1_train, y1_train)


# In[93]:


#Testing the model
y1_pred = mlp.predict(x1_test)
print('Prediction:', y1_pred)


# In[94]:


#Accuracy of the model
print('Our Accuracy is %.2f' % mlp.score(x1_test, y1_test))


# In[95]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x1_test.shape[0], (y1_test != mlp.predict(x1_test)).sum()))


# In[96]:


#confusion matrix
conf4 = metrics.confusion_matrix(y1_test, y1_pred)


# In[97]:


conf4


# In[98]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(10,5))
sns.heatmap(conf4, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[99]:


#classification report
print(classification_report(y_test,y_pred))


# In[100]:


#defining input and output
x4 = df[df.columns[1:-1]].values #input
y3 = df.iloc[:, -1].values #output


# In[101]:


x4


# In[102]:


scaler = StandardScaler()
x_new = scaler.fit_transform(x4)
scaled_x4 = pd.DataFrame(x_new,  columns =['Games Played', 'Minutes Played', 'Points Per Game', 'Field Goals Made', 'Field Goals Attempt', 'Field Goal Percent', '3 Point Made', '3 Point Attempt', '3 Point Percent', 'Free Throw Made', 'Free Throw Attempt', 'Free Throw Percent', 'Offensive Rebounds', 'Defensive Rebounds', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers'])


# In[103]:


x3=scaled_x4


# In[104]:


# training and test sets:
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size= 1/3, 
random_state=0)


# In[105]:


#Using logistic regression
logre = LogisticRegression(max_iter=1000)
logre.fit(x3_train, y3_train)


# In[106]:


#predicting the dataset
y3_pred = logre.predict(x3_test)
print('Prediction:', y3_pred)


# In[107]:


#checking the accuracy of the model
print('Our Accuracy is %.4f' % logre.score(x3_test, y3_test))


# In[108]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x3_test.shape[0], (y3_test != logre.predict(x3_test)).sum()))


# In[109]:


#confusion matrix
conf5 = metrics.confusion_matrix(y3_test, y3_pred)


# In[110]:


conf5


# In[111]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(10,5))
sns.heatmap(conf5, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[112]:


#classification report
print(classification_report(y3_test,y3_pred))


# In[113]:


#Training the data set with Guassian Naive Bayes Model
Gnb = GaussianNB()
Gnb.fit(x3_train, y3_train)


# In[114]:


y_pred = logre.predict(x3_test)
print('Prediction:', y3_pred)


# In[115]:


#checking the accuracy of the model
print('Our Accuracy is %.4f' % Gnb.score(x3_test, y3_test))


# In[116]:


#checking the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x3_test.shape[0], (y3_test != Gnb.predict(x3_test)).sum()))


# In[117]:


#confusion matrix
conf6 = metrics.confusion_matrix(y3_test, y3_pred)


# In[118]:


conf6


# In[119]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(10,5))
sns.heatmap(conf6, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[120]:


#classification report
print(classification_report(y3_test,y3_pred))


# In[121]:


#Neural Network
#Training the data set with MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic", random_state=0, max_iter = 2000)
mlp.fit(x3_train, y3_train)


# In[122]:


#Testing the model
y3_pred = mlp.predict(x3_test)
print('Prediction:', y3_pred)


# In[123]:


#Accuracy of the model
print('Our Accuracy is %.4f' % mlp.score(x3_test, y3_test))


# In[124]:


#number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (x3_test.shape[0], (y3_test != mlp.predict(x3_test)).sum()))


# In[125]:


#confusion matrix
conf7 = metrics.confusion_matrix(y3_test, y3_pred)


# In[126]:


conf7


# In[127]:


#visualizing confusion matrix
ylabel = ["Actual [>=5yrs]","Actual [<=5yrs]"]
xlabel = ["Pred [>=5yrs]","Pred [<=5yrs]"]
#sns.set(font_scale = 1.5)
plt.figure(figsize=(10,5))
sns.heatmap(conf7, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='black', linewidths=1)


# In[128]:


#classification report
print(classification_report(y3_test,y3_pred))


# In[ ]:




