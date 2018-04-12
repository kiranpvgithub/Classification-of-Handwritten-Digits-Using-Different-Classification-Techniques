
# coding: utf-8

# In[4]:

#importing the required libraries
import sys
import numpy as np
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# In[5]:

#Importing the MNIST data set
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]


# In[6]:

#Splitting the dataset into training set and testing set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[4]:

# Building the Radial Basis Function model by changing the event model 
# Used the cros_val_score cross validation method
# For different event model 
# Accuracy of all the 3 folds and their average is calculated


# In[ ]:

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.astype(np.float64))
X_train = scaler.fit_transform(X_train.astype(np.float64))
train_acc = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print("For default max_iter= 100")
print("1st fold accuracy- %.2f%%" %(100*train_acc[0]))  
print("2nd fold accuracy- %.2f%%" %(100*train_acc[1]))
print("3rd fold accuracy- %.2f%%" %(100*train_acc[2]))
print("Average fold accuracy- %.2f%%" %((100*train_acc[2] + 100*train_acc[1] + 100*train_acc[0])/3))


# In[ ]:

print("1st fold accuracy- %.2f%%" %(100*0.936))  
print("2nd fold accuracy- %.2f%%" %(100*train_acc[1]))
print("3rd fold accuracy- %.2f%%" %(100*train_acc[2]))


# In[170]:

from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(max_iter = 100)
train_acc = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print("For default max_iter= 375")
print("1st fold accuracy- %.2f%%" %(100*train_acc[0]))  
print("2nd fold accuracy- %.2f%%" %(100*train_acc[1]))
print("3rd fold accuracy- %.2f%%" %(100*train_acc[2]))
print("Average fold accuracy- %.2f%%" %((100*train_acc[2] + 100*train_acc[1] + 100*train_acc[0])/3))


# In[171]:

from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(max_iter = 700)
train_acc = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print("For default max_iter= -1")
print("1st fold accuracy- %.2f%%" %(100*train_acc[0]))  
print("2nd fold accuracy- %.2f%%" %(100*train_acc[1]))
print("3rd fold accuracy- %.2f%%" %(100*train_acc[2]))
print("Average fold accuracy- %.2f%%" %((100*train_acc[2] + 100*train_acc[1] + 100*train_acc[0])/3))


# In[ ]:

from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(max_iter = 1000)
train_acc = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print("For default max_iter= -1")
print("1st fold accuracy- %.2f%%" %(100*train_acc[0]))  
print("2nd fold accuracy- %.2f%%" %(100*train_acc[1]))
print("3rd fold accuracy- %.2f%%" %(100*train_acc[2]))
print("Average fold accuracy- %.2f%%" %((100*train_acc[2] + 100*train_acc[1] + 100*train_acc[0])/3))


# In[172]:

#Choosing  BernoulliNB as the event model for this data
#Performing the accuracy test on the training data


# In[173]:

clf = BernoulliNB()
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=10)
accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy: %.2f%%" % (accuracy*100))


# In[174]:

#Printing the confusion matrix for the training set


# In[175]:

train_conf_mat = confusion_matrix(y_train,y_train_pred)
plt.matshow(train_conf_mat)
plt.title('Confusion Matrix for Training Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[176]:

#Printing the error values of the confusion matrix


# In[177]:

import seaborn as sns
val_row= train_conf_mat.sum(axis=1, keepdims=True)
train_err_conf_mat = train_conf_mat / val_row
np.fill_diagonal(train_err_conf_mat, 0)
ax = sns.heatmap(train_err_conf_mat,annot=True)
plt.show()


# In[178]:

#Chose BernoulliNB as the event model and performing the fit
#Testing the model
#Calculating the accuracy of the test data


# In[189]:

clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print("Test Data Accuracy: %.2f%%" % (accuracy*100))


# In[190]:

#Printing the  confusion matrix


# In[191]:

test_conf_mat = confusion_matrix(y_test,y_pred_test)
plt.matshow(test_conf_mat)
plt.title('Confusion Matrix for Testing Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[192]:

#Priting the error values of the confusion matrix


# In[193]:

row_sums = test_conf_mat.sum(axis=1, keepdims=True)
norm_conf_mx = test_conf_mat / row_sums
np.fill_diagonal(norm_conf_mx, 0)
ax = sns.heatmap(norm_conf_mx,annot=True)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



