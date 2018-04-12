
# coding: utf-8

# In[5]:

#importing the required libraries
import sys
import numpy as np
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# In[6]:

#Importing the MNIST data set
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]


# In[20]:

#Splitting the dataset into training set and testing set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[8]:

# Building the One Hidden Layer Neural Network model by changing the no of hidden layer neurons 
# Used the cros_val_score cross validation method
# For different values of hidden layer neurons 
# Accuracy of all the 3 folds and their average is calculated


# In[9]:

for i in [75, 100, 125]:
    from sklearn.model_selection import cross_val_score
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(i, ), random_state=1)
    train_acc = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
    print("Hidden Layer Nodes: " +str(i))
    print("1st fold accuracy- %.2f%%" %(100*train_acc[0]))  
    print("2nd fold accuracy- %.2f%%" %(100*train_acc[1]))
    print("3rd fold accuracy- %.2f%%" %(100*train_acc[2]))
    print("Average fold accuracy- %.2f%%" %((100*train_acc[2] + 100*train_acc[1] + 100*train_acc[0])/3))


# In[10]:

#Choosing  as the no of neurons in the hidden layer
#Performing the accuracy test on the training data


# In[23]:

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(125, ), random_state=42)
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=5)
accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy: %.2f%%" % (accuracy*100))


# In[12]:

#Printing the confusion matrix for the training set


# In[27]:

train_conf_mat = confusion_matrix(y_train,y_train_pred)
plt.matshow(train_conf_mat)
plt.title('Confusion Matrix for Training Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[28]:

#Priting the error values of the confusion matrix


# In[29]:

import seaborn as sns
val_row= train_conf_mat.sum(axis=1, keepdims=True)
train_err_conf_mat = train_conf_mat / val_row
np.fill_diagonal(train_err_conf_mat, 0)
ax = sns.heatmap(train_err_conf_mat,annot=True)
plt.show()


# In[19]:

#Chose no of neurons in the hidden layer and performing the fit
#Testing the model
#Calculating the accuracy of the test data


# In[21]:

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(125, ), random_state=1)
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print("Test Data Accuracy: %.2f%%" % (accuracy*100))


# In[ ]:

#Printing the  confusion matrix


# In[24]:

test_conf_mat = confusion_matrix(y_test,y_pred_test)
plt.matshow(test_conf_mat)
plt.title('Confusion Matrix for Testing Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[25]:

#Priting the error values of the confusion matrix


# In[26]:

row_sums = test_conf_mat.sum(axis=1, keepdims=True)
norm_conf_mx = test_conf_mat / row_sums
np.fill_diagonal(norm_conf_mx, 0)
ax = sns.heatmap(norm_conf_mx,annot=True)
plt.show()


# In[ ]:



