# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:38:51 2020

@author: Frédéric
"""
#first IA-based algorithm for this project, not really interesting
#mainly based on https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
#should allow me to check wether the problem can be solved with AI or not

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd 
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



#import data cf dataset_in_out

dataf= pd.read_csv('dataf.csv')
data1f= pd.read_csv('data1f.csv')
data2f= pd.read_csv('data2f.csv')
data3f= pd.read_csv('data3f.csv')
data4f= pd.read_csv('data4f.csv')


# Split-out validation dataset

array = dataf.values
X = array[:,1:-1]
y = array[:,-1]

# print(array)
# print('============')
# print(X)
# print('============')
# print(y)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

#Spot Check Algorithms

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = []
names = []
for name, model in models:
 	kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
 	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 	results.append(cv_results)
 	names.append(name)
 	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)
# # Evaluate predictions
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))