# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:15:18 2020

@author: Frédéric
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

#permet de voir les tableaux entiers

def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = None
    # display.precision = 2  # set as needed

set_pandas_display_options()

#import les stats

url="https://docs.google.com/spreadsheets/d/e/2PACX-1vQLTW09KN8gL1XwvCP2YgicJBKsjcPaFTPFLGrhV0VYVHRoeNp3-EdSJ3zjNRyIq8CE_xZQ52wuO4me/pub?gid=1708754188&single=true&output=csv"
names = ['personne', 'jour','noteM','class']

#création de l'array

dataset=pd.read_csv(url,names=names)

#séparation par personne
data1 = dataset[dataset['personne']==1]
data2 = dataset[dataset['personne']==2]
data3 = dataset[dataset['personne']==3]

#division des notes en groupe de 3
x1=int(data1.groupby('personne').size())
x2=int(data2.groupby('personne').size())
x3=int(data3.groupby('personne').size())

data4={}
data5={}
data6={}
data4f={}
data5f={}
data6f={}


for i in range (x1):
    data4[i]=data1.iloc[[i]]  
    

for i in range (x2):
    data5[i]=data2.iloc[[i]]

for i in range (x3):
    data6[i]=data3.iloc[[i]]


d={}


for i in range (x1-3):
    d[i]={'ClassF':[data4[i+3].iloc[0,1]]}
    d[i]= pd.DataFrame(d[i])
    data4f[i]= data4[i].merge(data4[i+1], how='right', on='personne').merge(data4[i+2], how='right', on='personne')
    data4f[i]=data4f[i].join(d[i])

d={}


for i in range (x2-3):
    d[i]={'ClassF':[data5[i+3].iloc[0,1]]}
    d[i]= pd.DataFrame(d[i])
    data5f[i]= data5[i].merge(data5[i+1], how='right', on='personne').merge(data5[i+2], how='right', on='personne')
    data5f[i]=data5f[i].join(d[i])

for i in range (x3-3):
    d[i]={'ClassF':[data6[i+3].iloc[0,1]]}
    d[i]= pd.DataFrame(d[i])
    data6f[i]= data6[i].merge(data6[i+1], how='right', on='personne').merge(data6[i+2], how='right', on='personne')
    data6f[i]=data6f[i].join(d[i])
    

# namef=['repas']
# data1f=[]
        
# #création du data set de vérification
# #       Jour Personne NoteR Class Jour Personne NoteR Class  Jour Personne NoteR Class  ClassF
# # Repas 1

# #création du dataset de vérification

# array = data1f.values
# X = array[:,:-1]
# y = array[:,-1]
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# # Spot Check Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.apxpend(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
# 	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
# 	results.append(cv_results)
# 	names.append(name)
# 	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

