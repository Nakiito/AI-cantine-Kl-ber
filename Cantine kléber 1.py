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

#import les stats

url="https://docs.google.com/spreadsheets/d/e/2PACX-1vQLTW09KN8gL1XwvCP2YgicJBKsjcPaFTPFLGrhV0VYVHRoeNp3-EdSJ3zjNRyIq8CE_xZQ52wuO4me/pub?gid=1708754188&single=true&output=csv"
names = ['personne', 'jour','noteM','class']

#création de l'array

dataset=pd.read_csv(url,names=names)
print(dataset.shape)

#séparation par personne à terme, à faire pour n personne
data1 = dataset[dataset['personne']==1]
print(data1.shape)
data2 = dataset[dataset['personne']==2]
data3 = dataset[dataset['personne']==3]

#division des notes en groupe de 3 => idem 
x1=int(data1.groupby('personne').size())
x2=int(data2.groupby('personne').size())
x3=int(data3.groupby('personne').size())

# à optimiser 

data4={}
data5={}
data6={}

#avec iloc[:i+3] => groupe de 3 directement, mais il faut changer x1 en x1-3

for i in range (x1):
    data4[i]=data1.iloc[i]  
    print(data4[i])
    
for i in range (x2):
    data5[i]=data2.iloc[i]

for i in range (x3):
    data6[i]=data3.iloc[i]

# namef=['repas']
# data1f=[]
        
# #création du data set de vérification --> ce qu'on veut avoir:
# #       Jour Personne NoteR Class Jour Personne NoteR Class  Jour Personne NoteR Class  ClassF
# # gp1
# # gp2
# # ...


# # Class F étant la class du repas (n+1) ie: pour le gp1 , class F = 1ère class du repas gp2 (= class data4[i+1])
# # Il faut donc passer d'un tableau en collonne à un tableau composé d'une unique ligne, et ensuite créer une nouvelle matrice qui regroupe 3 de ces tableaux

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

