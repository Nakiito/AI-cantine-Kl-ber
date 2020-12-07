# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:23:03 2020

@author: Frédéric
"""

#this script let me analyse the dataset and visualize its properties
#will not be useful by itself, I'm just curious :)

import pandas as pd 
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

#import data cf dataset_in_out

dataf= pd.read_csv('dataf.csv')
data1f= pd.read_csv('data1f.csv')
data2f= pd.read_csv('data2f.csv')
data3f= pd.read_csv('data3f.csv')
data4f= pd.read_csv('data4f.csv')

print(dataf.describe())

print('________')

print(dataf.groupby('class').size())

print('________')

#based on "https://machinelearningmastery.com/machine-learning-in-python-step-by-step/"

# box and whisker plots
dataf.plot(kind='box', subplots=True, layout=(12,12), sharex=False, sharey=False)
pyplot.show()
# histograms
dataf.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataf)

pyplot.show()
