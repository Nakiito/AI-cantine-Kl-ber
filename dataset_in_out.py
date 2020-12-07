# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:15:18 2020

@author: Frédéric
"""

import pandas as pd



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

#pris en compte individuel du repas

for i in range (x1-3):
    d[i]={'ClassF':[data4[i+3].iloc[0,1]]}
    d[i]= pd.DataFrame(d[i])
    data4f[i]= data4[i].merge(data4[i+1], how='right', on='personne').merge(data4[i+2], how='right', on='personne')
    data4f[i]=data4f[i].join(d[i])

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

#création des listes finales

data1f=data4f[1]
data2f=data5f[1]
data3f=data6f[1]

for i in range (1,x1-3):
    data1f=data1f.append(data4f[i])
for i in range (1,x2-3):
    data2f=data2f.append(data5f[i])
for i in range (1,x3-3):
    data3f=data3f.append(data6f[i])

#liste "totale"

dataf=data1f.append(data2f.append(data3f))

#création des fichiers csv

dataf.to_csv('dataf.csv')
data1f.to_csv('data1f.csv')
data2f.to_csv('data2f.csv')
data3f.to_csv('data3f.csv')