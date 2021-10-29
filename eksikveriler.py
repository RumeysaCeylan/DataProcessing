# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:14:33 2021

@author: Rumeysa
"""

#kutuphaneler
import pandas as pd
import numpy as np


#kodlar

#veri yükleme
veriler=pd.read_csv('eksikveriler.csv')
print(veriler)
#veri işleme

boy=veriler[['boy']]
print(boy)

boy_Kilo=veriler[['boy','kilo']]
print(boy_Kilo)

#eksik veriler
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

Age=veriler.iloc[:,1:4].values
print(Age)
imputer =imputer.fit(Age[:,1:4])
Age[:,1:4]=imputer.transform(Age[:,1:4])
print(Age)
