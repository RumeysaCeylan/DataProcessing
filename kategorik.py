# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:14:33 2021

@author: Rumeysa
"""

#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri onişleeme
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

ulke=veriler.iloc[:,0:1].values
print(ulke)


#kategorik verilerin numerik çevrilmesi
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

ulke[:,0] =le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe =preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)


#verilerin birleştirilmesi
sonuc =pd.DataFrame(data=ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Age, index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

gender=veriler.iloc[:,-1].values
print(gender)

sonuc3=pd.DataFrame(data=gender, index=range(22),columns=['cinsiyet'])
print(sonuc3)

#dataframe birleştirme işlemi
s=pd.concat([sonuc,sonuc2],axis=1) #axis=1 bire bir eşleme yapar ilk satırla ilk satırı
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#cinsiyet tahmini

from sklearn.model_selection import train_test_split
#verileri x ile yandan y ile sütunlardan bölüyoruz 
#yüzde 33 ünü test için ayırıyoruz
#doğruluğunu test edeceğiz
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,
                                               random_state=0)
#sayısal verileri ölçeklendirmek
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train =sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

















