# -*- coding: utf-8 -*-
"""fetal_health.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11yWQ8ARdkNbo6kdymaz4Rjf60k08dclz
"""

from google.colab import files
files.upload()

import numpy as np 
import pandas as pd

df = pd.read_csv('fetal_health.csv')

df.shape

df.head(10)

df.info()

df.columns

#Checking for duplicated values in dataset
df.duplicated().sum()
#there are no duplicate values

df = df.drop_duplicates()

#Checking for null values in dataset
df.isnull().sum()
#there are no null records

df['severe_decelerations'].value_counts()

df.isna().sum()

