# -*- coding: utf-8 -*-
"""Food_Preference.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VvWB5lra__NshVf0N44KwFv5mWDcncso
"""

from google.colab import files
files.upload()

import numpy as np
import pandas as pd

df = pd.read_csv('Food_Preference.csv')

df.columns

df.dtypes

df.drop(['Timestamp', 'Participant_ID'], axis=1, inplace=True)

df['Gender'].value_counts()

df['Gender']= df['Gender'].replace('Male', 1)
df['Gender']= df['Gender'].replace('Female', 0)

df['Food'].value_counts()

df['Food']= df['Food'].replace('Traditional food', 1)
df['Food']= df['Food'].replace('Western Food', 0)

df.describe

df['Dessert'].value_counts()

df.isnull().sum()

df = df.dropna()

df.isnull().sum()

df.reset_index(drop=True, inplace=True)

df.shape

df.head()

df.dtypes

df.to_csv("Food_Preference")
