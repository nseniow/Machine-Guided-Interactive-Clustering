import pandas as pd
import numpy as np
from sklearn import preprocessing

# https://www.journaldev.com/45109/normalize-data-in-python

df = pd.read_csv('datasets/Casestudy_copy_columns_removed_cleaned.csv')
scaler = preprocessing.MinMaxScaler()
names = df.columns.tolist()
names.remove(' Label')
print(names)
df[names] = scaler.fit_transform(df[names])
print(df.head(10))
df.to_csv('datasets/Casestudy_cleaned_normalized.csv', index=False)

