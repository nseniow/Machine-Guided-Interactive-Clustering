import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('use_inf_as_na', True)

#import csv
df = pd.read_csv('datasets/Casestudy_copy_columns_removed.csv')
#print(df.columns)

# for i,v in df.iterrows():
#     #if v['Flow Bytes/s'] == 'Infinity':
#     if i == 135:
#         print(v.to_dict()['Flow Bytes/s'])

#count how many times inf shows up in the dataframe
#print(len(df[df['Flow Bytes/s'] == np.nan]))

print(df['Flow Bytes/s'].isnull().sum())
print(df[' Flow Packets/s'].isnull().sum())

#remove all rows from the data frame that contain null values
df = df.dropna()

print(df['Flow Bytes/s'].isnull().sum())
print(df[' Flow Packets/s'].isnull().sum())

#save df to a csv
df.to_csv('datasets/Casestudy_copy_columns_removed_cleaned.csv', index=False)