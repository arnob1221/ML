import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('diabetes_risk.csv') #df=data frame




## Any Data_Set Understanding ##

#1. How big is the data?
df.shape

#2. How does the data look?
df.head() #first 5 ta
df,tail() #last 5 ta
df.sample() #jekonon akta randomly

#3. What is the data type of the columns?
df.info() #every column with datatype and memory

#4. Any missing?
df.isnull().sum()
df[df.isnull().any(axis=1)] # row all null
df[df['charges'].isnull()] # to see individual coloumn null

#5. How does the data look mathematically?
df.describe() #standard,min,max,25 & 50%,mean

#5. Are there any duplicate values?
df.duplicated().sum()
df[df.duplicated()] #to see that duplicate row
df[df.duplicated(keep=False)] # to see which one is actually the same

#6. Correlation between columns?
df.corr()["Age"]


##  Data visualization  ##
# 1. sob gulo numeric ak sathe histplot, using loop
numeric_columns = ['age', 'bmi', 'children', 'charges']
for col in numeric_columns:
  plt.figure(figsize=(6,4))
  sns.histplot(df[col],kde = True, bins = 20) #bins choto boro dakha r jonno


#2. Boxplot o sudhumatro numerical data r outliers bojha jabe
for col in numeric_columns:
  plt.figure(figsize=(6,4))
  sns.boxplot(x=df[col])

# 3. dataset e numerical data r visualization dakhar por, baki thaka categorical data dakhar jonno
sns.countplot( x = df["children"] )

# 4. Heatmap diye correlation dakhbo

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)   # (1=perfect positive, 0.9=high pos, 0.5=low pos, 0= no relation; negative also possible)



##  Data cleaning and processing  ##
# df e je data ase ota rekhe dibi, erpor new akta banabo df_cleaning nm e
df_cleaned = df.copy()
