import pandas as pd 
import numpy as np

# Load the dataset
df = pd.read_csv('diabetes_risk.csv') #df=data frame




## Any Data_Set Understanding ##
#1. How big is the data?
df.shape

#2.How does the data look like?
df.head() #first 5 ta
df,tail() #last 5 ta
df.sample() #jekonon akta randomly

#3.What is the data type of columns?
df.info() #every columns with datatype and memery

#4. Any missing?
df.isnull().sum()

#5. How does the data look like mathematically?
df.describe() #standard,min,max,25 & 50%,mean

#5. Are there any dupicate values?
df.duplicated().sum()

#6. Correlation between coloums?
df.corr()["Age"]

