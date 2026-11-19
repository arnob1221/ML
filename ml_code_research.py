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

#5. How does the data look mathematically?
df.describe() #standard,min,max,25 & 50%,mean

#5. Are there any duplicate values?
df.duplicated().sum()
df[df.duplicated()] #to see that duplicate row
df[df.duplicated(keep=False)] # to see which one is actually the same

#6. Correlation between columns?
df.corr()["Age"]
