(A). To call all functions and libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('diabetes_risk.csv') #df=data frame

##    ##
(B) Any Data_Set Understanding

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


##    ##
(C)  Data visualization
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


##    ##
(D)  Data cleaning and processing
# df e je data ase ota rekhe dibi, erpor new akta banabo df_cleaning nm e
df_cleaned = df.copy()
df_cleaned.head()
df_cleaned.shape
df_cleaned.drop_duplicates(inplace = True)
df_cleaned.shape
df_cleaned.isnull().sum()
df_cleaned.dtypes
df_cleaned["sex"].value_counts() # categorical data dakha
df_cleaned["sex"] = df_cleaned["sex"].map({"male" : 0, "female" : 1}) # label encoding 
# label encoding tokhon e korbo, jokhon gendrer, smoker-yes/no amn type of data thakbe


#encoding kore chaile sei columns er name change kore dite pari
df_cleaned.rename(columns={
 "sex" : "is_female",
  "smoker" : "is_smoker"
},inplace = True)


df_cleaned["region"].value_counts() #categorical but onk value ala  
#ek er odhik type like region type data te one hot encoding use korbo
df_cleaned = pd.get_dummies(df_cleaned,columns =["region"],drop_first = True) #one hot encoding, true/false e ase
df_cleaned = df_cleaned.astype(int) #true/false gulo re numerical e niya
# Now the dataset is cleaned, and all values are numerical


##    ##
(E)  Feature Scaling
# At first, see the columns again
df_cleaned.columns

# Data standardization
from sklearn.preprocessing import StandardScaler
cols = ["age","bmi","children"]
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])


df_cleaned.head() #standard e -3 to +3 ei range e sob convert hoye jabe


##    ##
(F)  Feature Extraction/selection
# Pearson Correlation Calculation

from scipy.stats import pearsonr

# List of features to check against target
selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweignt', 'bmi_category_Obese'
]

#target feature
correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])
correlation_df.sort_values(by='Pearson Correlation', ascending=False)




cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweignt', 'bmi_category_Obese'
]


from scipy.stats import chi2_contingency
import pandas as pd

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame.from_dict(chi2_results).T
chi2_df = chi2_df.sort_values(by="p_value")
chi2_df


final_df = df_cleaned[["age","is_female","bmi","children","is_smoker","charges","region_southeast","bmi_category_Obese"]]

















