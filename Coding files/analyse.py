import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display

df = pd.read_csv("diabetes.csv")
display(df.head())
display(df.describe())
display(df.nunique())
display(df.shape)
corre = df.corr()
display(corre)

fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df['Age'], bins = 100, ax=ax[0,0]) 
sns.distplot(df['Pregnancies'], bins = 100, ax=ax[0,1]) 
sns.distplot(df['Glucose'], bins = 100, ax=ax[1,0]) 
sns.distplot(df['BloodPressure'], bins = 100, ax=ax[1,1]) 
sns.distplot(df['SkinThickness'], bins = 100, ax=ax[2,0])
sns.distplot(df['Insulin'], bins = 100, ax=ax[2,1])
sns.distplot(df['DiabetesPedigreeFunction'], bins = 100, ax=ax[3,0]) 
sns.distplot(df['BMI'], bins = 100, ax=ax[3,1])
plt.show()

(df == 0).sum(axis=0)
df[['BMI','Insulin','SkinThickness','BloodPressure','Glucose']] = df[['BMI','Insulin','SkinThickness','BloodPressure','Glucose']].replace(0,np.nan)
df.isna().sum()

fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df['Age'], bins = 100, ax=ax[0,0]) 
sns.distplot(df['Pregnancies'], bins = 100, ax=ax[0,1]) 
sns.distplot(df['Glucose'], bins = 100, ax=ax[1,0]) 
sns.distplot(df['BloodPressure'], bins = 100, ax=ax[1,1]) 
sns.distplot(df['SkinThickness'], bins = 100, ax=ax[2,0])
sns.distplot(df['Insulin'], bins = 100, ax=ax[2,1])
sns.distplot(df['DiabetesPedigreeFunction'], bins = 100, ax=ax[3,0]) 
sns.distplot(df['BMI'], bins = 100, ax=ax[3,1])
plt.show()

df['BMI'].fillna(int(df['BMI'].mean()), inplace=True)
df['Insulin'].fillna(int(df['Insulin'].mean()), inplace=True)
df['SkinThickness'].fillna(int(df['SkinThickness'].mean()), inplace=True)
df['BloodPressure'].fillna(int(df['BloodPressure'].mean()), inplace=True)
df['Glucose'].fillna(int(df['Glucose'].mean()), inplace=True)
display(df.isnull().sum())

corre2 = df.corr()
display(corre2)

fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df['Age'], bins = 100, ax=ax[0,0]) 
sns.distplot(df['Pregnancies'], bins = 100, ax=ax[0,1]) 
sns.distplot(df['Glucose'], bins = 100, ax=ax[1,0]) 
sns.distplot(df['BloodPressure'], bins = 100, ax=ax[1,1]) 
sns.distplot(df['SkinThickness'], bins = 100, ax=ax[2,0])
sns.distplot(df['Insulin'], bins = 100, ax=ax[2,1])
sns.distplot(df['DiabetesPedigreeFunction'], bins = 100, ax=ax[3,0]) 
sns.distplot(df['BMI'], bins = 100, ax=ax[3,1])
plt.show()

sns.countplot(x = 'Outcome', data = df)
fig = plt.subplots(figsize=(16,8))
sns.countplot(x=df['Age'],hue=df['Outcome'])
plt.show()

fig = plt.subplots(figsize=(12,8))
sns.scatterplot(x=df['Insulin'], y=df['Glucose'],hue=df['Outcome'],color="#4CB391")
plt.show()