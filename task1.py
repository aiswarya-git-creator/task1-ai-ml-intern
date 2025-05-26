import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("Titanic-Dataset.csv")  
print(df.head())
print(df.info())
print(df.describe())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

sns.boxplot(x=df['Age'],color='green')
plt.title('Boxplot - Age', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12, fontweight='bold') 
plt.text(0.98, 0.98, 
         'Box = Middle 50% of Ages\nDots = Age Outliers\nLine = Median Age',
         transform=plt.gca().transAxes,
         fontsize=8,
         verticalalignment='top',
         horizontalalignment='right',
)
plt.show()
sns.boxplot(x=df['Fare'],color='palegreen')
plt.title("Boxplot - Fare",fontsize=14, fontweight='bold')
plt.xlabel('Fare', fontsize=12, fontweight='bold') 
plt.text(
    0.98, 0.98,  # almost at the top right corner inside axes
    'Box = Typical Fare Range\nDots = High Fare Outliers\nLine = Median Fare',
    transform=plt.gca().transAxes,
    fontsize=8,
    verticalalignment='top',
    horizontalalignment='right',  # Align right edge to 0.98 (right side)
    
)
plt.show()
for col in ['Age', 'Fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Cleaned Data Preview:")
print(df.head())



