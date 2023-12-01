import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from panda import models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('winequalityN.csv')

for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

df.isnull().sum().sum()



df = df.drop('total sulfur dioxide', axis=1)

df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

df.replace({'white': 1, 'red': 0}, inplace=True)

features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)

xtrain.shape, xtest.shape

metrics.plot_confusion_matrix(models[1], xtest, ytest)
plt.show()
