import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset into DataFrame
df = pd.read_csv('datasets/diabetes.csv')

df
df.head()
# Check for missing values
print(df.isnull().sum())

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

df.hist(bins=50, figsize=(20,15))
plt.show()

from LogisticRegression import LogisticRegression

print(df.columns)

X = df.drop(columns='Outcome')
y = df['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(lr=0.0001) #change lr
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)

0.6558441558441559
