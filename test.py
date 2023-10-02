import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('saveecobot_22622.csv')
df['logged_at'] = pd.to_datetime(df['logged_at'])
#df.head(15)

df = df.loc[:, df.notnull().any(axis=0)]
#df.head(15)

df.drop(df[df['value'] == 0].index, inplace = True)
print(df.head(15))

X = df.loc[df['phenomenon'] == 'pm25', 'logged_at'].values.reshape(-1, 1)
y = df.loc[df['phenomenon'] == 'pm25', 'value'].values

#X = np.array(df['phenomenon'] == 'pm25', 'logged_at').reshape(-1, 1)
#y = np.array(df['phenomenon'] == 'pm25', 'value').reshape(-1, 1)

# df.dropna(inplace=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# regr = LinearRegression()
# regr.fit(X_train, y_train)
# print(regr.score(X_test, y_test))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print(model.intercept_)
print(model.coef_)