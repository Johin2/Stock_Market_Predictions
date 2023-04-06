import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-darkgrid")

df = pd.read_csv("Reliance.csv")

df.index = pd.to_datetime(df['Date'])
print(df)

df['Open-Close'] = df['Open'] - df['Close']
df['High-Low'] = df['High'] - df['Low']

x = df[['Open-Close', 'High-Low']]
print(x.head())

y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
print(y)

split_percentage = 0.8
split = int(split_percentage * len(df))

# Train Dataset
x_train = x[:split]
y_train = y[:split]

x_test = x[split:]
y_test = y[split:]

# Support Vector Classifier
cls = SVC().fit(x_train, y_train)

df['Predicted_signal'] = cls.predict(x)
df['Daily_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Daily_Return'] * df['Predicted_signal'].shift(1)
df['Cum_Ret'] = df['Daily_Return'].cumsum()
print(df)

df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
print(df)

plt.plot(df['Cum_Ret'], color='Red')
plt.plot(df['Cum_Strategy'], color='Blue')
plt.show()

# Calculate the accuracy score
y_pred = cls.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
