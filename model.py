import numpy as np 
import pandas as pd
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("creditcard_2023.csv")
df.head()

df.shape

df.info()

df.isna().sum()

df.describe()


if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

x = df.drop(['Class'], axis=1)
y = df['Class']

stn_scaler = StandardScaler()

x_scaled = stn_scaler.fit_transform(x)

X = pd.DataFrame(x_scaled, columns=x.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

acc_baseline = y_train.value_counts(normalize=True).max()

print("Baseline Accuracy:", round(acc_baseline, 4))

clf = LogisticRegression()
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

clf.fit(X_train, y_train)

clf.predict(X_train)

acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)

print(f"Training accuracy: {round(acc_train, 4)}")
print(f"Test accuracy: {round(acc_test, 4)}")


with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)


