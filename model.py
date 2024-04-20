import numpy as np 
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt  
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay , classification_report ,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("/kaggle/input/credit-card-fraud-detection-dataset-2023/creditcard_2023.csv")
df.head()

df.shape

df.info()

df.isna().sum()

df.describe()

#corr = df.drop(columns=['Class']).corr()
#sns.heatmap(corr);
plt.style.use("seaborn")

plt.rcParams['figure.figsize']= (22,11)

plt.title("Correlation Heatmap",fontsize=18, weight= 'bold')

sns.heatmap(df.corr(), cmap="BuPu", annot=True)

plt.show()

df['Class'].value_counts(normalize= True).plot(kind= 'bar')
plt.xlabel("Class Distribution")
plt.ylabel("Frequancy")
plt.title("Class balance");

x= df.drop(['id', 'Class'], axis= 1)
y= df['Class']

stn_scaler = StandardScaler()

x_scaled = stn_scaler.fit_transform(x)

X = pd.DataFrame(x_scaled,columns=x.columns)

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

clf.fit(X_train , y_train)

clf.predict(X_train)

acc_train = clf.score(X_train , y_train)
acc_test = clf.score(X_test , y_test)

print(f"Training accuracy: {round(acc_train , 4)}")
print(f"test accuracy: {round(acc_test , 4)}")

ConfusionMatrixDisplay.from_estimator(

    clf,
    X_test,
    y_test

);

print(classification_report(

    y_test,
    clf.predict(X_test)

))

features = X_test.columns
importances = clf.coef_[0]

feat_imp = pd.Series(importances , index=features).sort_values()
feat_imp.tail().plot(kind= 'barh')
plt.xlabel("Scale Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");
