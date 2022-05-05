import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("Iris.csv")
df.head(3)
df = df.drop(columns = ['Id'])
df.head(5)
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.sample(5)
df[df.Species==1].head()
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['SepalLengthCm'], df0['SepalWidthCm'],color="green",marker='+')
plt.scatter(df1['SepalLengthCm'], df1['SepalWidthCm'],color="blue",marker='.')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['PetalLengthCm'], df0['PetalWidthCm'],color="green",marker='+')
plt.scatter(df1['PetalLengthCm'], df1['PetalWidthCm'],color="blue",marker='.')
from sklearn.model_selection import train_test_split
X = df.drop(columns = ['Species'])
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
random_state=42)
len(X_train)
len(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
knn.predict([[4.8,3.0,1.5,0.3]])
le.classes_
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
