import pandas as pd
df = pd.read_csv("iris.csv")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['class'] = le.fit_transform(df['class'])

from sklearn.model_selection import train_test_split

train,test = train_test_split(df, test_size = 0.1)
result_column = 'class'
X_train = train.drop(columns = [result_column]).values
y_train = train[result_column].values
X_test = test.drop(columns = [result_column]).values
y_test = test[result_column].values

from RandomForest import RandomForest
model = RandomForest(num_trees=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average = 'macro')
precision = precision_score(y_test, y_pred, average = 'macro')
recall = recall_score(y_test, y_pred, average = 'macro')
print(acc, f1, precision, recall)