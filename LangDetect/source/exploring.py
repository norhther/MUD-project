import pandas as pd
from sklearn.model_selection import train_test_split

# Read data
df = pd.read_csv('../data/dataset.csv')

#count the language occurences
X = df['Text']
y = df['language']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.value_counts())
print(y_test.value_counts())