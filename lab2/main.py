import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('heart_data.csv')

data.replace('?', np.nan, inplace=True)

data = data.apply(pd.to_numeric, errors='coerce')

data.fillna(data.mean(), inplace=True)

X = data.iloc[:, :-1]
y = data['goal']

results = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    for depth in [2, 3, 5, 7, 10, 15]:
        for min_samples_leaf in [2, 4, 6, 10, 13]:
            forest = RandomForestClassifier(max_depth=depth, random_state=i, n_estimators=min_samples_leaf)
            forest.fit(X_train, y_train)
            
            forest.score(X_test, y_test)
            results.append({'iteration': i+1, 'depth': depth, 'min_samples_leaf': min_samples_leaf, 'accuracy': forest.score(X_test, y_test)*100})

results_df = pd.DataFrame(results)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
best = results_df['accuracy']
print(results_df)
print(f"The best {results[best.argmax()]}")