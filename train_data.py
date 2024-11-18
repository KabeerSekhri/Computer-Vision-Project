import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('hand_landmarks.csv')

X = data.iloc[:, 0:42] 
Y = data.iloc[:, 42]   

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(score*100)

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
