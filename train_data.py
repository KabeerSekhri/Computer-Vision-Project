import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv('hand_landmarks.csv')

# Ensure X has only 84 features (42 for each hand, for example)
X = data.iloc[:, 0:84]  # Ensure you are using the first 84 columns for the features
Y = data['label']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)

# Save feature names (84 features)
feature_names = [str(i) for i in range(84)]  # Example: feature names 0, 1, 2, ..., 83

# Train the model
model = RandomForestClassifier()  # Example model
model.fit(x_train, y_train)

# Save the model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('Accuracy:', score * 100)

# Optional: Save the model again (redundant here)
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Optional: Plot feature importances
import matplotlib.pyplot as plt
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()
