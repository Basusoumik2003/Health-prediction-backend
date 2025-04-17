import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load sample dataset (Heart Disease UCI)
df = pd.read_csv('./data/heart_disease_dataset.csv')

X = df.drop(columns='target')
y = df['target']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'risk_model.pkl')
print("Model saved as risk_model.pkl")
