import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

data = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')

# Preprocessing
data = data.dropna()
data['ocean_proximity'] = data['ocean_proximity'].astype('category').cat.codes

# Train model
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, random_state=42)  
model.fit(X_train, y_train)

# Save model to API folder
joblib.dump(model, '../api/real_estate_model.pkl')
print(f"Model saved! Test R²: {model.score(X_test, y_test):.2f}")
