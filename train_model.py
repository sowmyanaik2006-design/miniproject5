import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Create sample house data
data = pd.DataFrame({
    "Area": [800, 1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 1100, 1600,
             2100, 2400, 2700],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 4, 5,
                 2, 3, 4, 4, 5],
    "Age": [10, 8, 5, 7, 4, 3, 6, 2, 5, 1,
            12, 9, 4, 3, 2],
    "Price": [25, 30, 40, 50, 60, 70, 72, 85, 90,
              100, 28, 48, 75, 82, 95]
})

# Input features and target
X = data[["Area", "Bedrooms", "Age"]]
y = data["Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Model training completed!")
print("Mean Absolute Error:", round(
    mean_absolute_error(y_test, predictions), 2
))
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# Save the trained model
with open("house_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as house_model.pkl")