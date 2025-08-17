import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Generate synthetic dataset
np.random.seed(42)
n_samples = 200

sqft = np.random.randint(500, 4000, n_samples)
bedrooms = np.random.randint(1, 5, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 30, n_samples)

# Fake price formula (for demo)
price = (sqft * 200) + (bedrooms * 10000) + (bathrooms * 5000) - (age * 500) + np.random.randint(10000, 50000, n_samples)

df = pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "price": price
})

X = df[["sqft", "bedrooms", "bathrooms", "age"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Save model
bundle = {
    "model": model,
    "features": list(X.columns),
    "metrics": {"rmse": rmse, "r2": r2}
}
joblib.dump(bundle, "house_model.pkl")
print("Model saved to house_model.pkl")
