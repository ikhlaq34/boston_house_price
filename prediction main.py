# ==========================================
# Boston House Price Prediction (Professional)
# Multivariable Linear Regression
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset (Excel)
# -----------------------------
DATA_PATH = "C:\\Users\\akhal\\OneDrive\\Desktop\\Advanced House price prediction\\boston dataset.xlsx"
df = pd.read_excel(DATA_PATH)

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
# Fill missing numerical features with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# -----------------------------
# 3. Separate Features and Target
# -----------------------------
TARGET_COLUMN = "MEDV"  # House price
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Create and Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. Predictions on Test Set
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. Model Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("========== Model Evaluation ==========")
print("RMSE:", rmse)
print("R² Score:", r2)

# -----------------------------
# 8. Feature Importance (Coefficients)
# -----------------------------
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\n========== Feature Coefficients ==========")
print(coeff_df)

# -----------------------------
# 9. Predict New House (Professional Way)
# -----------------------------
# Example input (same order as X.columns)
new_house_data = [[
    0.02, 0, 7.0, 0, 0.469, 6.5, 65, 4.1, 4, 300, 18, 390, 5.0
]]

new_house_df = pd.DataFrame(new_house_data, columns=X.columns)
predicted_price = model.predict(new_house_df)

print("\nPredicted House Price (New Input):", predicted_price[0])

plt.figure(figsize=(10,6))
sns.barplot(x="Coefficient", y="Feature", data=coeff_df)
plt.title("Feature Importance")
plt.show()