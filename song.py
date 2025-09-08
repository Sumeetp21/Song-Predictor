#  Song Popularity Predictor

# Step 01 : Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 02 : Create Dataset
data = {
    "danceability": [0.6, 0.7, 0.8, 0.4, 0.9, 0.5, 0.65, 0.75, 0.55, 0.85],
    "energy":       [0.7, 0.8, 0.9, 0.5, 0.95, 0.6, 0.68, 0.78, 0.58, 0.88],
    "loudness":     [-5, -4, -3, -8, -2, -7, -6, -4.5, -7.5, -3.5],
    "tempo":        [120, 130, 140, 110, 150, 100, 125, 135, 115, 145],
    "popularity":   [65, 70, 80, 50, 90, 55, 68, 75, 58, 85]
}
df = pd.DataFrame(data)

# Step 03 : EDA = Exploratory Data Analysis
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 04 : Define Feature and Target
X = df[["danceability", "energy", "loudness", "tempo"]]
Y = df["popularity"]

# Step 05 : Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 06 : Train the Model
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 07 : Make Predictions
y_pred = model.predict(X_test)

# Step 08 : Evaluation
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Step 09 : Visualization
plt.scatter(Y_test, y_pred, color="green")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Song Popularity")
plt.show()
