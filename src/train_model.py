import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
data_path = "data/processed_world_energy_consumption.csv"
df = pd.read_csv(data_path)

# Define the target variable and selected features
# Da wir den Datensatz nicht weiter manipulieren, wählen wir nur vorhandene, sinnvolle Features.
target = "energy_cons_change_pct"
features = ["year", "gdp_per_capita", "population", "energy_per_capita", "energy_intensity"]
# Falls fossil_share berechnet wurde, könnte man es auch hinzufügen:
if 'fossil_share' in df.columns:
    features.append("fossil_share")

# Visualize the distribution of the target variable before transformation
plt.hist(df[target], bins=30)
plt.title(f'Distribution of {target}')
plt.xlabel(target)
plt.ylabel('Frequency')
plt.show()

# Apply log transformation to stabilize variance
df[target] = df[target].apply(lambda x: np.log(x + 1))  # Adding 1 to handle zero values

# Visualize the transformed target variable distribution
plt.hist(df[target], bins=30)
plt.title(f'Transformed Distribution of {target}')
plt.xlabel(target)
plt.ylabel('Frequency')
plt.show()

# Compute and visualize feature-target correlations
correlations = df[features + [target]].corr()
sns.heatmap(correlations, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Check for missing values in the dataset
print("Missing values per feature:")
print(df.isnull().sum())

# Visualize relationships between each feature and the target variable
for feature in features:
    plt.scatter(df[feature], df[target])
    plt.title(f'{feature} vs {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()

# Prepare the feature matrix (X) and target vector (y)
X = df[features]
y = df[target]

# Standardize the features to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print basic statistics for verification
print(f"Target mean: {y.mean()}, Target std: {y.std()}")
print(f"Feature means: {X.mean()}, Feature std: {X.std()}")

# Split the dataset into training and testing subsets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Define a hyperparameter grid for optimization
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 500],
}

# Execute Grid Search with cross-validation for hyperparameter tuning
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Retrieve the best-performing model
best_model = grid_search.best_estimator_

# Generate predictions on the test dataset
y_pred = best_model.predict(X_test)

# Evaluate model performance using multiple metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot feature importance derived from the trained model
xgb.plot_importance(best_model)
plt.title("Feature Importance")
plt.show()

# Save the trained model for future inference
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "energy_model.pkl")
with open(model_path, "wb") as model_file:
    pickle.dump(best_model, model_file)

print(f"\nTraining complete. Best model saved at: {model_path}")