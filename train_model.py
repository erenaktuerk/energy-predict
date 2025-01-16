import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv('data/energy_consumption.csv')

# print the column names to verify
print("Columns in the dataset:", data.columns)

# Prepare features (X) and target (y)
X = data[['hour', 'temperature']]  # Features: hour of the day, temperature
y = data['energyConsumption']     # Target: energy consumption

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model to a file using pickle
with open('model/energy_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model training complete and saved as 'energy_model.pkl'.")