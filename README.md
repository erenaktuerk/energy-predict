# Energy Consumption Prediction

This project aims to predict energy consumption based on the time of day and temperature using machine learning techniques. The model is trained on historical energy consumption data and predicts based on two main features: the hour of the day and temperature.

## Table of Contents

1. [Project Description](#project-description)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Directory Structure](#directory-structure)
6. [License](#license)

## Project Description

The goal of this project is to develop a machine learning model that predicts energy consumption based on the time of day and temperature. The model is trained on a dataset containing historical energy usage data, which is used to predict future consumption. The model is stored in a pickle file for easy deployment.

The code structure separates data processing, model training, and prediction, following best practices for machine learning workflows.

## Features

- *Machine Learning Model*: A regression model trained to predict energy consumption based on time of day and temperature.
- *Data Processing*: Includes preprocessing steps to clean and prepare the dataset.
- *Model Saving*: The trained model is saved as a .pkl file for easy reuse.
- *Energy Prediction*: The model can be used to make predictions based on user input.

## Installation

To set up this project locally, follow these steps:

### Prerequisites

- Python 3.6+
- pip (Python package installer)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/erenaktuerk/energy-consumption-prediction.git
   cd energy-consumption-prediction

	2.	Create and activate a virtual environment:

python -m venv energy_predict_env
source energy_predict_env/bin/activate  # On Windows, use energy_predict_env\Scripts\activate


	3.	Install the required dependencies:

pip install -r requirements.txt


	4.	Ensure the dataset (energy_consumption.csv) is placed in the data folder.

Usage

Training the Model

To train the model, run the following command:

python train_model.py

This will train the model on the dataset and save it as energy_model.pkl in the model/ directory.

Making Predictions

Once the model is trained, use the following code to make predictions:

import pandas as pd
from app import predict_energy_consumption

# Example input data
data = {
    'hour': [12, 14, 18],  # Example hours of the day
    'temperature': [25, 22, 19]  # Example temperatures in °C
}

# Convert the data to a DataFrame
input_data = pd.DataFrame(data)

# Predict energy consumption
predictions = predict_energy_consumption(input_data)

# Print the predictions
print(predictions)

This code imports the prediction function, prepares the input data, and outputs energy consumption predictions.

Directory Structure

Here’s a breakdown of the project’s folder structure:

energy-consumption-prediction/
│
├── app/
│   ├── _init_.py
│   ├── energy_analysis.ipynb
│   └── predict_energy.py
│
├── data/
│   └── energy_consumption.csv
│
├── model/
│   └── energy_model.pkl
│
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md

	•	app/: Contains the logic for making predictions and the Jupyter notebook for analysis.
	•	data/: Contains the dataset (energy_consumption.csv).
	•	model/: Stores the trained model (energy_model.pkl).
	•	train_model.py: Script for training the model.
	•	requirements.txt: List of dependencies required for the project.
	•	.gitignore: Git ignore file to exclude unnecessary files from version control.

License

This project is licensed under the MIT License - see the LICENSE file for details.