﻿Energy Consumption Prediction — Advanced, Professional, and Problem-Solving Oriented

This project aims to predict energy consumption based on economic and demographic indicators using advanced machine learning techniques. I’ve evolved it into a more professional and practical solution by incorporating more sophisticated features and a modular, production-ready architecture. The project structure now follows industry best practices and ensures a clear separation of concerns across data processing, model training, and API deployment.

Table of Contents
	1.	Project Overview
	2.	Key Features
	3.	Data Manipulation and Feature Engineering
	4.	Installation
	5.	Usage
	6.	API Endpoints
	7.	Directory Structure
	8.	License

Project Overview

This project is designed to predict energy consumption per capita using a combination of historical data and enriched features like:
	•	Yearly data
	•	GDP per capita
	•	Population
	•	Energy consumption per capita

I’ve moved far beyond simple temperature-based predictions and now leverage a more comprehensive and impactful set of features to better model real-world energy consumption behavior. The model is trained using machine learning techniques and optimized for performance. It’s stored in a serialized .pkl file for efficient deployment in a production environment.

What makes this project stand out:
	•	Advanced feature engineering to capture economic and demographic indicators.
	•	Robust data preprocessing and transformation workflows.
	•	Industry-standard model selection and hyperparameter tuning techniques.
	•	A production-ready API for real-time predictions.

Key Features
	•	Advanced Machine Learning Model:
A regression model trained on enriched economic and demographic data to predict energy consumption per capita.
	•	Feature Engineering:
Creation of advanced features like GDP per capita and energy intensity to improve prediction accuracy.
	•	Optimized Data Pipeline:
Efficient and modular data preprocessing, ensuring clean and well-structured data for training and inference.
	•	Flexible Model Training:
Seamless training, hyperparameter tuning, and model evaluation wrapped in a well-structured pipeline.
	•	Production-Ready API:
A Flask-based REST API that provides real-time predictions for energy consumption based on user input.

Data Manipulation and Feature Engineering

To improve model performance and create a more meaningful feature set, I applied advanced data manipulation techniques:
	•	Feature Transformation:
Cleaned and normalized the raw data for consistency and model readiness.
	•	Feature Engineering:
Added economic and demographic indicators like GDP per capita and population.
	•	Time-Based Features:
Extracted and structured yearly data for trend analysis and more accurate predictions.
	•	Data Enrichment:
Combined multiple data sources to enhance the dataset’s predictive power.

These steps ensure that the model captures both short-term and long-term influences on energy consumption, making predictions more robust and insightful.

Installation

Prerequisites
	•	Python 3.8+
	•	Virtual environment (recommended)
	•	Pip (Python package installer)

Setup
	1.	Clone the repository:

git clone https://github.com/erenaktuerk/energy-predict.git  
cd energy-predict  

	2.	Create and activate a virtual environment:

For Linux/Mac:

python -m venv energy_predict_env  
source energy_predict_env/bin/activate  

For Windows:

python -m venv energy_predict_env  
energy_predict_env\Scripts\activate  

	3.	Install required dependencies:

pip install -r requirements.txt  

	4.	Ensure the dataset is in place:

The preprocessed dataset is stored in:

data/processed/processed_world_energy_consumption.csv  

Usage

Data Preprocessing

To clean and transform the raw data:

python src/data_preprocessing.py  

Model Training

To train the machine learning model and save it:

python src/train.py  

API Deployment

Start the Flask API to serve real-time predictions:

python main.py  

API Endpoints

Endpoint	Method	Description	Expected Input	Output
/	GET	Home route with project description	None	HTML welcome message
/predict	POST	Predicts energy consumption per capita	{"year": float, "gdp_per_capita": float, "population": float, "energy_per_capita": float}	{"predicted_energy_consumption": float}

Directory Structure

energy-predict/  
├── data/                          # Contains raw and processed data  
│   ├── raw/                       # Raw datasets  
│   └── processed/                 # Cleaned and feature-engineered datasets  
│  
├── energy_predict_env/            # Virtual environment folder (local)  
│  
├── model/                         # Trained machine learning models  
│   └── energy_model.pkl           # Serialized model file  
│  
├── notebooks/                     # Jupyter Notebooks for analysis and experimentation  
│  
├── src/                           # Source code folder  
│   ├── _init_.py                # Package initializer  
│   ├── data_preprocessing.py      # Data cleaning, feature engineering pipeline  
│   ├── train.py                   # Model training and evaluation  
│   ├── utils.py                   # Utility functions for data processing and metrics  
│  
├── .gitignore                     # Specifies files and folders to exclude from version control  
├── main.py                        # Flask API for serving real-time predictions  
├── README.md                      # Project documentation (this file)  
├── requirements.txt               # Python dependencies  

License

This project is licensed under the MIT License - see the LICENSE file for details.

Latest Improvements
	•	Enhanced Feature Engineering: Moving beyond simple features like time and temperature, I now use economic and demographic data to improve prediction accuracy.
	•	More Modular Code: A clean separation between data processing, model training, and API deployment makes this project production-ready.
	•	Optimized Performance: Efficient data handling and feature creation minimize fragmentation and improve speed and memory usage.
	•	Professional API: A fully-featured Flask API delivers real-time predictions and meaningful error handling.
	•	Clearer Structure: Updated and simplified directory layout makes the project more intuitive and easier to maintain.