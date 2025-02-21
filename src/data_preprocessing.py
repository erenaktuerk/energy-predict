import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
def load_data(file_path):
    """
    Load the dataset from the given file path.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)

# Clean the dataset
def clean_data(df):
    """
    Perform data cleaning by handling missing values, removing unnecessary columns,
    and ensuring data consistency.

    Args:
    - df (pd.DataFrame): The raw dataset to clean.

    Returns:
    - pd.DataFrame: The cleaned dataset.
    """
    
    # Remove non-essential columns
    df.drop(columns=['iso_code'], errors='ignore', inplace=True)
    
    # Identify numerical columns and fill missing values with their respective means
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Ensure proper data types for key numerical columns
    df['year'] = df['year'].astype(int)
    df['population'] = df['population'].astype(float)
    df['gdp'] = df['gdp'].astype(float)
    df['primary_energy_consumption'] = df['primary_energy_consumption'].astype(float)
    df['energy_cons_change_pct'] = df['energy_cons_change_pct'].astype(float)

    return df

# Feature engineering
def feature_engineering(df):
    """
    Generate new features to enhance predictive power.

    Args:
    - df (pd.DataFrame): The cleaned dataset.

    Returns:
    - pd.DataFrame: The dataset with additional features.
    """
    df['gdp_per_capita'] = df['gdp'] / df['population']
    df['energy_per_capita'] = df['primary_energy_consumption'] / df['population']
    
    return df

# Normalize selected features
def normalize_data(df):
    """
    Normalize the dataset's features to bring them into a common scale.

    Args:
    - df (pd.DataFrame): The dataset to normalize.

    Returns:
    - pd.DataFrame: The normalized dataset.
    """
    scaler = MinMaxScaler()
    features_to_normalize = ['primary_energy_consumption', 'energy_per_capita', 'gdp_per_capita', 'energy_cons_change_pct']
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    
    return df

# Save the preprocessed dataset
def save_cleaned_data(df, output_path):
    """
    Save the preprocessed dataset to a CSV file.

    Args:
    - df (pd.DataFrame): The cleaned dataset.
    - output_path (str): The path to save the dataset.
    """
    df.to_csv(output_path, index=False)

# Main function orchestrating the preprocessing pipeline
def main():
    """
    Execute the full preprocessing pipeline, including data loading, 
    cleaning, feature engineering, normalization, and saving the output.
    """
    # Define file paths
    file_path = 'data/world_energy_consumption.csv'  
    output_path = 'data/processed_world_energy_consumption.csv'

    # Load, clean, and preprocess the dataset
    df = load_data(file_path)
    df_cleaned = clean_data(df)
    df_featured = feature_engineering(df_cleaned)
    df_normalized = normalize_data(df_featured)
    
    # Save the final preprocessed dataset
    save_cleaned_data(df_normalized, output_path)
    
    print(f"Data preprocessing complete. Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    main()