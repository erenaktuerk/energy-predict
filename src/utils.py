import numpy as np
import pandas as pd

def calculate_energy_intensity(df):
    """
    Calculate energy intensity as the ratio of primary energy consumption to GDP.
    
    Args:
        df (pd.DataFrame): The input dataset.
        
    Returns:
        pd.Series: The energy intensity values.
    """
    # To avoid division by zero, replace zeros in GDP with NaN (or a small value)
    return df['primary_energy_consumption'] / df['gdp'].replace(0, np.nan)

def calculate_fossil_share(df):
    """
    Calculate the fossil share of energy consumption as the ratio of fossil fuel consumption
    to primary energy consumption.
    
    Args:
        df (pd.DataFrame): The input dataset.
        
    Returns:
        pd.Series: The fossil share values.
    """
    if 'fossil_fuel_consumption' in df.columns:
        return df['fossil_fuel_consumption'] / df['primary_energy_consumption'].replace(0, np.nan)
    else:
        return pd.Series(np.nan, index=df.index)

def add_additional_features(df):
    """
    Add additional features to enhance the dataset's predictive power.
    
    Currently adds:
    - energy_intensity: primary_energy_consumption / gdp
    - fossil_share: fossil_fuel_consumption / primary_energy_consumption (if available)
    
    Args:
        df (pd.DataFrame): The input dataset.
        
    Returns:
        pd.DataFrame: The dataset with the additional features.
    """
    df['energy_intensity'] = calculate_energy_intensity(df)
    df['fossil_share'] = calculate_fossil_share(df)
    return df

def add_advanced_features(df):
    """
    Add advanced engineered features to the dataset to improve model performance.

    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with new advanced features added.
    """
    # Energy intensity: Energy consumption relative to GDP (how much energy is needed to generate economic output)
    df['energy_intensity'] = df['primary_energy_consumption'] / df['gdp']

    # Fossil fuel dependency: Share of fossil fuels in primary energy consumption
    df['fossil_share'] = df['fossil_fuels_consumption'] / df['primary_energy_consumption']

    # Renewable share: Share of renewable energy in total consumption
    df['renewable_share'] = df['renewables_consumption'] / df['primary_energy_consumption']

    # Energy efficiency indicator: Energy per capita relative to GDP per capita
    df['energy_efficiency'] = df['energy_per_capita'] / df['gdp_per_capita']

    # Avoid fragmentation by returning a fresh, defragmented DataFrame
    return df.copy()