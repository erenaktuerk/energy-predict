import pandas as pd

def load_data(file_path):
    """
    Loads the energy consumption data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.set_index('DateTime', inplace=True)
    data.drop(['Date', 'Time'], axis=1, inplace=True)
    return data

def preprocess_data(data):
    """
    Preprocesses the data by normalizing temperature and energy consumption.

    Args:
        data (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    data['Temperature'] = (data['Temperature'] - data['Temperature'].min()) / (data['Temperature'].max() - data['Temperature'].min())
    data['EnergyConsumption'] = (data['EnergyConsumption'] - data['EnergyConsumption'].min()) / (data['EnergyConsumption'].max() - data['EnergyConsumption'].min())
    return data