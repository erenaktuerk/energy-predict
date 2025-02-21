import pandas as pd

# Load the processed dataset
data_path = "data/processed_world_energy_consumption.csv"
df = pd.read_csv(data_path)

# Define the columns to examine
columns_to_check = [
    "primary_energy_consumption",
    "energy_per_capita",
    "energy_per_gdp",
    "renewables_share_energy",
    "fossil_fuel_consumption",
    "greenhouse_gas_emissions",
    "energy_cons_change_pct",
    "renewables_cons_change_pct",
    "fossil_cons_change_pct"
]

# Print the first few rows of the dataset for these columns to observe their values
print("First few rows of selected columns:")
print(df[columns_to_check].head())

# Print some basic statistics for these columns to understand their distribution and dynamics
print("\nBasic statistics for selected columns:")
print(df[columns_to_check].describe())

# Visualize the distribution of the selected columns to get an idea of the spread and trends
import matplotlib.pyplot as plt

for column in columns_to_check:
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=30, alpha=0.7)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Correlation between selected columns
print("\nCorrelation matrix for selected columns:")
correlations = df[columns_to_check].corr()
print(correlations)

# Optionally, you can print the trends over the years for each of the columns
for column in columns_to_check:
    plt.figure(figsize=(10, 6))
    plt.plot(df['year'], df[column], label=column)
    plt.title(f'Trend of {column} over time')
    plt.xlabel('Year')
    plt.ylabel(column)
    plt.legend()
    plt.show()