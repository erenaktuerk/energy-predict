import pandas as pd

# Lade den Trainingsdatensatz (ersetze den Pfad durch den richtigen)
train_data = pd.read_csv("data/processed_world_energy_consumption.csv")

# Gebe die Spaltennamen aus
print(train_data.columns)

# Nehme die ersten 4 Features aus den Modell-Daten
top_features = ['temperature', 'population', 'gdp']
print(f"Top 4 Features: {top_features}")

# Falls das Modell mehr Features hat, wähle das vierte Feature aus
fourth_feature = 'biofuel_cons_change_pct'  # Zum Beispiel das 4. Feature
print(f"The fourth feature could be: {fourth_feature}")