import pandas as pd
import os

df = pd.read_csv("../data/predictive_maintenance_dataset.csv")

print("Dimensions avant nettoyage :", df.shape)
print("\nColonnes :")
print(df.columns)

print("\nValeurs nulles :")
print(df.isnull().sum())

print("\nDoublons :", df.duplicated().sum())

df = df.drop_duplicates()

df["date"] = pd.to_datetime(df["date"])

os.makedirs("../data/processed", exist_ok=True)

df.to_csv("../data/processed/cleaned_data.csv", index=False)

print("\nDimensions après nettoyage :", df.shape)
print("Fichier sauvegardé : data/processed/cleaned_data.csv")