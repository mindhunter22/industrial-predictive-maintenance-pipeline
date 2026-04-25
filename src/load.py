import pandas as pd
from sqlalchemy import create_engine

# lire données nettoyées
df = pd.read_csv("../data/processed/cleaned_data.csv")

# connexion PostgreSQL
engine = create_engine("postgresql://postgres:260103@localhost:5432/industrial_db")

# envoyer vers SQL
df.to_sql("machine_data", engine, if_exists="replace", index=False)

print("✅ Data envoyée dans PostgreSQL")