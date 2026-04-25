import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. Charger les données
# =========================
df = pd.read_csv("../data/processed/cleaned_data.csv")

features = [
    "metric1", "metric2", "metric3", "metric4", "metric5",
    "metric6", "metric7", "metric8", "metric9"
]

# =========================
# 2. Gérer déséquilibre
# =========================
df_0 = df[df["failure"] == 0]
df_1 = df[df["failure"] == 1]

df_0_sampled = df_0.sample(len(df_1) * 5, random_state=42)

df_balanced = pd.concat([df_0_sampled, df_1])

X = df_balanced[features]
y = df_balanced["failure"]

# =========================
# 3. Train / Test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. Modèle
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight={0: 1, 1: 50}
)

model.fit(X_train, y_train)

# =========================
# 5. Prédictions sur TOUT le dataset
# =========================
df["prediction_proba"] = model.predict_proba(df[features])[:, 1]

threshold = 0.3
df["prediction"] = (df["prediction_proba"] > threshold).astype(int)

# =========================
# 6. Sauvegarde
# =========================
df.to_csv("../data/processed/data_with_predictions.csv", index=False)

print("✅ Fichier avec prédictions créé : data_with_predictions.csv")