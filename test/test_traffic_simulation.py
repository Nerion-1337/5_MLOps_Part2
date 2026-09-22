import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Chargement des données de test
data_path = Path("data/processed/X_test_sample.parquet")
if not data_path.exists():
    data_path = Path("../5_MLOps_Part1/data/processed/X_test_sample.parquet")

df = pd.read_parquet(data_path).head(10)
api_url = "http://127.0.0.1:8000/predict"

for idx, row in df.iterrows():
    # Détection de l'ID client
    client_id = int(row["SK_ID_CURR"]) if "SK_ID_CURR" in row and not pd.isna(row["SK_ID_CURR"]) else int(idx)
    
    # Nettoyage dictionnaire : conversion NaN -> None et types scalaires natifs
    features_dict = {}
    for col_name, val in row.items():
        if col_name in ["TARGET", "SK_ID_CURR"]:
            continue
        if pd.isna(val) or val is None:
            features_dict[col_name] = None
        elif isinstance(val, (np.integer, int)):
            features_dict[col_name] = int(val)
        elif isinstance(val, (np.floating, float)):
            features_dict[col_name] = float(val)
        else:
            features_dict[col_name] = val

    payload = {
        "client_id": client_id,
        "features": features_dict
    }

    resp = requests.post(api_url, json=payload)
    if resp.status_code == 200:
        res_data = resp.json()
        print(f"✅ Client {client_id} -> HTTP 200 | Proba: {res_data['probability']} | Décision: {res_data['decision']}")
    else:
        print(f"❌ Client {client_id} -> HTTP {resp.status_code} | Détail: {resp.text}")