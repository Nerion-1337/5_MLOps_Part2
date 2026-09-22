# monitoring/drift_analysis.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Imports robustes Evidently multi-versions
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ModuleNotFoundError:
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset
    except ModuleNotFoundError:
        from evidently.report import Report
        from evidently.presets import DataDriftPreset

BASE_DIR = Path(__file__).resolve().parent.parent
PROD_LOGS_PATH = BASE_DIR / "data" / "production" / "api_predictions.jsonl"
REPORTS_DIR = BASE_DIR / "monitoring" / "reports"


def load_production_data() -> pd.DataFrame:
    """Charge les données clients journalisées lors des appels API."""
    if not PROD_LOGS_PATH.exists():
        raise FileNotFoundError(f"Fichier de logs introuvable : {PROD_LOGS_PATH}")

    records = []
    with open(PROD_LOGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                row = item.get("features", {}).copy()
                row["PREDICTED_PROBABILITY"] = item.get("probability")
                records.append(row)

    if not records:
        raise ValueError("Le fichier de logs de production est vide.")

    return pd.DataFrame(records)


def run_drift_detection():
    """Calcule le Data Drift entre le set de référence et le trafic de production."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Localisation du fichier de référence
    ref_paths = [
        BASE_DIR / "data" / "processed" / "X_test_sample.parquet",
        BASE_DIR.parent / "5_MLOps_Part1" / "data" / "processed" / "X_test_sample.parquet",
    ]
    ref_file = next((p for p in ref_paths if p.exists()), None)
    if not ref_file:
        raise FileNotFoundError("Fichier de référence X_test_sample.parquet introuvable.")

    df_ref = pd.read_parquet(ref_file)
    df_prod = load_production_data()

    # 2. Filtrage des colonnes communes prédictives
    cols_ignore = ["TARGET", "SK_ID_CURR", "index", "PREDICTED_PROBABILITY"]
    common_cols = [c for c in df_prod.columns if c in df_ref.columns and c not in cols_ignore]

    # Forcer la sélection sur les features financières majeures pour un calcul propre et rapide
    top_features = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_BIRTH"
    ]
    selected_cols = [c for c in top_features if c in common_cols]
    if not selected_cols:
        selected_cols = common_cols[:15]

    ref_subset = df_ref[selected_cols].copy()
    prod_subset = df_prod[selected_cols].copy()

    # Conversion stricte en float pour éviter les conflits d'object/None
    for col in selected_cols:
        ref_subset[col] = pd.to_numeric(ref_subset[col], errors="coerce").astype(np.float64)
        prod_subset[col] = pd.to_numeric(prod_subset[col], errors="coerce").astype(np.float64)

    print(f"📊 Analyse du drift sur {len(selected_cols)} variables clés ({len(prod_subset)} requêtes en prod)...")

    # 3. Génération du rapport Evidently
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_subset, current_data=prod_subset)

    # 4. Sauvegarde
    output_html = REPORTS_DIR / "data_drift_report.html"
    report.save_html(str(output_html))
    print(f"✅ Rapport généré avec succès : {output_html}")


if __name__ == "__main__":
    run_drift_detection()