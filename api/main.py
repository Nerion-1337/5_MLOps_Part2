# api/main.py
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient

from api.schemas import ClientInput, PredictionResponse

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# État global chargé une seule fois au boot
MODEL_STATE: Dict[str, Any] = {
    "model": None,
    "feature_names": None,
    "optimal_threshold": None,
    "run_id": None
}


def load_champion_model_and_threshold():
    """Récupère le modèle et son seuil optimal directement depuis MLflow."""
    db_path = BASE_DIR / "data" / "mlflow" / "metadata.db"
    sqlite_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(sqlite_uri)
    client = MlflowClient()

    # 1. Récupération des métadonnées du modèle 'prod'
    version_prod = client.get_model_version_by_alias("CreditScoringModel", "prod")
    run_id = version_prod.run_id
    raw_source = version_prod.source

    # 2. Récupération dynamique du seuil sauvegardé dans MLflow
    run_data = client.get_run(run_id).data
    # On cherche en priorité la métrique 'final_optimal_threshold', sinon fallback sur 0.54
    optimal_threshold = run_data.metrics.get("optimal_threshold", 0.5)

    # 3. Localisation physique du dossier du modèle
    clean_source = urllib.parse.unquote(raw_source).replace("\\", "/")
    folder_ids = [run_id] + [p for p in clean_source.split("/") if p.startswith("m-") or len(p) == 32]

    local_model_path = None
    mlmodel_files = list(BASE_DIR.rglob("MLmodel"))
    for mlm in mlmodel_files:
        str_path = str(mlm.resolve()).replace("\\", "/")
        if any(fid in str_path for fid in folder_ids if fid):
            local_model_path = mlm.parent
            break

    if local_model_path is None and mlmodel_files:
        mlmodel_files.sort(key=lambda x: x.stat().st_mtime)
        local_model_path = mlmodel_files[-1].parent

    if local_model_path is None or not local_model_path.exists():
        raise RuntimeError("Impossible de localiser physiquement le modèle sur le disque.")

    # 4. Chargement du modèle avec l'URI sécurisée
    chemin_posix = local_model_path.resolve().as_posix()
    model_uri = f"file:///{chemin_posix}"
    model = mlflow.lightgbm.load_model(model_uri)

    # 5. Extraction de la liste des variables d'entraînement
    feature_names = getattr(model, "feature_name_", None)
    if feature_names is None and hasattr(model, "booster_"):
        feature_names = model.booster_.feature_name()

    return model, feature_names, float(optimal_threshold), run_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie FastAPI : chargement unique du modèle au démarrage."""
    print("⏳ Initialisation de l'API et récupération des artefacts MLflow...")
    try:
        model, features, threshold, run_id = load_champion_model_and_threshold()
        MODEL_STATE["model"] = model
        MODEL_STATE["feature_names"] = features
        MODEL_STATE["optimal_threshold"] = threshold
        MODEL_STATE["run_id"] = run_id
        print(f"✅ Modèle MLflow chargé (Run ID: {run_id}).")
        print(f"🎯 Seuil métier synchronisé depuis MLflow : {threshold}")
        print(f"📋 Variables d'entraînement : {len(features) if features else 'Non défini'}")
    except Exception as e:
        print(f"❌ Erreur critique lors de la synchronisation MLflow : {e}")
        MODEL_STATE["model"] = None
    yield
    MODEL_STATE.clear()
    print("🛑 Arrêt de l'API.")


app = FastAPI(
    title="Prêt à Dépenser - API de Scoring Crédit",
    description="API exposant le modèle de scoring validé dans MLflow pour le service Crédit Express.",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", tags=["Monitoring"])
def health():
    """Vérification technique de disponibilité."""
    ready = MODEL_STATE["model"] is not None
    return {
        "status": "ready" if ready else "not_ready",
        "model_loaded": ready,
        "run_id": MODEL_STATE["run_id"],
        "optimal_threshold": MODEL_STATE["optimal_threshold"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(payload: ClientInput):
    """Exécute l'inférence pour un profil client."""
    model = MODEL_STATE["model"]
    feature_names = MODEL_STATE["feature_names"]
    threshold = MODEL_STATE["optimal_threshold"]

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle de production n'est pas disponible."
        )

    try:
        # Transformation du payload JSON en DataFrame
        df_client = pd.DataFrame([payload.features])

        # Suppression des colonnes interdites à l'inférence
        for col_to_remove in ["TARGET", "SK_ID_CURR"]:
            if col_to_remove in df_client.columns:
                df_client = df_client.drop(columns=[col_to_remove])

        # Alignement strict sur les variables du modèle
        if feature_names:
            for col in feature_names:
                if col not in df_client.columns:
                    df_client[col] = np.nan
            df_client = df_client[feature_names]

        # Inférence de la probabilité de faillite (classe 1)
        proba_defaut = float(model.predict_proba(df_client)[0][1])

        # Décision selon le seuil métier issu de MLflow
        decision = "REFUSE" if proba_defaut >= threshold else "ACCORDE"

        return PredictionResponse(
            client_id=payload.client_id,
            probability=round(proba_defaut, 4),
            threshold=threshold,
            decision=decision,
            status="SUCCESS"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur d'inférence : {str(e)}"
        )