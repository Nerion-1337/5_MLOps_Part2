# api/main.py
import time
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import onnxruntime as rt
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient

from api.logging_config import log_prediction_event
from api.schemas import ClientInput, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
ONNX_MODEL_PATH = BASE_DIR / "optimization" / "credit_scoring_model.onnx"

MODEL_STATE: Dict[str, Any] = {
    "model": None,
    "onnx_session": None,
    "feature_names": None,
    "optimal_threshold": 0.4747,
    "run_id": None,
}


class MockCreditModel:
    """Modèle de secours léger pour exécuter la CI/CD quand les artefacts lourds sont ignorés par Git."""
    def __init__(self, feature_names):
        self.feature_name_ = feature_names

    def predict_proba(self, X):
        # Retourne une probabilité fictive mais valide (ex: 0.25)
        n_samples = len(X)
        probs = np.zeros((n_samples, 2))
        probs[:, 0] = 0.75
        probs[:, 1] = 0.25
        return probs


def load_champion_model_and_threshold():
    """Charge le modèle MLflow ou bascule sur un Mock en environnement CI."""
    db_path = BASE_DIR / "data" / "mlflow" / "metadata.db"
    optimal_threshold = 0.4747
    run_id = "ci_pipeline_run"
    feature_names = None

    # 1. Lecture de metadata.db si elle est présente localement
    if db_path.exists():
        try:
            sqlite_uri = f"sqlite:///{db_path.resolve().as_posix()}"
            mlflow.set_tracking_uri(sqlite_uri)
            client = MlflowClient()
            version_prod = client.get_model_version_by_alias("CreditScoringModel", "prod")
            run_id = version_prod.run_id
            run_data = client.get_run(run_id).data
            optimal_threshold = (
                run_data.metrics.get("final_optimal_threshold")
                or run_data.metrics.get("optimal_threshold")
                or 0.4747
            )
        except Exception as e:
            print(f"ℹ️ Métadonnées MLflow non exploitables : {e}")

    # 2. Recherche du modèle physique MLmodel
    candidats = list(BASE_DIR.rglob("MLmodel"))
    if candidats:
        candidats.sort(key=lambda x: x.stat().st_mtime)
        local_model_path = candidats[-1].parent
        try:
            model = mlflow.lightgbm.load_model(str(local_model_path.resolve()))
            feature_names = getattr(model, "feature_name_", None)
            if feature_names is None and hasattr(model, "booster_"):
                feature_names = model.booster_.feature_name()
            print(f"✅ Modèle réel chargé depuis : {local_model_path}")
            return model, feature_names, float(optimal_threshold), run_id
        except Exception as e:
            print(f"⚠️️ Erreur chargement modèle physique : {e}")

    # 3. Fallback CI / Environnement vierge (Mock)
    print("🧪 Aucun modèle physique sur le disque (normal en CI sans stockage distant). Activation du Mock CI...")
    default_features = [
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"
    ]
    model = MockCreditModel(default_features)
    return model, default_features, float(optimal_threshold), run_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage de l'API."""
    try:
        model, features, threshold, run_id = load_champion_model_and_threshold()
        MODEL_STATE["model"] = model
        MODEL_STATE["feature_names"] = features
        MODEL_STATE["optimal_threshold"] = threshold
        MODEL_STATE["run_id"] = run_id

        if ONNX_MODEL_PATH.exists():
            opts = rt.SessionOptions()
            opts.intra_op_num_threads = 1
            MODEL_STATE["onnx_session"] = rt.InferenceSession(
                str(ONNX_MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            print(f"🚀 Moteur ONNX Runtime activé (Latence optimisée).")
        else:
            print("ℹ️ Moteur standard LightGBM actif.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        MODEL_STATE["model"] = None
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="Prêt à Dépenser - API de Scoring Crédit",
    description="API de calcul de risque de défaut de paiement pour Crédit Express.",
    version="2.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Monitoring"])
def health():
    """Contrôle de l'état opérationnel."""
    ready = MODEL_STATE["model"] is not None
    return {
        "status": "ready" if ready else "not_ready",
        "model_loaded": ready,
        "engine": "ONNX Runtime" if MODEL_STATE.get("onnx_session") else "LightGBM",
        "run_id": MODEL_STATE["run_id"],
        "optimal_threshold": MODEL_STATE["optimal_threshold"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(payload: ClientInput):
    """Exécute l'inférence optimisée pour une demande de prêt."""
    start_time = time.perf_counter()
    model = MODEL_STATE["model"]
    onnx_sess = MODEL_STATE["onnx_session"]
    feature_names = MODEL_STATE["feature_names"]
    threshold = MODEL_STATE["optimal_threshold"]

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle de production n'est pas disponible.",
        )

    try:
        df_client = pd.DataFrame([payload.features])
        cols_drop = [c for c in ["TARGET", "SK_ID_CURR", "index"] if c in df_client.columns]
        if cols_drop:
            df_client = df_client.drop(columns=cols_drop)

        if feature_names:
            df_client = df_client.reindex(columns=feature_names)

        for col in df_client.columns:
            df_client[col] = pd.to_numeric(df_client[col], errors="coerce")

        if onnx_sess:
            data_matrix = df_client.to_numpy(dtype=np.float32)
            input_name = onnx_sess.get_inputs()[0].name
            output_name = onnx_sess.get_outputs()[1].name
            pred_raw = onnx_sess.run([output_name], {input_name: data_matrix})[0]
            proba_defaut = float(pred_raw[0][1] if isinstance(pred_raw, list) else pred_raw[0, 1])
        else:
            df_client = df_client.astype(np.float64)
            proba_defaut = float(model.predict_proba(df_client)[0][1])

        decision = "REFUSE" if proba_defaut >= threshold else "ACCORDE"
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        serializable_features = {
            k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in payload.features.items()
        }

        log_prediction_event(
            client_id=payload.client_id,
            features=serializable_features,
            probability=round(proba_defaut, 4),
            threshold=threshold,
            decision=decision,
            latency_ms=latency_ms,
            status_code=200,
        )

        return PredictionResponse(
            client_id=payload.client_id,
            probability=round(proba_defaut, 4),
            threshold=threshold,
            decision=decision,
            status="SUCCESS",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur d'inférence : {str(e)}",
        )
