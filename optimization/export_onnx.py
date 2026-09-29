# optimization/export_onnx.py
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as rt
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from api.main import load_champion_model_and_threshold

BASE_DIR = Path(__file__).resolve().parent.parent
ONNX_DIR = BASE_DIR / "optimization"
ONNX_MODEL_PATH = ONNX_DIR / "credit_scoring_model.onnx"


def export_to_onnx():
    """Convertit le modèle LightGBM vers le format ONNX."""
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Chargement du modèle de production MLflow...")
    model, feature_names, threshold, run_id = load_champion_model_and_threshold()
    booster = model.booster_ if hasattr(model, "booster_") else model

    num_features = len(feature_names)
    print(f"⚙️ Conversion du modèle pour {num_features} variables d'entrée...")

    # Définition du type d'entrée attendu par le graphe ONNX
    initial_type = [("float_input", FloatTensorType([None, num_features]))]

    onnx_model = convert_lightgbm(
        booster,
        initial_types=initial_type,
        target_opset=15
    )

    with open(ONNX_MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ Modèle exporté au format ONNX : {ONNX_MODEL_PATH}")

    # --- Validation de non-régression ---
    sample_path = BASE_DIR / "data" / "processed" / "X_test_sample.parquet"
    if not sample_path.exists():
        sample_path = BASE_DIR.parent / "5_MLOps_Part1" / "data" / "processed" / "X_test_sample.parquet"

    if sample_path.exists():
        df_sample = pd.read_parquet(sample_path).head(10)
        cols_drop = [c for c in ["TARGET", "SK_ID_CURR", "index"] if c in df_sample.columns]
        if cols_drop:
            df_sample = df_sample.drop(columns=cols_drop)
        df_sample = df_sample.reindex(columns=feature_names)
        for col in df_sample.columns:
            df_sample[col] = pd.to_numeric(df_sample[col], errors="coerce")
        data_matrix = df_sample.to_numpy(dtype=np.float32)

        # Inférence avec LightGBM original
        pred_lgbm = model.predict_proba(df_sample)[:, 1]

        # Inférence avec ONNX Runtime
        sess = rt.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[1].name  # Probabilités
        pred_onnx_raw = sess.run([output_name], {input_name: data_matrix})[0]

        # Extraction des probabilités de classe 1 selon le format de sortie ONNX
        if isinstance(pred_onnx_raw, list):
            pred_onnx = np.array([row[1] for row in pred_onnx_raw])
        elif pred_onnx_raw.ndim == 2:
            pred_onnx = pred_onnx_raw[:, 1]
        else:
            pred_onnx = pred_onnx_raw

        max_diff = np.max(np.abs(pred_lgbm - pred_onnx))
        print(f"🔍 Test de non-régression : Écart absolu max = {max_diff:.8f}")
        if max_diff < 1e-4:
            print("🎯 Validation réussie : le modèle ONNX produit des scores identiques.")
        else:
            print("⚠️ Attention : divergence détectée entre les prédictions.")


if __name__ == "__main__":
    export_to_onnx()