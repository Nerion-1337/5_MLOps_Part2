# optimization/profiling.py
import cProfile
import pstats
import time
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as rt

from api.main import load_champion_model_and_threshold

BASE_DIR = Path(__file__).resolve().parent.parent
ONNX_MODEL_PATH = BASE_DIR / "optimization" / "credit_scoring_model.onnx"
REPORTS_DIR = BASE_DIR / "optimization"


def benchmark_inference(n_iterations: int = 1000):
    """Compare la vitesse d'inférence unitaire entre LightGBM et ONNX Runtime."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des modèles et données
    model, feature_names, threshold, run_id = load_champion_model_and_threshold()
    sample_path = BASE_DIR / "data" / "processed" / "X_test_sample.parquet"
    if not sample_path.exists():
        sample_path = BASE_DIR.parent / "5_MLOps_Part1" / "data" / "processed" / "X_test_sample.parquet"

    df_sample = pd.read_parquet(sample_path).head(1)
    cols_drop = [c for c in ["TARGET", "SK_ID_CURR", "index"] if c in df_sample.columns]
    if cols_drop:
        df_sample = df_sample.drop(columns=cols_drop)
    df_sample = df_sample.reindex(columns=feature_names)
    for col in df_sample.columns:
        df_sample[col] = pd.to_numeric(df_sample[col], errors="coerce")
    
    # Matrice préparée pour ONNX
    data_matrix = df_sample.to_numpy(dtype=np.float32)

    # Session ONNX Runtime
    session_options = rt.SessionOptions()
    session_options.intra_op_num_threads = 1
    session = rt.InferenceSession(
        str(ONNX_MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[1].name

    # 2. Benchmark LightGBM
    latencies_lgbm = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict_proba(df_sample)[0][1]
        latencies_lgbm.append((time.perf_counter() - t0) * 1000.0)

    # 3. Benchmark ONNX Runtime
    latencies_onnx = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: data_matrix})
        latencies_onnx.append((time.perf_counter() - t0) * 1000.0)

    # 4. Calcul des métriques
    mean_lgbm, p95_lgbm = np.mean(latencies_lgbm), np.percentile(latencies_lgbm, 95)
    mean_onnx, p95_onnx = np.mean(latencies_onnx), np.percentile(latencies_onnx, 95)
    speedup = mean_lgbm / mean_onnx if mean_onnx > 0 else 1.0

    report_txt = f"""=== RAPPORT DE PROFILAGE & OPTIMISATION (N={n_iterations}) ===
1. LightGBM (Standard) :
   - Latence moyenne : {mean_lgbm:.3f} ms
   - Latence P95     : {p95_lgbm:.3f} ms

2. ONNX Runtime (Optimisé) :
   - Latence moyenne : {mean_onnx:.3f} ms
   - Latence P95     : {p95_onnx:.3f} ms

3. Gain de performance :
   - Accélération    : x{speedup:.2f}
   - Réduction temps : {((mean_lgbm - mean_onnx) / mean_lgbm) * 100:.1f} %
"""
    print(report_txt)

    # Sauvegarde du rapport pour la soutenance
    with open(REPORTS_DIR / "optimization_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)


def run_profiling():
    """Profile les fonctions internes les plus consommatrices via cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()
    benchmark_inference(n_iterations=200)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.dump_stats(str(REPORTS_DIR / "inference_profile.prof"))
    print(f"📊 Fichier de profilage généré : {REPORTS_DIR / 'inference_profile.prof'}")


if __name__ == "__main__":
    run_profiling()