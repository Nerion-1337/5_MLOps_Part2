# monitoring/dashboard_monitoring.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent.parent
PROD_LOGS_PATH = BASE_DIR / "data" / "production" / "api_predictions.jsonl"
REPORT_HTML_PATH = BASE_DIR / "monitoring" / "reports" / "data_drift_report.html"

st.set_page_config(page_title="Prêt à Dépenser - Monitoring Production", layout="wide")
st.title("🛡️ Dashboard de Monitoring - Modèle de Scoring Crédit")


def load_logs() -> pd.DataFrame:
    """Charge l'historique des appels API depuis les logs JSONL."""
    if not PROD_LOGS_PATH.exists():
        return pd.DataFrame()

    records = []
    with open(PROD_LOGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


df_logs = load_logs()

if df_logs.empty:
    st.warning("⚠️ Aucun log d'appel API disponible dans `data/production/api_predictions.jsonl`.")
    st.stop()

# --- 1. Indicateurs Opérationnels Clés ---
total_calls = len(df_logs)
avg_latency = df_logs["latency_ms"].mean() if "latency_ms" in df_logs else 0.0
refusal_rate = (df_logs["decision"] == "REFUSE").mean() * 100 if "decision" in df_logs else 0.0
optimal_threshold = df_logs["threshold"].iloc[-1] if "threshold" in df_logs else 0.4747

col1, col2, col3, col4 = st.columns(4)
col1.metric("Volume d'appels", f"{total_calls}")
col2.metric("Latence moyenne", f"{avg_latency:.2f} ms")
col3.metric("Taux de refus crédit", f"{refusal_rate:.1f} %")
col4.metric("Seuil appliqué", f"{optimal_threshold:.4f}")

st.divider()

# --- 2. Visualisations des Distributions ---
tab_distrib, tab_drift, tab_logs = st.tabs(
    ["📊 Distributions & Latence", "🧪 Rapport Data Drift (Evidently)", "📄 Logs Bruts"]
)

with tab_distrib:
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Distribution des Probabilités de Défaut")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df_logs["probability"], bins=15, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.axvline(optimal_threshold, color="red", linestyle="--", label=f"Seuil ({optimal_threshold:.2f})")
        ax.set_xlabel("Probabilité de défaut")
        ax.set_ylabel("Nombre de dossiers")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with c_right:
        st.subheader("Distribution de la Latence d'Inférence")
        fig_lat, ax_lat = plt.subplots(figsize=(6, 3.5))
        ax_lat.hist(df_logs["latency_ms"], bins=15, color="#2ca02c", edgecolor="black", alpha=0.7)
        ax_lat.set_xlabel("Latence (ms)")
        ax_lat.set_ylabel("Nombre de requêtes")
        st.pyplot(fig_lat)
        plt.close(fig_lat)

with tab_drift:
    st.subheader("Analyse de Dérive des Données")
    if REPORT_HTML_PATH.exists():
        with open(REPORT_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.info("Rapport HTML non trouvé. Exécutez `uv run python monitoring/drift_analysis.py`.")

with tab_logs:
    st.subheader("Derniers Appels Enregistrés")
    display_cols = [c for c in ["timestamp", "client_id", "probability", "threshold", "decision", "latency_ms", "status_code"] if c in df_logs.columns]
    st.dataframe(df_logs[display_cols].tail(20), use_container_width=True)