# api/logging_config.py
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
PROD_DATA_DIR = BASE_DIR / "data" / "production"
LOG_FILE_PATH = PROD_DATA_DIR / "api_predictions.jsonl"


def log_prediction_event(
    client_id: int | None,
    features: Dict[str, Any],
    probability: float,
    threshold: float,
    decision: str,
    latency_ms: float,
    status_code: int = 200,
) -> None:
    """Enregistre un événement d'inférence en local sous forme de log JSON structuré."""
    PROD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_id": client_id,
        "probability": probability,
        "threshold": threshold,
        "decision": decision,
        "latency_ms": round(latency_ms, 2),
        "status_code": status_code,
        "features": features,
    }

    with open(LOG_FILE_PATH, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")