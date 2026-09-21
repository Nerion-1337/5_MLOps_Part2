# api/schemas.py
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class ClientInput(BaseModel):
    """Payload envoyé à l'API pour un client donné."""
    client_id: Optional[int] = Field(None, description="Identifiant unique du client (ex: SK_ID_CURR)")
    features: Dict[str, Any] = Field(
        ..., 
        description="Dictionnaire des caractéristiques du client (clé: nom de colonne, valeur: valeur)"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("Le dictionnaire de features ne peut pas être vide.")

        # Contrôle sur AMT_CREDIT
        if "AMT_CREDIT" in v and v["AMT_CREDIT"] is not None:
            try:
                credit_val = float(v["AMT_CREDIT"])
            except (ValueError, TypeError):
                raise ValueError("AMT_CREDIT doit être un nombre valide.")
            if credit_val <= 0:
                raise ValueError("AMT_CREDIT doit être strictement supérieur à 0.")

        # Contrôle sur AMT_INCOME_TOTAL
        if "AMT_INCOME_TOTAL" in v and v["AMT_INCOME_TOTAL"] is not None:
            try:
                income_val = float(v["AMT_INCOME_TOTAL"])
            except (ValueError, TypeError):
                raise ValueError("AMT_INCOME_TOTAL doit être un nombre valide.")
            if income_val <= 0:
                raise ValueError("AMT_INCOME_TOTAL doit être strictement supérieur à 0.")

        # Contrôle sur les scores externes
        for ext_col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            if ext_col in v and v[ext_col] is not None:
                try:
                    val = float(v[ext_col])
                except (ValueError, TypeError):
                    raise ValueError(f"{ext_col} doit être un nombre valide.")
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"{ext_col} doit être compris entre 0.0 et 1.0 (reçu : {val}).")

        return v


class PredictionResponse(BaseModel):
    """Réponse structurée retournée par l'API."""
    client_id: Optional[int]
    probability: float = Field(..., description="Probabilité estimée de défaut de paiement")
    threshold: float = Field(..., description="Seuil d'arbitrage métier récupéré depuis MLflow")
    decision: str = Field(..., description="'ACCORDE' ou 'REFUSE'")
    status: str = Field(..., description="Statut de l'inférence")