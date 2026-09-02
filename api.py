
# Librairies 
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pickle
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict



# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    )

logger = logging.getLogger(__name__)



# Configuration via variables d'environnement
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "hist_gradient_boosting_best.pkl"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", BASE_DIR / "models" / "MinMax_scaler.pkl"))
API_KEY = os.getenv("API_KEY")  # si None, l'authentification est désactivée (utile en dev)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

FEATURE_ORDER = [
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

MODEL_VERSION = "1.0.0"


# Class ModelContainer
class ModelContainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.malignant_class_idx: int = 1
        self.supports_proba: bool = False


models = ModelContainer()


# Cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles au démarrage et les libère à l'arrêt."""
    logger.info("Démarrage de l'API")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modèle introuvable: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Scaler introuvable: {SCALER_PATH}")

    try:
        with open(MODEL_PATH, "rb") as f:
            models.model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            models.scaler = pickle.load(f)

    except Exception as e:
        logger.error(f"Erreur critique au chargement: {e}")
        raise RuntimeError(f"Impossible de charger les modèles: {e}") from e

    # Détection dynamique de l'index de la classe maligne (1 ou 'M')
    if hasattr(models.model, "classes_"):
        classes = list(models.model.classes_)

        if 1 in classes:
            models.malignant_class_idx = classes.index(1)

        elif "M" in classes:
            models.malignant_class_idx = classes.index("M")

        else:
            logger.warning(
                f"Classe maligne non identifiée parmi {classes}, index par défaut utilisé: "
                f"{models.malignant_class_idx}"
            )

    models.supports_proba = hasattr(models.model, "predict_proba")

    if not models.supports_proba:
        logger.warning("Le modèle chargé ne supporte pas predict_proba; les probabilités seront fixées à None")

    logger.info(f"Modèles chargés avec succès (classe maligne: index {models.malignant_class_idx})")

    yield

    logger.info("Arrêt de l'API")
    models.model = None
    models.scaler = None


# Application FastAPI
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API de prédiction du cancer du sein",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# CORS : allow_credentials=True est incompatible avec allow_origins=["*"] côté navigateur.
# On n'active les credentials que si des origines explicites sont fournies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=bool(ALLOWED_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Schémas Pydantic
class InputData(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "radius_worst": 16.5,
                "texture_worst": 25.3,
                "perimeter_worst": 110.2,
                "area_worst": 850.0,
                "smoothness_worst": 0.14,
                "compactness_worst": 0.28,
                "concavity_worst": 0.32,
                "concave_points_worst": 0.15,
                "symmetry_worst": 0.29,
                "fractal_dimension_worst": 0.30,
            }
        }
    )

    radius_worst: float = Field(..., gt=7.93, le=40)
    texture_worst: float = Field(..., gt=12, le=50)
    perimeter_worst: float = Field(..., gt=50.4, le=300)
    area_worst: float = Field(..., gt=180, le=5000)
    smoothness_worst: float = Field(..., gt=0.07, le=0.25)
    compactness_worst: float = Field(..., ge=0.03, le=1.10)
    concavity_worst: float = Field(..., ge=0, le=1.30)
    concave_points_worst: float = Field(..., ge=0, le=0.40)
    symmetry_worst: float = Field(..., gt=0.16, le=0.80)
    fractal_dimension_worst: float = Field(..., gt=0.21, le=0.30)

    @field_validator("*", mode="before")
    @classmethod
    def check_not_nan(cls, v):
        """Vérifie qu'aucune valeur n'est NaN ou None."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            raise ValueError("Les valeurs NaN ou Null ne sont pas autorisées")
        return v


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""

    prediction: Literal["M", "B"]
    label: str
    probability: float | None = Field(None, ge=0, le=1)
    model_version: str = MODEL_VERSION


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str
    model_loaded: bool
    scaler_loaded: bool
    model_version: str = MODEL_VERSION


# Endpoints
@app.get("/", tags=["Info"])
def root():
    """Point d'entrée de l'API."""
    return {
        "name": "Breast Cancer Prediction API",
        "version": MODEL_VERSION,
        "author": "Madiba",
        "endpoints": ["/info", "/health", "/predict"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Vérifie l'état de santé de l'API."""
    is_ready = models.model is not None and models.scaler is not None
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=models.model is not None,
        scaler_loaded=models.scaler is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: InputData):
    """Prédit la présence de cancer du sein."""

    if models.model is None or models.scaler is None:
        logger.error("Tentative de prédiction avec modèles non chargés")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèles non disponibles",
        )

    try:
        # Construction du vecteur dans l'ordre explicite
        features = np.array([[getattr(data, feature) for feature in FEATURE_ORDER]])

        # Normalisation
        features_scaled = models.scaler.transform(features)

        # Inférence
        prediction_raw = models.model.predict(features_scaled)[0]
        is_malignant = prediction_raw in (1, "M")

        probability = None
        if models.supports_proba:
            probabilities = models.model.predict_proba(features_scaled)[0]
            prob_m = float(probabilities[models.malignant_class_idx])
            probability = round(prob_m if is_malignant else (1.0 - prob_m), 3)

        prediction = "M" if is_malignant else "B"
        label = "Risque de cancer" if is_malignant else "Absence de cancer"

        logger.info(f"Prédiction réussie: {prediction} | Probabilité: {probability}")

        return PredictionResponse(
            prediction=prediction,
            label=label,
            probability=probability
            )

    except (ValueError, TypeError) as e:
        # Erreur liée aux données/format (ex: shape incompatible avec le scaler)
        logger.error(f"Erreur de données lors de la prédiction: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Les données fournies sont incompatibles avec le modèle",
        )
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la prédiction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue lors de la prédiction",
        )