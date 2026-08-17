import mlflow
import numpy as np
import pandas as pd

BASE_COLUMNS = [
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]

FULL_COLUMNS = BASE_COLUMNS + ["prev_and_damage", "log_annual_premium"]


def build_feature_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["prev_and_damage"] = (
        (df["Previously_Insured"] == 1) & (df["Vehicle_Damage"] == "Yes")
    ).astype(int)
    df["log_annual_premium"] = np.log1p(df["Annual_Premium"])
    return df[FULL_COLUMNS]


class ModelService:
    """Responsável por carregar o modelo do MLflow e gerar predições."""

    def __init__(self, tracking_uri: str, model_name: str, model_alias: str) -> None:
        self._tracking_uri = tracking_uri
        self._model_uri = f"models:/{model_name}@{model_alias}"
        self._model = None

    @property
    def model_uri(self) -> str:
        return self._model_uri

    def load(self) -> "ModelService":
        mlflow.set_tracking_uri(self._tracking_uri)
        self._model = mlflow.sklearn.load_model(self._model_uri)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Modelo ainda não foi carregado.")
        return self._model.predict_proba(features)