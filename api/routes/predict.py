import numpy as np
from fastapi import APIRouter, HTTPException, Request

from api.models.schemas import BatchPrediction, InsuranceRecord, Prediction
from api.services.model_service import ModelService, build_feature_frame

router = APIRouter(prefix="/predict", tags=["prediction"])


def _get_service(request: Request) -> ModelService:
    return request.app.state.model_service


def _prediction_from_proba(p0: float, p1: float) -> Prediction:
    return Prediction(
        response_score=float(p1),
        probability_no=float(p0),
        prediction=int(p1 >= 0.5),
    )


@router.post("", response_model=Prediction)
def predict(record: InsuranceRecord, request: Request) -> Prediction:
    service = _get_service(request)
    features = build_feature_frame([record.model_dump()])
    proba = service.predict_proba(features)[0]
    return _prediction_from_proba(proba[0], proba[1])


@router.post("/batch", response_model=BatchPrediction)
def predict_batch(records: list[InsuranceRecord], request: Request) -> BatchPrediction:
    if not records:
        raise HTTPException(status_code=400, detail="Lista de registros vazia.")
    service = _get_service(request)
    features = build_feature_frame([r.model_dump() for r in records])
    proba = service.predict_proba(features)
    return BatchPrediction(
        predictions=[
            _prediction_from_proba(p0, p1) for p0, p1 in zip(proba[:, 0], proba[:, 1])
        ]
    )