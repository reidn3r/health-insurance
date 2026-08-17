from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.core.config import get_settings
from api.routes import health, predict
from api.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    service = ModelService(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        model_name=settings.MODEL_NAME,
        model_alias=settings.MODEL_ALIAS,
    )
    service.load()
    app.state.model_service = service
    yield
    app.state.model_service = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Health Insurer Model API",
        description="API de scoring do modelo de cross-sell de seguro de veículo.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(predict.router)
    return app


app = create_app()