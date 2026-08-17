from fastapi import APIRouter, Request

router = APIRouter(tags=["meta"])


@router.get("/health")
def health(request: Request) -> dict:
    service = request.app.state.model_service
    return {"status": "ok", "model": service.model_uri}