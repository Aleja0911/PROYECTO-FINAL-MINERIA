from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from sklearn.utils.validation import check_is_fitted


logger = logging.getLogger("online_retail_api")

EXPECTED_FEATURES = [
    "historical_spend",
    "order_count",
    "units_purchased",
    "distinct_products",
    "active_days",
    "days_since_last_purchase",
    "mean_order_spend",
    "historical_return_lines",
    "historical_cancellation_lines",
    "historical_adjustment_lines",
    "country",
]

PositiveAmount = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeAmount = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveCount = Annotated[StrictInt, Field(gt=0)]
NonNegativeCount = Annotated[StrictInt, Field(ge=0)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    historical_spend: PositiveAmount
    order_count: PositiveCount
    units_purchased: PositiveCount
    distinct_products: PositiveCount
    active_days: PositiveCount
    days_since_last_purchase: NonNegativeCount
    mean_order_spend: PositiveAmount
    historical_return_lines: NonNegativeCount
    historical_cancellation_lines: NonNegativeCount
    historical_adjustment_lines: NonNegativeCount
    country: str = Field(min_length=1, max_length=100)

    @field_validator("country")
    @classmethod
    def country_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("country must not be blank")
        return normalized


class PredictionResponse(BaseModel):
    prediction: float
    target: str
    model: str


def load_model_bundle() -> dict:
    configured_path = os.environ.get("MODEL_PATH")
    model_path = (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else Path(__file__).resolve().parents[1]
        / "05_MACHINE_LEARNING"
        / "modelo.pkl"
    )
    if not model_path.is_file():
        raise FileNotFoundError("The configured model file does not exist.")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("The model artifact has an invalid format.")
    if bundle.get("artifact_format") != "online-retail-customer-spend-regressor":
        raise ValueError("The model artifact format is not supported.")
    if bundle.get("input_features") != EXPECTED_FEATURES:
        raise ValueError("The model input schema is not supported.")

    estimator = bundle.get("estimator")
    check_is_fitted(estimator)
    return bundle


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model_bundle = None
    try:
        application.state.model_bundle = load_model_bundle()
        logger.info("Prediction model loaded successfully.")
    except Exception:
        logger.exception("Prediction model could not be loaded.")
    yield


app = FastAPI(
    title="Online Retail II - Customer Spend API",
    description=(
        "API para estimar el gasto elegible futuro de clientes usando el pipeline "
        "serializado de regresión."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    safe_errors = [
        {
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", "Invalid value."),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors},
    )


@app.get("/", tags=["Información"])
def root() -> dict[str, str]:
    return {
        "service": "Online Retail II customer spend prediction API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "prediction": "POST /predict",
    }


@app.get(
    "/health",
    tags=["Servicio"],
    responses={503: {"description": "El modelo no está disponible."}},
)
def health(request: Request) -> JSONResponse:
    bundle = request.app.state.model_bundle
    if bundle is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_available": False},
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_available": True,
            "model": bundle["metadata"]["model_name"],
        },
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predicción"],
    summary="Estimar gasto futuro de un cliente",
    responses={
        500: {"description": "No se pudo generar la predicción."},
        503: {"description": "El modelo no está disponible."},
    },
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    bundle = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        )

    try:
        features = pd.DataFrame(
            [payload.model_dump()],
            columns=bundle["input_features"],
        )
        raw_prediction = float(bundle["estimator"].predict(features)[0])
        if not np.isfinite(raw_prediction):
            raise ValueError("The model returned a non-finite result.")
        prediction = max(raw_prediction, 0.0)
        return PredictionResponse(
            prediction=prediction,
            target=bundle["output"]["name"],
            model=bundle["metadata"]["model_name"],
        )
    except Exception:
        logger.exception("Prediction execution failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be generated.",
        ) from None
