from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.validation import check_is_fitted


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PREPARATION_NOTEBOOK = MODEL_DIR / "preparacion_modelo.ipynb"
MODEL_FILE = MODEL_DIR / "modelo.pkl"


def build_training_context() -> dict:
    if not PREPARATION_NOTEBOOK.exists():
        raise FileNotFoundError(f"Required notebook not found: {PREPARATION_NOTEBOOK}")

    os.chdir(PROJECT_ROOT)
    notebook = nbformat.read(PREPARATION_NOTEBOOK, as_version=4)
    context: dict = {}
    for cell_number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(
                compile(
                    cell.source,
                    f"{PREPARATION_NOTEBOOK.name}:cell-{cell_number}",
                    "exec",
                ),
                context,
            )

    required = {
        "selected_model",
        "selected_name",
        "selected_prediction",
        "FEATURES",
        "NUMERIC_FEATURES",
        "CATEGORICAL_FEATURES",
        "X_train",
        "X_test",
        "metrics",
    }
    missing = required.difference(context)
    if missing:
        raise RuntimeError(f"Model preparation did not produce: {sorted(missing)}")
    return context


def main() -> None:
    context = build_training_context()
    estimator = context["selected_model"]
    check_is_fitted(estimator)

    probe = context["X_test"].head(10).copy()
    expected_prediction = np.maximum(estimator.predict(probe), 0.0)
    if not np.isfinite(expected_prediction).all():
        raise RuntimeError("The selected estimator returned a non-finite prediction.")

    selected_metrics = context["metrics"].loc[
        context["metrics"]["Modelo"].eq(context["selected_name"])
    ].iloc[0]
    feature_names = list(context["FEATURES"])
    artifact = {
        "artifact_format": "online-retail-customer-spend-regressor",
        "format_version": 1,
        "estimator": estimator,
        "input_features": feature_names,
        "numeric_features": list(context["NUMERIC_FEATURES"]),
        "categorical_features": list(context["CATEGORICAL_FEATURES"]),
        "output": {
            "name": "future_spend",
            "description": (
                "Non-negative eligible customer spend over the future 162-day horizon, "
                "in the source Price field's monetary units."
            ),
            "postprocess": "clip predictions below zero to zero",
        },
        "metadata": {
            "model_name": context["selected_name"],
            "selection_criterion": "lowest MAE on the temporal comparison set",
            "temporal_train_observation": "2010-01-01 through 2010-06-30",
            "temporal_train_target": "2010-07-01 through 2010-12-09",
            "training_customers": int(len(context["X_train"])),
            "test_customers_for_selection": int(len(context["X_test"])),
            "mae_on_temporal_comparison": float(selected_metrics["MAE"]),
            "rmse_on_temporal_comparison": float(selected_metrics["RMSE"]),
            "r2_on_temporal_comparison": float(selected_metrics["R²"]),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    joblib.dump(artifact, MODEL_FILE, compress=3)
    if not MODEL_FILE.is_file() or MODEL_FILE.stat().st_size == 0:
        raise RuntimeError(f"Model artifact was not written correctly: {MODEL_FILE}")

    loaded_artifact = joblib.load(MODEL_FILE)
    loaded_estimator = loaded_artifact["estimator"]
    check_is_fitted(loaded_estimator)
    reloaded_prediction = np.maximum(loaded_estimator.predict(probe), 0.0)
    if not np.allclose(expected_prediction, reloaded_prediction, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Predictions changed after serializing and reloading the model.")
    if not np.isfinite(reloaded_prediction).all():
        raise RuntimeError("The reloaded estimator returned a non-finite prediction.")
    if list(loaded_artifact["input_features"]) != feature_names:
        raise RuntimeError("The saved input schema does not match the training schema.")

    print(f"Saved model: {MODEL_FILE}")
    print(f"Selected estimator: {loaded_artifact['metadata']['model_name']}")
    print(f"Artifact size: {MODEL_FILE.stat().st_size:,} bytes")
    print(f"Reload verification: OK ({len(reloaded_prediction)} predictions)")
    print(
        "Versions: "
        f"Python {platform.python_version()}, pandas {pd.__version__}, "
        f"NumPy {np.__version__}, scikit-learn {sklearn.__version__}, "
        f"joblib {joblib.__version__}"
    )


if __name__ == "__main__":
    main()
