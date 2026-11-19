import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR, PROCESSED_DIR
from src.data_prep import clean_raw_data

app = FastAPI(
    title="Ames House Price Prediction API",
    description="Production REST API using an Ensemble (Ridge + XGBoost) pipeline.",
    version="1.0.0"
)

# Load trained pipeline and create a baseline default row from X_train
pipeline = joblib.load(MODEL_DIR / "final_ensemble_pipeline.joblib")
xgb_pipeline = joblib.load(MODEL_DIR / "xgb_shap_pipeline.joblib")
X_train_ref = pd.read_csv(PROCESSED_DIR / "X_train.csv")


def get_default_house() -> dict:
    """Creates a baseline house dictionary using median for numbers and mode for categories."""
    defaults = {}
    for col in X_train_ref.columns:
        if pd.api.types.is_numeric_dtype(X_train_ref[col]):
            defaults[col] = float(X_train_ref[col].median())
        else:
            defaults[col] = str(X_train_ref[col].mode()[0])
    return defaults


DEFAULT_HOUSE = get_default_house()


# Pydantic Schema for Input Validation
class HouseFeaturesInput(BaseModel):
    OverallQual: int = Field(7, ge=1, le=10, description="Overall material and finish quality (1-10)")
    GrLivArea: float = Field(1600.0, gt=300, description="Above grade living area in sq ft")
    TotalBsmtSF: float = Field(1000.0, ge=0, description="Total basement area in sq ft")
    GarageCars: int = Field(2, ge=0, le=5, description="Size of garage in car capacity")
    YearBuilt: int = Field(2005, ge=1870, le=2025, description="Original construction year")
    FullBath: int = Field(2, ge=0, le=5, description="Full bathrooms above grade")
    Neighborhood: str = Field("CollgCr", description="Physical location within Ames city limits")
    KitchenQual: str = Field("Gd", description="Kitchen quality (Ex, Gd, TA, Fa, Po)")
    ExterQual: str = Field("Gd", description="Exterior material quality (Ex, Gd, TA, Fa, Po)")
    Fireplaces: int = Field(1, ge=0, le=4, description="Number of fireplaces")


@app.get("/")
def health_check():
    return {"status": "healthy", "model": "Tuned Ensemble (Ridge + XGBoost)"}


@app.post("/predict")
def predict_price(user_input: HouseFeaturesInput):
    try:
        # Start with baseline house and update user-provided features
        house_data = DEFAULT_HOUSE.copy()
        house_data.update(user_input.model_dump())
        house_data["1stFlrSF"] = house_data["GrLivArea"] * 0.65
        house_data["2ndFlrSF"] = house_data["GrLivArea"] * 0.35
        house_data.update(house_data)

        # Convert to DataFrame and apply deterministic pre-split cleaning
        input_df = pd.DataFrame([house_data])
        cleaned_df = clean_raw_data(input_df, is_train=False)

        # 1. Ensemble Prediction (Predict log price and transform back to USD)
        log_pred = pipeline.predict(cleaned_df)[0]
        predicted_price = float(np.expm1(log_pred))


        # 2. Local SHAP Feature Contributions (for Dash interactive bar chart)
        preprocessor = xgb_pipeline.named_steps["preprocessor"]
        xgb_model = xgb_pipeline.named_steps["model"]
        transformed = preprocessor.transform(cleaned_df)
        feat_names = [n.split("__")[-1] for n in preprocessor.named_steps["preprocessing"].get_feature_names_out()]

        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(transformed)[0]

        # Top 8 absolute drivers
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:8]
        shap_drivers = [
            {"feature": feat_names[i], "impact": round(float(shap_vals[i]), 4)}
            for i in top_idx
        ]

        return {
            "predicted_price_usd": round(predicted_price, 2),
            "formatted_price": f"${predicted_price:,.0f}",
            "top_drivers": shap_drivers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
