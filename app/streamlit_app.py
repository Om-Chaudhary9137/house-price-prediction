import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import MODEL_DIR, PROCESSED_DIR
from src.data_prep import clean_raw_data

st.set_page_config(
    page_title="Ames Real Estate Valuation Intelligence",
    page_icon="🏛️",
    layout="wide"
)

# Glossy Glassmorphism & Executive Layout Styling
st.markdown(
    """
    <style>
    /* Frosted Glossy Top Header */
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(14px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
        border-bottom: 1px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05) !important;
    }

    /* Fix Top Spacing so Title Never Clips */
    .block-container {
        padding-top: 3.8rem !important;
        padding-bottom: 2rem !important;
    }

    /* Glossy Styling for Status & Manage App Controls */
    div[data-testid="stStatusWidget"],
    .stAppDeployButton,
    button[kind="manageApp"] {
        background: rgba(255, 255, 255, 0.72) !important;
        backdrop-filter: blur(14px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
        color: #0f172a !important;
        border: 1px solid rgba(203, 213, 225, 0.8) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
    }

    /* Clean Executive Metric Card */
    .valuation-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9));
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0f172a;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
    }
    .valuation-label {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .valuation-price {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.15;
    }
    .valuation-sub {
        font-size: 0.9rem;
        color: #475569;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

NEIGHBORHOOD_NAMES = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT & Rail Road",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes": "North Ames",
    "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames",
    "OldTown": "Old Town",
    "SWISU": "South & West of ISU",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker"
}

QUALITY_LABELS = {
    "Ex": "Ex — Excellent",
    "Gd": "Gd — Good",
    "TA": "TA — Typical / Average",
    "Fa": "Fa — Fair"
}


@st.cache_resource
def load_artifacts():
    ensemble = joblib.load(MODEL_DIR / "final_ensemble_pipeline.joblib")
    xgb_pipe = joblib.load(MODEL_DIR / "xgb_shap_pipeline.joblib")
    X_ref = pd.read_csv(PROCESSED_DIR / "X_train.csv")

    defaults = {}
    for col in X_ref.columns:
        if pd.api.types.is_numeric_dtype(X_ref[col]):
            defaults[col] = float(X_ref[col].median())
        else:
            defaults[col] = str(X_ref[col].mode().iloc)
    return ensemble, xgb_pipe, X_ref, defaults


ensemble_pipe, xgb_pipe, X_train_ref, DEFAULT_HOUSE = load_artifacts()

# Executive Header
st.title("Ames Real Estate Valuation Intelligence")
st.markdown(
    "Automated Residential Property Appraisal Engine powered by a **Leakage-Free Ensemble Pipeline** "
    "(Ridge + XGBoost | **Test R²: 0.923** | **MAPE: 8.15%**) with real-time **SHAP Attribution**."
)
st.divider()

# Sidebar Inputs
st.sidebar.subheader("Property Parameters")

neighborhoods = sorted(X_train_ref["Neighborhood"].unique().tolist())
selected_neighborhood = st.sidebar.selectbox(
    "Location / Neighborhood",
    options=neighborhoods,
    index=neighborhoods.index("CollgCr"),
    format_func=lambda x: f"{NEIGHBORHOOD_NAMES.get(x, x)} ({x})"
)

overall_qual = st.sidebar.slider("Overall Material & Finish Quality (1–10)", 1, 10, 7)
gr_liv_area = st.sidebar.number_input("Above-Grade Living Area (sq ft)", 400, 5000, 1700, step=50)
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 1000, step=50)
year_built = st.sidebar.slider("Year Built", 1880, 2010, 2004)
garage_cars = st.sidebar.selectbox("Garage Capacity (Cars)", options=[0, 1, 2, 3, 4], index=2)
full_bath = st.sidebar.selectbox("Full Bathrooms", options=[1, 2, 3, 4], index=1)
kitchen_qual = st.sidebar.selectbox(
    "Kitchen Finish Quality",
    options=["Ex", "Gd", "TA", "Fa"],
    index=1,
    format_func=lambda x: QUALITY_LABELS.get(x, x)
)
exter_qual = st.sidebar.selectbox(
    "Exterior Material Quality",
    options=["Ex", "Gd", "TA", "Fa"],
    index=1,
    format_func=lambda x: QUALITY_LABELS.get(x, x)
)
fireplaces = st.sidebar.slider("Fireplaces", 0, 3, 1)

st.sidebar.divider()
st.sidebar.caption("Model Architecture: 40% Ridge + 60% Optuna-Tuned XGBoost | Holdout RMSLE: 0.1142")

# Prepare Input Record
house_dict = DEFAULT_HOUSE.copy()
house_dict.update({
    "Neighborhood": selected_neighborhood,
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "TotalBsmtSF": total_bsmt_sf,
    "YearBuilt": year_built,
    "YrSold": 2010,
    "GarageCars": garage_cars,
    "FullBath": full_bath,
    "BsmtFullBath": 1,
    "KitchenQual": kitchen_qual,
    "ExterQual": exter_qual,
    "Fireplaces": fireplaces,
    "1stFlrSF": gr_liv_area * 0.65,
    "2ndFlrSF": gr_liv_area * 0.35
})

input_df = clean_raw_data(pd.DataFrame([house_dict]), is_train=False)

# Inference
log_price = ensemble_pipe.predict(input_df)
predicted_price = float(np.expm1(log_price).item())

total_area = gr_liv_area + total_bsmt_sf
price_per_sqft = predicted_price / total_area if total_area > 0 else 0
low_est = predicted_price * (1 - 0.0815)
high_est = predicted_price * (1 + 0.0815)

# Main Dashboard Layout
col1, col2 = st.columns([1.05, 1.45], gap="large")

with col1:
    st.subheader("Valuation Summary")

    st.markdown(
        f"""
        <div class="valuation-card">
            <div class="valuation-label">Estimated Fair Market Value</div>
            <div class="valuation-price">${predicted_price:,.0f}</div>
            <div class="valuation-sub">
                Expected Range (±8.15% MAPE): <strong>${low_est:,.0f} – ${high_est:,.0f}</strong><br>
                Unit Valuation: <strong>${price_per_sqft:,.2f} / total sq ft</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    area_label = NEIGHBORHOOD_NAMES.get(selected_neighborhood, selected_neighborhood)
    st.markdown("#### Property Specification Brief")
    spec_df = pd.DataFrame(
        {
            "Attribute": [
                "Neighborhood",
                "Overall Quality",
                "Total Finished & Basement Area",
                "Construction Year",
                "Garage & Bath Configuration"
            ],
            "Specification": [
                f"{area_label} ({selected_neighborhood})",
                f"{overall_qual} / 10 ({QUALITY_LABELS.get(exter_qual, exter_qual).split(' — ')})",
                f"{total_area:,.0f} sq ft ({gr_liv_area:,} above-grade + {total_bsmt_sf:,} bsmt)",
                f"{year_built} (Age: {2010 - year_built} yrs at benchmark)",
                f"{garage_cars}-Car Garage | {full_bath} Full Bath"
            ]
        }
    )
    st.dataframe(spec_df, hide_index=True, use_container_width=True)

with col2:
    st.subheader("SHAP Feature Attribution Analysis")
    st.caption(
        "Marginal contribution of top property attributes relative to the baseline log-price benchmark (E[f(x)])."
    )

    preprocessor = xgb_pipe.named_steps["preprocessor"]
    xgb_model = xgb_pipe.named_steps["model"]

    transformed_row = preprocessor.transform(input_df)
    feat_names = [n.split("__")[-1] for n in preprocessor.named_steps["preprocessing"].get_feature_names_out()]
    transformed_df = pd.DataFrame(transformed_row, columns=feat_names)

    explainer = shap.TreeExplainer(xgb_model)
    shap_exp = explainer(transformed_df)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    shap.plots.waterfall(shap_exp[0], max_display=8, show=False)
    st.pyplot(fig)
