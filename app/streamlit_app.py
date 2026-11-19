import sys
from pathlib import Path

# Project root folder ko Python path mein add karein taaki 'src' import ho sake
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

st.set_page_config(page_title="Real Estate Valuation AI", page_icon="🏠", layout="wide")


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
            defaults[col] = str(X_ref[col].mode())
    return ensemble, xgb_pipe, X_ref, defaults


ensemble_pipe, xgb_pipe, X_train_ref, DEFAULT_HOUSE = load_artifacts()

# Header
st.title("🏠 Smart Real Estate Valuation & Explainability Engine")
st.markdown("Estimate fair market value using a **Leakage-Free Ensemble ML Pipeline (Ridge + XGBoost)** with live **SHAP Explanations**.")

# Sidebar Inputs
st.sidebar.header("🔧 Configure Property Features")

neighborhoods = sorted(X_train_ref["Neighborhood"].unique().tolist())
selected_neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods, index=neighborhoods.index("CollgCr"))

overall_qual = st.sidebar.slider("Overall Material & Finish Quality (1-10)", 1, 10, 7)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 400, 5000, 1650, step=50)
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 1000, step=50)
year_built = st.sidebar.slider("Year Built", 1880, 2010, 2004)
garage_cars = st.sidebar.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4], index=2)
full_bath = st.sidebar.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)
kitchen_qual = st.sidebar.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"], index=1)
exter_qual = st.sidebar.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa"], index=1)
fireplaces = st.sidebar.slider("Number of Fireplaces", 0, 3, 1)

# Build input row
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
    "KitchenQual": kitchen_qual,
    "ExterQual": exter_qual,
    "Fireplaces": fireplaces,
    "1stFlrSF": gr_liv_area * 0.65,
    "2ndFlrSF": gr_liv_area * 0.35
})

input_df = clean_raw_data(pd.DataFrame([house_dict]), is_train=False)

# Predict
log_price = ensemble_pipe.predict(input_df)[0]
predicted_price = float(np.expm1(log_price))

# Display Results
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("💰 Estimated Market Value")
    st.metric(label="Predicted Sale Price", value=f"${predicted_price:,.0f}")
    st.info(
        f"**Property Summary:**\n"
        f"- **Location:** {selected_neighborhood}\n"
        f"- **Quality Rating:** {overall_qual}/10\n"
        f"- **Total Living + Bsmt:** {gr_liv_area + total_bsmt_sf:,.0f} sq ft\n"
        f"- **Built Year:** {year_built}"
    )

with col2:
    st.subheader("🔍 Why This Price? (Local SHAP Waterfall)")
    preprocessor = xgb_pipe.named_steps["preprocessor"]
    xgb_model = xgb_pipe.named_steps["model"]

    transformed_row = preprocessor.transform(input_df)
    feat_names = [n.split("__")[-1] for n in preprocessor.named_steps["preprocessing"].get_feature_names_out()]
    transformed_df = pd.DataFrame(transformed_row, columns=feat_names)

    explainer = shap.TreeExplainer(xgb_model)
    shap_exp = explainer(transformed_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_exp[0], max_display=8, show=False)
    st.pyplot(fig)

