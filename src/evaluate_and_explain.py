import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.config import MODEL_DIR, PROCESSED_DIR

sns.set_theme(style="whitegrid")


def evaluate_and_interpret():
    # 1. Load Locked Test Set (For the FIRST and ONLY time!)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test_log = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()

    # Load Saved Pipelines
    ensemble_pipeline = joblib.load(MODEL_DIR / "final_ensemble_pipeline.joblib")
    xgb_pipeline = joblib.load(MODEL_DIR / "xgb_shap_pipeline.joblib")

    # ==========================================
    # STEP 10: FINAL TEST SET EVALUATION
    # ==========================================
    print("\n==========================================")
    print("   STEP 10: FINAL TEST SET EVALUATION")
    print("==========================================")

    # Predict on log scale
    y_pred_log = ensemble_pipeline.predict(X_test)

    # Convert back to actual USD dollars using expm1
    y_test_actual = np.expm1(y_test_log)
    y_pred_actual = np.expm1(y_pred_log)

    # Calculate Metrics
    test_rmsle = root_mean_squared_error(y_test_log, y_pred_log)
    test_mae = mean_absolute_error(y_test_actual, y_pred_actual)
    test_r2 = r2_score(y_test_log, y_pred_log)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

    print(f"Final Test RMSLE (Official Metric) : {test_rmsle:.4f}")
    print(f"Final Test R² Score                : {test_r2:.4f}")
    print(f"Final Test MAE (in USD)            : ${test_mae:,.2f}")
    print(f"Mean Absolute Percentage Error     : {mape:.2f}%")

    # Save Residual & Actual vs Predicted Plot for README
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Actual vs Predicted
    sns.scatterplot(x=y_test_actual, y=y_pred_actual, alpha=0.6, ax=axes[0], color="royalblue")
    min_val, max_val = min(y_test_actual.min(), y_pred_actual.min()), max(y_test_actual.max(), y_pred_actual.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    axes[0].set_title(f"Actual vs Predicted SalePrice (R² = {test_r2:.3f})")
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Predicted Price ($)")
    axes[0].legend()

    # Plot 2: Residuals Distribution (Log Scale)
    residuals = y_test_log - y_pred_log
    sns.histplot(residuals, kde=True, ax=axes[1], color="darkorange")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_title("Residuals Distribution (Log Scale)")
    axes[1].set_xlabel("Prediction Error (Actual - Predicted)")

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "test_evaluation_plots.png", dpi=300)
    plt.close()
    print(f"\nSaved evaluation plots to {MODEL_DIR / 'test_evaluation_plots.png'}")

    # ==========================================
    # STEP 11: MODEL INTERPRETATION (SHAP)
    # ==========================================
    print("\n==========================================")
    print("   STEP 11: SHAP GLOBAL EXPLAINABILITY")
    print("==========================================")

    # Extract preprocessor and XGBoost model from pipeline
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    xgb_model = xgb_pipeline.named_steps["model"]

    # Transform X_test and extract clean feature names
    X_test_transformed = preprocessor.transform(X_test)
    raw_feature_names = preprocessor.named_steps["preprocessing"].get_feature_names_out()
    clean_feature_names = [name.split("__")[-1] for name in raw_feature_names]

    X_test_df = pd.DataFrame(X_test_transformed, columns=clean_feature_names)

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_df)

    # Save SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_df, max_display=15, show=False)
    plt.title("Top 15 Features Driving House Prices (SHAP Values)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print Top 10 Most Important Features in Terminal
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features = pd.Series(mean_abs_shap, index=clean_feature_names).sort_values(ascending=False).head(10)
    print("\nTop 10 Most Influential Features (Mean |SHAP|):")
    for feat, val in top_features.items():
        print(f"  {feat:<22}: {val:.4f}")

    print(f"\nSaved SHAP summary chart to {MODEL_DIR / 'shap_summary_plot.png'}")


if __name__ == "__main__":
    evaluate_and_interpret()
