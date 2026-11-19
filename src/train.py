import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import MODEL_DIR, PROCESSED_DIR, RANDOM_STATE
from src.features import build_preprocessor

# Suppress Optuna verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_cv(model, X_train, y_train, cv) -> tuple[float, float]:
    """
    Wraps preprocessor + model in a single Pipeline and runs 5-Fold CV.
    Returns mean and std of RMSLE (RMSE on log-transformed target).
    """
    preprocessor = build_preprocessor(X_train)
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = -cross_val_score(
        full_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    return scores.mean(), scores.std()


def run_training_pipeline():
    # 1. Load Training Data ONLY (X_test remains locked!)
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ==========================================
    # STEP 7: BASELINE MODEL
    # ==========================================
    print("\n--- STEP 7: BASELINE BENCHMARK ---")
    dummy_mean, dummy_std = evaluate_cv(DummyRegressor(strategy="mean"), X_train, y_train, cv)
    print(f"Dummy Mean Predictor  | 5-Fold RMSLE: {dummy_mean:.4f} (+/- {dummy_std:.4f})")

    # ==========================================
    # STEP 8: MODEL SELECTION (5 Algorithms)
    # ==========================================
    print("\n--- STEP 8: MODEL SELECTION (5-FOLD CV) ---")
    models = {
        "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=0.0005, max_iter=10000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    }

    results = []
    for name, model in models.items():
        mean_score, std_score = evaluate_cv(model, X_train, y_train, cv)
        results.append({"Model": name, "Mean_RMSLE": mean_score, "Std_RMSLE": std_score})
        print(f"{name:<20} | 5-Fold RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")

    # ==========================================
    # STEP 9: HYPERPARAMETER TUNING (OPTUNA)
    # ==========================================
    print("\n--- STEP 9: HYPERPARAMETER TUNING (OPTUNA) ---")

    # 9A. Tune Ridge Alpha
    def ridge_objective(trial):
        alpha = trial.suggest_float("alpha", 1.0, 50.0, log=True)
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        mean_score, _ = evaluate_cv(model, X_train, y_train, cv)
        return mean_score

    ridge_study = optuna.create_study(direction="minimize")
    ridge_study.optimize(ridge_objective, n_trials=20)
    best_ridge_params = ridge_study.best_params
    print(f"Best Ridge RMSLE: {ridge_study.best_value:.4f} | Params: {best_ridge_params}")

    # 9B. Tune XGBoost Hyperparameters
    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        model = XGBRegressor(**params)
        mean_score, _ = evaluate_cv(model, X_train, y_train, cv)
        return mean_score

    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(xgb_objective, n_trials=25)
    best_xgb_params = xgb_study.best_params
    best_xgb_params["random_state"] = RANDOM_STATE
    best_xgb_params["n_jobs"] = -1
    print(f"Best XGBoost RMSLE: {xgb_study.best_value:.4f} | Params: {best_xgb_params}")

    # ==========================================
    # PRO MOVE: ENSEMBLE (RIDGE + XGBOOST)
    # ==========================================
    # Linear models and Tree models make different errors; blending them reduces variance!
    tuned_ridge = Ridge(**best_ridge_params, random_state=RANDOM_STATE)
    tuned_xgb = XGBRegressor(**best_xgb_params)

    ensemble_model = VotingRegressor(
        estimators=[("ridge", tuned_ridge), ("xgb", tuned_xgb)],
        weights=[0.4, 0.6]
    )

    ens_mean, ens_std = evaluate_cv(ensemble_model, X_train, y_train, cv)
    print(f"\n★ Tuned Ensemble (40% Ridge + 60% XGB) | 5-Fold RMSLE: {ens_mean:.4f} (+/- {ens_std:.4f})")

    # Fit final pipeline on full X_train and save artifacts
    final_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X_train)),
        ("model", ensemble_model)
    ])
    final_pipeline.fit(X_train, y_train)

    # Also fit standalone tuned XGBoost pipeline for SHAP explainability in Step 11
    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X_train)),
        ("model", tuned_xgb)
    ])
    xgb_pipeline.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_DIR / "final_ensemble_pipeline.joblib")
    joblib.dump(xgb_pipeline, MODEL_DIR / "xgb_shap_pipeline.joblib")
    print(f"\nModels successfully saved to {MODEL_DIR}/!")


if __name__ == "__main__":
    run_training_pipeline()
