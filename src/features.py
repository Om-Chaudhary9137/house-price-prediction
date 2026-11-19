import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to create domain-specific real estate features.
    Stateless (row-wise operations), so completely safe from data leakage.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Total Square Footage (Basement + 1st Floor + 2nd Floor)
        X["TotalSF"] = (
            X["TotalBsmtSF"].fillna(0)
            + X["1stFlrSF"].fillna(0)
            + X["2ndFlrSF"].fillna(0)
        )

        # 2. Total Bathrooms (Full baths + 0.5 * Half baths)
        X["TotalBath"] = (
            X["FullBath"].fillna(0)
            + 0.5 * X["HalfBath"].fillna(0)
            + X["BsmtFullBath"].fillna(0)
            + 0.5 * X["BsmtHalfBath"].fillna(0)
        )

        # 3. Age of the House & Years Since Remodel at the time of sale
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["IsRemodeled"] = (X["YearBuilt"] != X["YearRemodAdd"]).astype(int)

        # 4. Total Porch Area
        X["TotalPorchSF"] = (
            X["OpenPorchSF"].fillna(0)
            + X["EnclosedPorch"].fillna(0)
            + X["3SsnPorch"].fillna(0)
            + X["ScreenPorch"].fillna(0)
        )

        # 5. Binary Indicators for High-Value Amenities
        X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)
        X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
        X["HasBsmt"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"].fillna(0) > 0).astype(int)

        return X


def build_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """
    Builds a complete preprocessing pipeline that applies:
    1. Custom Feature Engineering
    2. Ordinal Encoding for ordered quality features
    3. Imputation + RobustScaling for numerical features
    4. Imputation + OneHotEncoding for nominal categorical features
    """
    # Apply feature engineering temporarily to identify final column names
    temp_X = HouseFeatureEngineer().transform(X_train)

    # 1. Define Ordinal Columns & Their Exact Order
    qual_order = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
    qual_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC"
    ]
    # Keep only columns present in dataset
    qual_cols = [c for c in qual_cols if c in temp_X.columns]

    # 2. Identify Numerical and Remaining Nominal Categorical Columns
    num_cols = temp_X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [
        c for c in temp_X.select_dtypes(exclude=[np.number]).columns
        if c not in qual_cols
    ]

    # Define sub-pipelines
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[qual_order] * len(qual_cols),
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine all column transformers
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("ord", ordinal_pipeline, qual_cols),
            ("nom", nominal_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    # Full Preprocessing Pipeline
    full_preprocessor = Pipeline(steps=[
        ("feature_engineering", HouseFeatureEngineer()),
        ("preprocessing", column_transformer)
    ])

    return full_preprocessor


if __name__ == "__main__":
    # Quick sanity check to ensure pipeline fits and transforms cleanly
    from src.config import PROCESSED_DIR

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Pipeline Test Successful!")
    print(f"Original features: {X_train.shape[1]} -> Processed features: {X_train_processed.shape[1]}")
    print(f"Any NaNs in X_train_processed? {np.isnan(X_train_processed).sum()}")
    print(f"Any NaNs in X_test_processed? {np.isnan(X_test_processed).sum()}")

