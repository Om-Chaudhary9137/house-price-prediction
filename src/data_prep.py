import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import RAW_TRAIN_PATH, PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, TARGET_COL


def clean_raw_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Performs leakage-free, deterministic row-level cleaning before train-test split.
    """
    df = df.copy()

    # 1. Drop 'Id' column as it has no predictive power
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # 2. Remove duplicates if any exist
    df = df.drop_duplicates()

    # 3. Fix Wrong Data Types: Numeric codes that represent categories
    # MSSubClass (building class) and MoSold (month sold) are categorical
    categorical_as_num = ["MSSubClass", "MoSold"]
    for col in categorical_as_num:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 4. Domain-Specific Missing Values ("NA" means "Feature Absent", NOT missing data)
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType"
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numerical columns where NA means 0 (e.g., No Garage -> 0 GarageArea)
    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 5. Remove Extreme Outliers (ONLY on training data, never on test/inference data)
    if is_train and "GrLivArea" in df.columns and TARGET_COL in df.columns:
        outlier_condition = (df["GrLivArea"] > 4000) & (df[TARGET_COL] < 300000)
        print(f"Removing {outlier_condition.sum()} extreme GrLivArea outliers...")
        df = df[~outlier_condition].reset_index(drop=True)

    return df


def prepare_and_split_data():
    """
    Loads raw data, applies pre-split cleaning, transforms target,
    and performs the Leakage-Firewall Train-Test Split.
    """
    print("Loading raw training data...")
    raw_df = pd.read_csv(RAW_TRAIN_PATH)

    # Step 4: Pre-split cleaning
    cleaned_df = clean_raw_data(raw_df, is_train=True)

    # Separate Features (X) and Target (y)
    X = cleaned_df.drop(columns=[TARGET_COL])
    # Log-transform target variable as decided in EDA
    y = np.log1p(cleaned_df[TARGET_COL])

    # Step 5: ★ TRAIN-TEST SPLIT ★ (Leakage Firewall)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Save processed splits to disk
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print(f"Data Split Complete & Saved!")
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

    # Verify remaining true missing values in X_train
    remaining_missing = X_train.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    print("\nTrue missing values left for Step 6 (Pipeline Imputation):")
    print(remaining_missing)


if __name__ == "__main__":
    prepare_and_split_data()
