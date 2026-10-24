from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_and_preprocess_data(file_path=PROJECT_ROOT / "data" / "Churn_Modelling.csv"):
    print("Loading churn dataset...")
    df = pd.read_csv(file_path)

    target_column = next(
        (column for column in ("Churn", "Exited") if column in df.columns),
        None,
    )
    if target_column is None:
        raise ValueError(
            "Dataset must contain a target column named 'Churn' or 'Exited'."
        )

    y = df.pop(target_column)
    if y.isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")

    if y.dtype == object or pd.api.types.is_string_dtype(y.dtype):
        normalized_target = y.astype("string").str.strip().str.lower()
        if normalized_target.isin(["yes", "no"]).all():
            y = normalized_target.map({"yes": 1, "no": 0})
        else:
            y = pd.to_numeric(y, errors="coerce")
            if y.isna().any():
                raise ValueError(
                    f"Target column '{target_column}' must contain numeric values "
                    "or 'Yes'/'No'."
                )

    # Remove identifiers that do not represent customer behavior.
    identifier_columns = ["customerID", "CustomerId", "RowNumber", "Surname"]
    df = df.drop(columns=identifier_columns, errors="ignore")

    # Handle TotalCharges if it contains whitespace strings
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, y_train, X_test, y_test