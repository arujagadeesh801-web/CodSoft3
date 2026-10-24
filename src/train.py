from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from src.preprocessing import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_model(
    data_path=PROJECT_ROOT / "data" / "Churn_Modelling.csv",
    model_type="random_forest",
    model_save_path=PROJECT_ROOT / "models" / "model.pkl",
    scaler_save_path=PROJECT_ROOT / "models" / "scaler.pkl",
):
    X_train, y_train, X_test, y_test = load_and_preprocess_data(data_path)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training classifier ({model_type})...")
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type specified.")

    model.fit(X_train_scaled, y_train)

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)

    print("Saving trained model and scaler...")
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print("Training pipeline finished successfully!")

    return model, scaler, X_test_scaled, y_test