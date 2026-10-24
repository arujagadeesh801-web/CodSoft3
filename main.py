import argparse
from pathlib import Path

from src.train import train_model
from src.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "Churn_Modelling.csv"),
        help="Path to the churn dataset file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["logistic", "random_forest", "gradient_boosting"],
        help="Choose classifier type"
    )

    args = parser.parse_args()

    # Run training
    model, scaler, X_test, y_test = train_model(
        data_path=args.data_path,
        model_type=args.model_type
    )

    # Run evaluation
    evaluate_model(X_test=X_test, y_test=y_test)