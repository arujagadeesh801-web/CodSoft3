from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_model(model_path=PROJECT_ROOT / "models" / "model.pkl", X_test=None, y_test=None):
    print("Loading trained model for evaluation...")
    model = joblib.load(model_path)
    
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test, y_pred))
        
        if y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob)
            print(f"\nROC-AUC Score: {roc_auc:.4f}")
            
        print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")