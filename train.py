"""
Training for diabetes progression model and saving artifacts

Outputs:
- model.pkl: Trained pipeline (scaler + model)
- metrics.json: Test set performance metrics
- training_info.json: Metadata (timestamp, seed, sklearn version)
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RSEED = 42
np.random.seed(RSEED)

# Output paths
MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("metrics.json")
TRAINING_INFO_PATH = Path("training_info.json")


def load_data():
    """Load and prepare diabetes dataset."""
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]
    
    return X, y


def train_model(X_train, y_train):
    """
    Train a baseline model: LinearRegression.

    """
    
    # Setup pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)    
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate model on test set.
    Metrics (RMSE, MAE, R^2)
    """
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_test_samples": len(y_test)
    }
    
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  R²:   {metrics['r2']:.3f}")
    
    return metrics


def save_artifacts(pipeline, metrics):
    """Save artifacts."""
    
    # 1. Trained pipeline using joblib
    joblib.dump(pipeline, MODEL_PATH)
    
    # 2. Metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 3. Training metadata (for audit trail)
    training_info = {
        "model_version": "v0.1",
        "model_type": "StandardScaler + LinearRegression",
        "training_date": datetime.now().isoformat(),
        "random_seed": RSEED,
        "sklearn_version": sklearn.__version__
    }
    with open(TRAINING_INFO_PATH, "w") as f:
        json.dump(training_info, f, indent=2)


def main():
    """Main training pipeline."""

    # 1. Load data
    X, y = load_data()
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=RSEED
    )

    # 3. Train model
    pipeline = train_model(X_train, y_train)
    
    # 4. Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # 5. Save all
    save_artifacts(pipeline, metrics)



if __name__ == "__main__":
    main()