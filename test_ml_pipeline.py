"""
Basic tests for the Diabetes Triage API
"""

import json
import numpy as np
from sklearn.datasets import load_diabetes


def test_data_loading():
    """Test that diabetes dataset loads correctly."""
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]

    assert X.shape[1] == 10, f"Expected 10 features, got {X.shape[1]}"
    assert len(X) == len(y), "Features and target must have same length"
    assert list(X.columns) == ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    print("✓ Data loading test passed")


def test_model_training():
    """Test that model training works."""
    from train import load_data, train_model, evaluate_model
    from sklearn.model_selection import train_test_split

    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    pipeline = train_model(X_train, y_train)

    # Test prediction
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(y_test), "Predictions must match test set size"
    assert all(np.isfinite(predictions)), "All predictions must be finite"

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] > 0, "RMSE must be positive"

    print("✓ Model training test passed")


def test_model_serialization():
    """Test that model can be saved and loaded."""
    import joblib
    from train import load_data, train_model
    from sklearn.model_selection import train_test_split

    # Train a model
    X, y = load_data()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = train_model(X_train, y_train)

    # Save and load
    joblib.dump(pipeline, "test_model.pkl")
    loaded_pipeline = joblib.load("test_model.pkl")

    # Test that loaded model works
    test_sample = np.array([[0.02, -0.044, 0.06, -0.03, -0.02, 0.03, -0.02, 0.02, 0.02, -0.001]])
    original_pred = pipeline.predict(test_sample)
    loaded_pred = loaded_pipeline.predict(test_sample)

    assert np.allclose(original_pred, loaded_pred), "Loaded model must give same predictions"

    # Clean up
    import os
    os.remove("test_model.pkl")

    print("✓ Model serialization test passed")


if __name__ == "__main__":
    test_data_loading()
    test_model_training()
    test_model_serialization()
    print("All tests passed! ✓")