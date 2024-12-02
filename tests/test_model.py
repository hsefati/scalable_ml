import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from app.ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def mock_data():
    """
    Creates mock data for testing the train_model function.
    Returns:
        X_train: np.array, features
        y_train: np.array, labels
    """
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )
    return X, y


def test_train_model(mock_data):
    """
    Tests the train_model function.
    """
    X_train, y_train = mock_data

    # Call the train_model function
    model = train_model(X_train, y_train, n_splits=3)

    # Check if the returned model is an instance of RandomForestClassifier
    assert isinstance(
        model, RandomForestClassifier
    ), "The model should be a RandomForestClassifier."

    # Check if the model can make predictions
    predictions = model.predict(X_train)
    assert len(predictions) == len(
        y_train
    ), "The model should make predictions for the input data."


@pytest.fixture
def mock_labels_and_predictions():
    """
    Creates mock labels and predictions for testing.
    Returns:
        y: np.array, true labels
        preds: np.array, predicted labels
    """
    y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    preds = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    return y, preds


def test_compute_model_metrics(mock_labels_and_predictions):
    """
    Tests the compute_model_metrics function without using sklearn metrics.
    """
    y, preds = mock_labels_and_predictions

    # Call the compute_model_metrics function
    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Manually calculate precision, recall, and F-beta
    true_positive = np.sum((y == 1) & (preds == 1))
    false_positive = np.sum((y == 0) & (preds == 1))
    false_negative = np.sum((y == 1) & (preds == 0))

    expected_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 1
    )
    expected_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 1
    )
    expected_fbeta = (
        (1 + 1**2)
        * (expected_precision * expected_recall)
        / (1**2 * expected_precision + expected_recall)
        if (expected_precision + expected_recall) > 0
        else 0
    )

    # Assertions
    assert isinstance(precision, float), "Precision should be a float."
    assert isinstance(recall, float), "Recall should be a float."
    assert isinstance(fbeta, float), "F-beta should be a float."

    # Ensure the metrics are within valid ranges
    assert 0 <= precision <= 1, "Precision should be between 0 and 1."
    assert 0 <= recall <= 1, "Recall should be between 0 and 1."
    assert 0 <= fbeta <= 1, "F-beta should be between 0 and 1."

    assert precision == pytest.approx(
        expected_precision
    ), f"Expected precision {expected_precision}, got {precision}."
    assert recall == pytest.approx(
        expected_recall
    ), f"Expected recall {expected_recall}, got {recall}."
    assert fbeta == pytest.approx(
        expected_fbeta
    ), f"Expected F-beta {expected_fbeta}, got {fbeta}."


@pytest.fixture
def mock_model_and_data():
    """
    Creates a mock trained model and test data for inference.
    Returns:
        model: Trained RandomForestClassifier
        X: np.array, test data
        y: np.array, true labels for validation
    """
    # Generate synthetic data
    from sklearn.datasets import make_classification

    X_train, y_train = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )

    X_test, y_test = make_classification(
        n_samples=10, n_features=10, n_classes=2, random_state=24
    )

    # Train a simple RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_inference(mock_model_and_data):
    """
    Tests the inference function.
    """
    model, X_test, y_test = mock_model_and_data

    # Run inference
    preds = inference(model, X_test)

    # Assertions
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array."
    assert (
        len(preds) == len(X_test)
    ), f"Number of predictions ({len(preds)}) should match the number of test samples ({len(X_test)})."
