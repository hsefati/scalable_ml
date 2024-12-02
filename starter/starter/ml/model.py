import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold


def train_model(X_train, y_train, n_splits=5):
    """
    Trains a machine learning model using K-Fold Cross-Validation.

    Parameters:
    ----------
    X_train: np.array
        Preprocessed training features.
    y_train: np.array
        Training labels.
    n_splits: int
        Number of splits for K-Fold Cross-Validation.

    Returns:
    -------
    model: Trained machine learning model.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    for train_index, val_index in kf.split(X_train):
        X_kf_train, X_kf_val = X_train[train_index], X_train[val_index]
        y_kf_train, y_kf_val = y_train[train_index], y_train[val_index]

        model.fit(X_kf_train, y_kf_train)

        preds = model.predict(X_kf_val)
        fbeta = fbeta_score(y_kf_val, preds, beta=1, zero_division=1)
        scores.append(fbeta)

    print(f"K-Fold Cross-Validation Scores: {scores}")
    print(f"Mean F-Beta Score: {np.mean(scores):.2f}")

    # Refit the model on the full training data
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Parameters:
    ----------
    y: np.array
        Known labels, binarized.
    preds: np.array
        Predicted labels, binarized.

    Returns:
    -------
    precision: float
    recall: float
    fbeta: float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Parameters:
    ----------
    model: Trained machine learning model.
    X: np.array
        Data used for prediction.

    Returns:
    -------
    preds: np.array
        Predictions from the model.
    """
    return model.predict(X)


def evaluate_slices(y_true, preds, X_test, feature, output_file="slice_metrics.json"):
    """
    Evaluates the performance of the model on slices of the data based on a specific feature.

    Parameters:
    ----------
    y_true: np.array
        True labels from the test set.
    preds: np.array
        Predicted labels from the model for the test set.
    X_test: pd.DataFrame
        Feature dataset corresponding to the test set.
    feature: str
        Feature column name to slice the data on.

    Returns:
    -------
    slice_metrics: dict
        Dictionary containing performance metrics for each slice of the data.
    """
    unique_values = X_test[feature].unique()  # Get unique values for the feature
    slice_metrics = {}

    for value in unique_values:
        # Filter the test set for the current slice
        slice_indices = X_test[feature] == value
        y_slice = y_true[slice_indices]
        preds_slice = preds[slice_indices]

        # Compute classification metrics for this slice
        report = classification_report(y_slice, preds_slice, output_dict=True)

        # Store the metrics for this slice
        slice_metrics[value] = {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1-score": report["weighted avg"]["f1-score"],
            "support": len(y_slice),
        }

    # Save the slice metrics to a JSON file
    with open(output_file, "w") as f:
        json.dump(slice_metrics, f, indent=4)

    print(f"Slice metrics saved to {output_file}")

    return slice_metrics
