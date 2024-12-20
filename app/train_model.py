# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split

from app.ml.data import process_data
from app.ml.model import compute_model_metrics, evaluate_slices, inference, train_model

# Define the save path
abs_file_dir_path = Path(__file__).resolve().parent
save_path = Path("model")
model_file = abs_file_dir_path / save_path / "model.pkl"
encoder_file = abs_file_dir_path / save_path / "encoder.pkl"
lb_file = abs_file_dir_path / save_path / "label_binarizer.pkl"

# Load data
data = pd.read_csv(abs_file_dir_path / "data/census.csv")

# Remove empty space from column names from CSV files
data.columns = data.columns.str.strip()

train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
# Check if all files exist
if model_file.exists() and encoder_file.exists() and lb_file.exists():
    print("Model, encoder, and label binarizer exist. Loading...")
    model = load(model_file)
    encoder = load(encoder_file)
    lb = load(lb_file)
else:
    print("Models do not exist, training the models")
    model = train_model(X_train, y_train, n_splits=5)

    # Define the save path
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Save the model, encoder, and label binarizer
    dump(model, model_file)
    dump(encoder, encoder_file)
    dump(lb, lb_file)
    print("Model, encoder, and label binarizer saved.")

# Run inference on the test set
preds = inference(model, X_test)

# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Test F1 Score: {fbeta:.2f}")


# iterate through the categorical features and save results to log and txt file
slice_metrics_dfs = []
for feature in cat_features:
    slice_df = evaluate_slices(y_test, preds, test, feature)
    slice_df["feature"] = feature
    slice_metrics_dfs.append(slice_df)

# Concatenate all slice DataFrames
combined_df = pd.concat(slice_metrics_dfs, ignore_index=True)
# Save the combined DataFrame to a CSV file
output_file = "slice_output.txt"
combined_df.to_csv(output_file, index=False)
