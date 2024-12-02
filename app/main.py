from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

from app.ml.data import process_data

# Define the save path
abs_file_dir_path = Path(__file__).resolve().parent
save_path = Path("model")
model_file = abs_file_dir_path / save_path / "model.pkl"
encoder_file = abs_file_dir_path / save_path / "encoder.pkl"
lb_file = abs_file_dir_path / save_path / "label_binarizer.pkl"


# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 189123,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Sales",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 4500,
                "capital_loss": 0,
                "hours_per_week": 60,
                "native_country": "Canada",
            }
        }


# instantiate FastAPI app
app = FastAPI(
    title="Inference API",
    description="An API that takes a sample and runs an inference",
    version="1.0.0",
)


# load model artifacts on startup of the application to reduce latency
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoder, lb

    try:
        if model_file.exists() and encoder_file.exists() and lb_file.exists():
            model = load(model_file.open("rb"))
            encoder = load(encoder_file.open("rb"))
            lb = load(lb_file.open("rb"))
        else:
            missing_files = []
            if not model_file.exists():
                missing_files.append(str(model_file))
            if not encoder_file.exists():
                missing_files.append(str(encoder_file))
            if not lb_file.exists():
                missing_files.append(str(lb_file))
            raise FileNotFoundError(
                f"Missing required files: {', '.join(missing_files)}"
            )
    except Exception as e:
        raise e


@app.get("/")
async def greetings():
    return "Welcome to our model API"


@app.post("/inference/")
async def ingest_data(inference: InputData):
    data = {
        "age": inference.age,
        "workclass": inference.workclass,
        "fnlgt": inference.fnlgt,
        "education": inference.education,
        "education-num": inference.education_num,
        "marital-status": inference.marital_status,
        "occupation": inference.occupation,
        "relationship": inference.relationship,
        "race": inference.race,
        "sex": inference.sex,
        "capital-gain": inference.capital_gain,
        "capital-loss": inference.capital_loss,
        "hours-per-week": inference.hours_per_week,
        "native-country": inference.native_country,
    }

    # prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
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

    # if saved model exits, load the model from disk
    if model_file.exists() and encoder_file.exists() and lb_file.exists():
        model = load(model_file.open("rb"))
        encoder = load(encoder_file.open("rb"))
        lb = load(lb_file.open("rb"))
    else:
        missing_files = []
        if not model_file.exists():
            missing_files.append(str(model_file))
        if not encoder_file.exists():
            missing_files.append(str(encoder_file))
        if not lb_file.exists():
            missing_files.append(str(lb_file))
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    sample, _, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # get model prediction which is a one-dim array like [1]
    prediction = model.predict(sample)

    # convert prediction to label and add to data output
    if prediction[0] > 0.5:
        prediction = ">50K"
    else:
        prediction = ("<=50K",)
    data["prediction"] = prediction

    return data


if __name__ == "__main__":
    pass
