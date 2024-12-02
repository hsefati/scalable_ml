from fastapi.testclient import TestClient
import json

from app.main import app

client = TestClient(app)


def test_greetings():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == '"Welcome to our model API"'


def test_inference_correct():
    sample = {
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

    data = json.dumps(sample)

    response = client.post("/inference/", data=data)

    # test response and output
    assert response.status_code == 200
    assert response.json()["age"] == 35
    assert response.json()["fnlgt"] == 189123
    assert response.json()["prediction"][0] == "<=50K"


def test_wrong_inference_query():
    sample = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234721,
    }

    data = json.dumps(sample)
    response = client.post("/inference/", data=data)

    assert response.status_code == 422
    assert "prediction" not in response.json().keys()
