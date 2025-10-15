import pytest
import api


@pytest.fixture()
def app():
    app = api.app
    app.config.update(
        {
            "TESTING": True,
        }
    )

    yield app

@pytest.fixture()
def client(app):
    return app.test_client()

@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


def test_health(client):
    # assert {"model_type":"StandardScaler + LinearRegression","model_version":"v0.1","status":"ok"}
    response = client.get("/health")
    print(response.data)
    assert b"ok" in response.data
    assert response.status_code == 200

def test_metrics(client):
    # {"rmse": 56.4, "mae": 45.2, "r2": 0.45, ...}
    response = client.get("/metrics")
    print(response.data)
    assert b"mae" in response.data
    assert b"rmse" in response.data
    assert b"r2" in response.data

def test_predict(client):
    response = client.post("/predict", json = {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001})

    assert response.status_code == 200
    assert b"prediction" in response.data

    # Missing "age" should return 400 with an error message
    response = client.post("/predict", json = {
        "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02,
        "s4": 0.02, "s5": 0.02, "s6": -0.001
    })
    assert response.status_code == 400
    assert b"Missing required fields" in response.data

    # Invalid field name should also return 400, not raise a KeyError
    response = client.post("/predict", json = {
        "name": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02,
        "s4": 0.02, "s5": 0.02, "s6": -0.001
    })
    assert response.status_code == 400
    assert b"Missing required fields" in response.data