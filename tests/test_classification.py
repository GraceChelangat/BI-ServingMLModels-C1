import pytest

from api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    return {
        "industry_sector": "Manufacturing",
        "listing_status": "Private",
        "credit_rating_band": "Investment Grade",
        "audit_going_concern_flag": "No",
        "debt_to_equity_ratio": 1.05,
        "current_ratio": 1.25,
        "interest_coverage_ratio": 4.50,
        "revenue_growth_pct": 5.20,
        "net_profit_margin_pct": 8.00,
    }


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.get_json()


def test_successful_classification(client, valid_payload):
    response = client.post(
        "/predict/classification",
        json=valid_payload,
    )

    assert response.status_code == 200

    result = response.get_json()

    assert result["predicted_category"] in {
        "Healthy",
        "Watch",
        "Distressed",
    }


def test_classification_returns_probabilities(client, valid_payload):
    response = client.post(
        "/predict/classification",
        json=valid_payload,
    )

    assert response.status_code == 200

    result = response.get_json()

    assert "class_probabilities" in result

    probabilities = result["class_probabilities"]

    assert set(probabilities.keys()) == {
        "Healthy",
        "Watch",
        "Distressed",
    }

    assert all(
        0 <= probability <= 1
        for probability in probabilities.values()
    )

    assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.01)


def test_missing_features(client):
    response = client.post(
        "/predict/classification",
        json={
            "industry_sector": "Manufacturing",
            "listing_status": "Private",
        },
    )

    assert response.status_code == 400

    result = response.get_json()

    assert "missing_features" in result
    assert "credit_rating_band" in result["missing_features"]


def test_invalid_numeric_input(client, valid_payload):
    valid_payload["debt_to_equity_ratio"] = "invalid"

    response = client.post(
        "/predict/classification",
        json=valid_payload,
    )

    assert response.status_code == 400

    result = response.get_json()

    assert "must be numeric" in result["error"]


def test_invalid_request_body(client):
    response = client.post(
        "/predict/classification",
        data="not valid json",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "error" in response.get_json()