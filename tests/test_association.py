import pytest

from api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_recommendation_by_name(client):
    response = client.post("/recommend/association", json={"items": ["Crime City", "The Detective"]})
    body = response.get_json()

    assert response.status_code == 200
    assert body["recommendations"], "expected at least one recommendation"
    assert body["recommendations"][0]["show_name"] == "Silent Witness Redux"
    # Already-watched shows are never recommended back.
    names = [r["show_name"] for r in body["recommendations"]]
    assert "Crime City" not in names and "The Detective" not in names
    # Ranked by confidence, highest first.
    confs = [r["confidence"] for r in body["recommendations"]]
    assert confs == sorted(confs, reverse=True)


def test_ids_and_names_give_same_result(client):
    a = client.post("/recommend/association", json={"items": ["Crime City", "The Detective"]}).get_json()
    b = client.post("/recommend/association", json={"items": ["s01", " S02 "]}).get_json()
    assert a == b


def test_top_n_limits_results(client):
    body = client.post("/recommend/association", json={"items": ["Crime City"], "top_n": 1}).get_json()
    assert len(body["recommendations"]) <= 1


def test_no_matching_rule(client):
    response = client.post("/recommend/association", json={"items": ["Toon Town"]})
    body = response.get_json()
    assert response.status_code == 200
    assert body["recommendations"] == []
    assert "message" in body


def test_unknown_show(client):
    response = client.post("/recommend/association", json={"items": ["Not A Real Show"]})
    assert response.status_code == 400
    assert response.get_json()["unknown_items"] == ["Not A Real Show"]


@pytest.mark.parametrize("payload", [{}, {"items": []}, {"items": "Crime City"}, {"items": ["Crime City"], "top_n": 0}])
def test_bad_requests(client, payload):
    assert client.post("/recommend/association", json=payload).status_code == 400


def test_non_json_body(client):
    response = client.post("/recommend/association", data="hello", content_type="text/plain")
    assert response.status_code == 400
