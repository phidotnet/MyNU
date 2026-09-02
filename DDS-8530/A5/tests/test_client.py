import os
import json
import httpx

BASE_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8000")
PREDICTION_ENDPOINTS = [
    "/predict/random-forest-100",
    "/predict/random-forest-200",
    "/predict/gradient-boosting-100",
    "/predict/gradient-boosting-200",
]
RETAIL_FEATURES = {
    "age": 30,
    "zipcode": 90210,
    "year": 2025,
    "ratings": 4,
    "transaction_hour": 14,
    "income": "High",
    "customer_segment": "Premium",
    "product_category": "Electronics",
    "product_brand": "Samsung",
}


def test_prediction_endpoints() -> None:
    """Call each deployed model endpoint and verify it returns a prediction."""
    # Print the input in the POST request, format with new line in JSON content for readability
    print("Input for prediction:", json.dumps(RETAIL_FEATURES, indent=2))
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        for endpoint in PREDICTION_ENDPOINTS:
            response = client.post(endpoint, json=RETAIL_FEATURES)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result.get("prediction-average-purchase-value"), float):
                raise AssertionError(f"{endpoint} did not return a numeric prediction: {result}")
            print(f"{endpoint}: {result}")


if __name__ == "__main__":
    test_prediction_endpoints()
