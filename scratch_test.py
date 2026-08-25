import requests

def test_api():
    with open("test_sample.csv", "rb") as f:
        r = requests.post("http://127.0.0.1:8000/api/v1/analyze", files={"file": f})
    print("Status code:", r.status_code)
    data = r.json()
    print("Response keys:", list(data.keys()))
    print("Session ID:", data.get("session_id"))
    print("Metrics sample:", {k: data.get("metrics", {}).get(k) for k in ["rows", "cols", "accuracy", "silhouette_score"]})

if __name__ == "__main__":
    test_api()
