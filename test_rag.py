import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_upload():
    print("Uploading sample_lease.pdf...")
    with open("sample_lease.pdf", "rb") as f:
        files = {"file": ("sample_lease.pdf", f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/upload", files=files)
        
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_chat(question):
    print(f"\nAsking question: '{question}'...")
    payload = {"question": question}
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    if test_upload():
        test_chat("What is the pet policy at this property?")
        test_chat("Who is responsible for major HVAC repairs?")
