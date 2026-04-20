import requests
import os

url = "http://127.0.0.1:8000/predict"
img_path = "data/raw/banana_black_sigatoka/scraped_000000.jpg"

if not os.path.exists(img_path):
    print(f"File not found: {img_path}")
else:
    with open(img_path, "rb") as f:
        files = {"files": ("test.jpg", f, "image/jpeg")}
        r = requests.post(url, files=files)
        print("Status Code:", r.status_code)
        print("Response:", r.json())
