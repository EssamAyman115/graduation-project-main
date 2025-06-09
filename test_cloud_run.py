import requests

# Replace this URL with your actual Cloud Run service URL after deployment
# Format: https://medical-qa-app-XXXXXXXXXX-uc.a.run.app
CLOUD_RUN_URL = 'https://your-service-url-here.run.app/chat'

# For local testing, use:
LOCAL_URL = 'http://127.0.0.1:8080/chat'

# Choose which URL to use
url = LOCAL_URL  # Change to CLOUD_RUN_URL after deployment

payload = {
    "title": "الصداع",
    "question": "علاج الصداع النصفي؟"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.ConnectionError:
    print("Connection error. Make sure the server is running.")
except Exception as e:
    print(f"Error: {e}") 