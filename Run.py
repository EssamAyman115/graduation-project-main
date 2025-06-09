import requests

url = 'http://127.0.0.1:5000/chat'  
payload = {
    "title": "الصداع",
    "question": "علاج الصداع النصفي؟"
}

response = requests.post(url, json=payload)
print(response.json())


