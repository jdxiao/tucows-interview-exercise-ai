import requests
import json

# Modify the ticket text as needed
TICKET_TEXT = "My domain was suspended. How do I reactivate it?"

url = "http://localhost:8000/resolve-ticket"
payload = {
    "ticket_text": TICKET_TEXT
    }

response = requests.post(url, json=payload)
data = response.json()

# Print the JSON response in a compact format
print(json.dumps(data, separators=(',', ':')))