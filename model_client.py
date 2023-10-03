import requests

text = "What time is it?"
print("Prompt: " + text)

response = requests.post("http://127.0.0.1:8000/", params={"text": text})
response_text = response.json()

print(response_text)
