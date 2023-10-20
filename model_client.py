import requests

text = "What is the current time, based on where you are"
print("Prompt: " + text)

response = requests.post("http://127.0.0.1:8000/", params={"text": text})
french_text = response.json()

print(french_text)
