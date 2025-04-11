import requests

url = "http://127.0.0.1:5000/predict"

sample_input = {
    "Contract": "Two year",
    "tenure": 40,
    "InternetService": "Fiber optic",
    "MonthlyCharges": 95.5,
    "PaymentMethod": "Electronic check",
    "TechSupport": "Yes",
    "StreamingTV": "No"
}

response = requests.post(url, json=sample_input)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
