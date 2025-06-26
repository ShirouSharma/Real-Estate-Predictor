import requests

data = {
    "longitude": -122.25,
    "latitude": 37.85,
    "housing_median_age": 52,
    "total_rooms": 1256,
    "total_bedrooms": 190,
    "population": 600,
    "households": 180,
    "median_income": 3.5,
    "ocean_proximity": 1
}

response = requests.post('http://localhost:5000/predict', json=data)
print("Status:", response.status_code)
print("Prediction:", response.json())