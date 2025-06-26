from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'real_estate_model.pkl')
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['longitude'],
        data['latitude'],
        data['housing_median_age'],
        data['total_rooms'],
        data['total_bedrooms'],
        data['population'],
        data['households'],
        data['median_income'],
        data['ocean_proximity']
    ]
    prediction = model.predict([features])[0]
    return jsonify({'predicted_price': round(float(prediction), 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)