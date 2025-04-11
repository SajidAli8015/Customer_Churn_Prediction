from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import joblib

app = Flask(__name__)

# Load the trained LightGBM model
model_path = os.path.join(os.path.dirname(__file__), 'final_lightgbm_model.pkl')
model = joblib.load(model_path)

# Required raw fields to be sent in POST request
required_fields = [
    'Contract', 'tenure', 'InternetService',
    'MonthlyCharges', 'PaymentMethod',
    'TechSupport', 'StreamingTV'
]

# Preprocessing logic based on training phase (SHAP-selected features)
def preprocess_input(raw_input):
    df = pd.DataFrame([raw_input])

    # Feature engineering
    df['is_long_term_contract'] = df['Contract'].apply(lambda x: 1 if x in ['One year', 'Two year'] else 0)
    df['Contract_Two year'] = df['Contract'].apply(lambda x: 1 if x == 'Two year' else 0)
    df['InternetService_Fiber optic'] = df['InternetService'].apply(lambda x: 1 if x == 'Fiber optic' else 0)
    df['PaymentMethod_Electronic check'] = df['PaymentMethod'].apply(lambda x: 1 if x == 'Electronic check' else 0)
    df['is_tech_dependent'] = df.apply(
        lambda row: 1 if (row['TechSupport'] == 'Yes' or row['StreamingTV'] == 'Yes') else 0,
        axis=1
    )

    selected_features = [
        'is_long_term_contract',
        'tenure',
        'InternetService_Fiber optic',
        'MonthlyCharges',
        'Contract_Two year',
        'PaymentMethod_Electronic check',
        'is_tech_dependent'
    ]

    return df[selected_features]

@app.route('/')
def home():
    return "✅ Welcome to the Telecom Churn Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        # Validate input
        missing = [field for field in required_fields if field not in input_data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

        # Preprocess
        processed = preprocess_input(input_data)

        # Predict
        prob = model.predict_proba(processed)[:, 1][0]
        return jsonify({'churn_probability': round(float(prob), 4)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
