import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and expected feature columns
model = joblib.load('random_forest_credit_risk_model.pkl')
model_features = joblib.load('model_features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from form
        input_data = {
            'person_age': int(request.form['person_age']),
            'person_income': float(request.form['person_income']),
            'person_home_ownership': request.form['person_home_ownership'],
            'person_emp_length': float(request.form['person_emp_length']),
            'loan_intent': request.form['loan_intent'],
            'loan_grade': request.form['loan_grade'],
            'loan_amnt': float(request.form['loan_amnt']),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'cb_person_cred_hist_length': int(request.form['cb_person_cred_hist_length']),
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables like during training
        input_encoded = pd.get_dummies(input_df)

        # Align input features with model features
        for col in model_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_features]

        # Make prediction
        prediction = model.predict(input_encoded)[0]
        result = "Defaulter" if prediction == 1 else "Not a Defaulter"

        # Recommendation logic
        recommendation = ""
        if prediction == 1:
            recommendation = "⚠ You're at risk. Improve by: paying on time, reducing debts, and keeping utilization low."
        else:
            recommendation = "✅ You're in a safe zone. Keep managing your credit responsibly."

        return render_template('index.html', prediction=result, recommendation=recommendation)

    except Exception as e:
        return render_template('index.html', prediction="Error", recommendation=str(e))

if __name__ == '__main__':
    app.run(debug=True)