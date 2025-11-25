from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved model and preprocessors
model = pickle.load(open('model.pkl', 'rb'))
standard_scaler = pickle.load(open('scalar.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))

# Features for display on features page
feature_names = [
    "location",
    "total_sqft",
    "bath",
    "bhk"
]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/zamanrashid_features')
def features():
    # Even if features.html is static, passing this does no harm
    return render_template("features.html", features=feature_names)

@app.route('/predict_price', methods=["POST"])
def predict_price():
    try:
        # Get form inputs
        location = request.form.get("location")
        total_sqft = request.form.get("total_sqft")
        bath = request.form.get("bath")
        bhk = request.form.get("bhk")

        # Convert numeric values
        total_sqft = float(total_sqft)
        bath = float(bath)
        bhk = float(bhk)

        # Numeric features dataframe
        numeric_df = pd.DataFrame(
            [[location,total_sqft, bath, bhk]],
            columns=["location","total_sqft", "bath", "bhk"]
        )

        # Location encoding using OHE
        
        loc_encoded = ohe.transform(numeric_df[["location"]])
        
        # Scale the data
        scaled_data = standard_scaler.transform(numeric_df.drop(columns=['location']))

        # Combine numeric + encoded categorical features
        X_test_transformed=np.concatenate((loc_encoded,scaled_data),axis=1)

        

        # Predict using  model
        result = model.predict(X_test_transformed)[0]

        # Round to 2 decimal places (lakhs)
        result = round(result, 2)

        return render_template('result.html', results=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
