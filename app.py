from flask import Flask, request, render_template
from joblib import load
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the pre-trained model
model = load('House.pkl')

# Define the dropdown options
dropdown_options = {
    'location': ['Ikeja', 'Lekki', 'Victoria Island'],
    'currency': ['NGN', 'USD', 'EUR'],
    'neighborhood': ['GRA Ikeja', 'Phase 1', 'Central']
}

# Function to prepare input data for prediction
def prepare_input_data(bedrooms, bathrooms, toilets, serviced, newly_built, furnished, location, currency, neighborhood):
    # Create DataFrame with the form data
    input_data = pd.DataFrame([[bedrooms, bathrooms, toilets, serviced, newly_built, furnished, location, currency, neighborhood]],
                              columns=['Bedrooms', 'Bathrooms', 'Toilets', 'Serviced', 'Newly Built', 'Furnished', 'Location', 'Currency', 'Neighborhood'])

    # Convert categorical columns to categorical dtype
    input_data[['Location', 'Currency', 'Neighborhood']] = input_data[['Location', 'Currency', 'Neighborhood']].astype('category')

    # Create an encoder instance
    encoder = OneHotEncoder(drop='first')

    # Fit the encoder on the dropdown options
    encoder.fit(pd.DataFrame(dropdown_options))

    # Transform the categorical features
    loc_encoded = encoder.transform(input_data[['Location']])
    cur_encoded = encoder.transform(input_data[['Currency']])
    neigh_encoded = encoder.transform(input_data[['Neighborhood']])

    # Get feature names from encoder categories
    loc_cols = encoder.get_feature_names_out(['Location'])
    cur_cols = encoder.get_feature_names_out(['Currency'])
    neigh_cols = encoder.get_feature_names_out(['Neighborhood'])

    # Concatenate encoded categorical features
    encoded_cols = pd.DataFrame(loc_encoded.toarray(), columns=loc_cols)
    encoded_cols = pd.concat([encoded_cols, pd.DataFrame(cur_encoded.toarray(), columns=cur_cols)], axis=1)
    encoded_cols = pd.concat([encoded_cols, pd.DataFrame(neigh_encoded.toarray(), columns=neigh_cols)], axis=1)

    # Combine encoded categorical columns with numeric columns
    input_data_processed = pd.concat([input_data[['Bedrooms', 'Bathrooms', 'Toilets', 'Serviced', 'Newly Built', 'Furnished']], encoded_cols], axis=1)

    return input_data_processed

@app.route('/')
def home():
    return render_template('index.html', options=dropdown_options)

@app.route('/predict_result', methods=['POST'])
def predict_result():
    try:
        # Get the data from the form
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        toilets = float(request.form['toilets'])
        serviced = float(request.form['serviced'])
        newly_built = float(request.form['newly_built'])
        furnished = float(request.form['furnished'])
        location = request.form['location']
        currency = request.form['currency']
        neighborhood = request.form['neighborhood']

        # Prepare input data for prediction
        input_data = prepare_input_data(bedrooms, bathrooms, toilets, serviced, newly_built, furnished,
                                        location, currency, neighborhood)

        # Make prediction using the model
        prediction = model.predict(input_data)

        # Render the predict.html template with the prediction result
        return render_template('predict.html', prediction=prediction)

    except Exception as e:
        error_message = "Error occurred during prediction: " + str(e)
        return render_template('index.html', error=error_message, options=dropdown_options)

if __name__ == "__main__":
    app.run(debug=True)
