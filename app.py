from flask import Flask, request, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model, encoder, and feature names
model = load('House.pkl')
encoder = load('encoder.pkl')
feature_names = load('feature_names.pkl')

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

    # Transform the categorical features
    encoded_cols = encoder.transform(input_data[['Location', 'Currency', 'Neighborhood']])
    encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['Location', 'Currency', 'Neighborhood']))

    # Combine encoded categorical columns with numeric columns
    input_data_processed = pd.concat([input_data[['Bedrooms', 'Bathrooms', 'Toilets', 'Serviced', 'Newly Built', 'Furnished']], encoded_cols_df], axis=1)

    # Ensure the columns match the training columns
    input_data_processed = input_data_processed.reindex(columns=feature_names, fill_value=0)

    return input_data_processed

# Define a custom Jinja filter for formatting floats
@app.template_filter('float_format')
def float_format(value):
    return "{:.2f}".format(value)

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
        input_data = prepare_input_data(bedrooms, bathrooms, toilets, serviced, newly_built, furnished, location, currency, neighborhood)

        # Make prediction using the model
        prediction = model.predict(input_data)

        # Modify the prediction by adding a zero to the back
        prediction = prediction[0]

        # Render the predict.html template with the prediction result
        return render_template('predict.html', prediction=prediction)

    except Exception as e:
        error_message = "Error occurred during prediction: " + str(e)
        return render_template('index.html', error=error_message, options=dropdown_options)

if __name__ == "__main__":
    app.run(debug=True)
