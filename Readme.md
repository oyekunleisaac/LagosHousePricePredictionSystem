# Lagos House Price Prediction Project

## Overview

This project predicts house prices in Lagos based on features such as the number of bedrooms, bathrooms, toilets, location, and other amenities. It utilizes a machine learning model (Random Forest Regressor) and is deployed as a web application using Flask.

## Table of Contents

1. [Languages and Tools Used](#languages-and-tools-used)
2. [Data Collection and Cleaning](#data-collection-and-cleaning)
3. [Feature Engineering](#feature-engineering)
4. [Model Building](#model-building)
5. [Model Deployment with Flask](#model-deployment-with-flask)
6. [Running the Project](#running-the-project)
7. [Conclusion](#conclusion)

## Languages and Tools Used

- **Python**: Data processing, machine learning, and server-side scripting
- **Flask**: Web framework for deploying the model
- **Jinja2**: Templating engine for rendering HTML
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning library
- **joblib**: Serialization library
- **HTML/CSS**: Web page creation and styling

## Data Collection and Cleaning

### Data Collection
The dataset includes:
- Bedrooms
- Bathrooms
- Toilets
- Location
- Currency
- Neighborhood
- Price
- Serviced
- Newly Built
- Furnished

### Data Cleaning
- **Convert to Numeric**: Ensured numeric columns are in the correct format.
- **Impute Missing Values**: Filled missing values using the median.
- **Categorical Encoding**: Converted categorical columns to appropriate formats and encoded them.

## Feature Engineering

Relevant features:
- Bedrooms
- Bathrooms
- Toilets
- Serviced
- Newly Built
- Furnished
- Encoded categorical features (Location, Currency, Neighborhood)

## Model Building

Using Random Forest Regressor:
1. Split data into training and testing sets.
2. Train the model on training data.
3. Save the trained model and encoders using `joblib`.

## Model Deployment with Flask

### Flask Application
1. Create a Flask application.
2. Define routes for home and prediction results.
3. Load the trained model and encoders.
4. Prepare input data for prediction.
5. Render prediction results on a web page.

### Flask Application Code

```python
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
    input_data = pd.DataFrame([[bedrooms, bathrooms, toilets, serviced, newly_built, furnished, location, currency, neighborhood]],
                              columns=['Bedrooms', 'Bathrooms', 'Toilets', 'Serviced', 'Newly Built', 'Furnished', 'Location', 'Currency', 'Neighborhood'])
    input_data[['Location', 'Currency', 'Neighborhood']] = input_data[['Location', 'Currency', 'Neighborhood']].astype('category')
    encoded_cols = encoder.transform(input_data[['Location', 'Currency', 'Neighborhood']])
    encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['Location', 'Currency', 'Neighborhood']))
    input_data_processed = pd.concat([input_data[['Bedrooms', 'Bathrooms', 'Toilets', 'Serviced', 'Newly Built', 'Furnished']], encoded_cols_df], axis=1)
    input_data_processed = input_data_processed.reindex(columns=feature_names, fill_value=0)
    return input_data_processed

@app.template_filter('float_format')
def float_format(value):
    return "{:.2f}".format(value)

@app.route('/')
def home():
    return render_template('index.html', options=dropdown_options)

@app.route('/predict_result', methods=['POST'])
def predict_result():
    try:
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        toilets = float(request.form['toilets'])
        serviced = float(request.form['serviced'])
        newly_built = float(request.form['newly_built'])
        furnished = float(request.form['furnished'])
        location = request.form['location']
        currency = request.form['currency']
        neighborhood = request.form['neighborhood']
        input_data = prepare_input_data(bedrooms, bathrooms, toilets, serviced, newly_built, furnished, location, currency, neighborhood)
        prediction = model.predict(input_data)
        prediction = prediction[0] * 10
        return render_template('predict.html', prediction=prediction)
    except Exception as e:
        error_message = "Error occurred during prediction: " + str(e)
        return render_template('index.html', error=error_message, options=dropdown_options)

if __name__ == "__main__":
    app.run(debug=True)
```

## Running the Project

1. **Install Dependencies**:
   ```bash
   pip install flask pandas scikit-learn joblib
   ```

2. **Prepare the Dataset**: Ensure your dataset is cleaned and preprocessed.

3. **Train the Model**: Use the code provided in the Model Building section to train the model and save it.

4. **Run the Flask Application**:
   ```bash
   python app.py
   ```

5. **Access the Application**: Open a web browser and navigate to `http://127.0.0.1:5000`.

## Conclusion

This project demonstrates how to predict house prices using machine learning and deploy the model as a web application using Flask. By following the steps outlined, you can replicate and extend the project based on your requirements.