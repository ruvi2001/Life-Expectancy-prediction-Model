import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the saved scaler and model
scaler = joblib.load('artifacts/scaler.pkl')
best_rf_model = joblib.load('artifacts/final_best_rf_model_with_top_12_features.pkl')

# Load the country mean target mapping (previously saved as 'country_mean_target_mapping.csv')
country_mean_target_mapping = pd.read_csv('artifacts/country_mean_target_mapping.csv', index_col=0)

# Load the list of countries
with open("artifacts/countries.json", "r") as f:
    countries = json.load(f)['countries']

# Function to predict life expectancy
def preprocess_and_predict(user_input):
    country = user_input.get('Country')
    if country in country_mean_target_mapping.index:
        country_mean_target = country_mean_target_mapping.loc[country].values[0]
    else:
        raise ValueError(f"Country '{country}' not found in the mapping!")

    # Convert inputs to appropriate format
    year = int(user_input.get('Year'))
    adult_mortality = float(user_input.get('Adult Mortality'))
    income_composition = float(user_input.get('Income composition of resources'))
    hiv_aids = float(user_input.get('HIV/AIDS'))
    schooling = float(user_input.get('Schooling'))
    bmi = float(user_input.get('BMI'))
    measles = float(user_input.get('Measles '))
    thinness_1_19 = float(user_input.get('thinness 1-19 years'))
    total_expenditure = float(user_input.get('Total expenditure'))
    thinness_5_9 = float(user_input.get('thinness 5-9 years'))
    gdp = float(user_input.get('GDP'))
    population = float(user_input.get('Population'))

    # Compute GDP per capita
    gdp_per_capita = gdp / (population + 1)

    # Preprocess input data
    input_data = {
        'Country_Mean_Target': country_mean_target,
        'Year': year,
        'Adult Mortality': adult_mortality,
        'Income composition of resources': income_composition,
        'HIV/AIDS': hiv_aids,
        'Schooling': schooling,
        'BMI': bmi,
        'Measles ': measles,
        'thinness 1-19 years': thinness_1_19,
        'Total expenditure': total_expenditure,
        'thinness 5-9 years': thinness_5_9,
        'GDP_per_capita': gdp_per_capita
    }

    input_df = pd.DataFrame([input_data])

    # Standardize the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    predicted_life_expectancy = best_rf_model.predict(input_scaled)
    return predicted_life_expectancy[0]

# Streamlit App
st.title("Life Expectancy Prediction")

# Create a form for user input
with st.form(key="life_expectancy_form"):
    # Dropdown for selecting country
    country = st.selectbox("Country", ["Select a country"] + countries, key="Country")
    year = st.slider("Year", 2000, 2024, 2013, key="Year")

    # Split the form into two columns
    col1, col2 = st.columns(2)

    # Column 1 inputs
    with col1:
        adult_mortality = st.number_input("Adult Mortality", min_value=0.0, step=1.0, key="Adult Mortality")
        income_composition = st.number_input("Income composition of resources", min_value=0.0, max_value=1.0, step=0.01, key="Income composition of resources")
        hiv_aids = st.number_input("HIV/AIDS", min_value=0.0, step=0.01, key="HIV/AIDS")
        schooling = st.number_input("Schooling", min_value=0.0, step=0.01, key="Schooling")
        bmi = st.number_input("BMI", min_value=0.0, step=0.01, key="BMI")
        population = st.number_input("Population", min_value=0.0, step=1.0, key="Population")
    
    # Column 2 inputs
    with col2:
        measles = st.number_input("Measles", min_value=0.0, step=1.0, key="Measles ")
        thinness_1_19 = st.number_input("Thinness 1-19 years", min_value=0.0, step=0.01, key="thinness 1-19 years")
        total_expenditure = st.number_input("Total expenditure", min_value=0.0, step=0.01, key="Total expenditure")
        thinness_5_9 = st.number_input("Thinness 5-9 years", min_value=0.0, step=0.01, key="thinness 5-9 years")
        gdp = st.number_input("GDP", min_value=0.0, step=1.0, key="GDP")
        

    # Center the submit button
    col_center = st.columns([1, 1, 1])
    with col_center[1]:
        submit_button = st.form_submit_button(label="Predict Life Expectancy")

# Process form submission
if submit_button:
    if country == "Select a country":
        st.error("Please select a valid country.")
    else:
        # Gather user inputs into a dictionary
        user_input = {
            'Country': country,
            'Year': year,
            'Adult Mortality': adult_mortality,
            'Income composition of resources': income_composition,
            'HIV/AIDS': hiv_aids,
            'Schooling': schooling,
            'BMI': bmi,
            'Measles ': measles,
            'thinness 1-19 years': thinness_1_19,
            'Total expenditure': total_expenditure,
            'thinness 5-9 years': thinness_5_9,
            'GDP': gdp,
            'Population': population
        }

        # Get the prediction from the model
        try:
            life_expectancy = preprocess_and_predict(user_input)
            years = int(life_expectancy)
            months = round((life_expectancy - years) * 12)
            st.success(f"Predicted Life Expectancy: {years} years and {months} months")
        except ValueError as e:
            st.error(f"Error: {e}")
