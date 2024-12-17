import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the dataset
data = pd.read_csv("C:\\Users\\HP\\Downloads\\Financial_inclusion_dataset.csv")

def run_streamlit_app():
    """Create and run the Streamlit web application"""
    st.title('Financial Inclusion Prediction')

    # Load the saved model
    with open('C:\\Users\\HP\\Downloads\\random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Create LabelEncoders for categorical features
    encoders = {
        'country': LabelEncoder().fit(data['country']),
        'bank_account': LabelEncoder().fit(data['bank_account']),
        'location_type': LabelEncoder().fit(data['location_type']),
        'cellphone_access': LabelEncoder().fit(data['cellphone_access']),
        'gender_of_respondent': LabelEncoder().fit(data['gender_of_respondent']),
        'relationship_with_head': LabelEncoder().fit(data['relationship_with_head']),
        'marital_status': LabelEncoder().fit(data['marital_status']),
        'education_level': LabelEncoder().fit(data['education_level']),
        'job_type': LabelEncoder().fit(data['job_type']),
    }

    # Input fields for features
    st.header('Enter Feature Values')
    country = st.selectbox('COUNTRY', data['country'].unique())
    bank_account = st.selectbox('BANK_ACCOUNT', data['bank_account'].unique())
    location_type = st.selectbox('LOCATION_TYPE', data['location_type'].unique())
    cellphone_access = st.selectbox('CELLPHONE_ACCESS', data['cellphone_access'].unique())
    gender_of_respondent = st.selectbox('GENDER_OF_RESPONDENT', data['gender_of_respondent'].unique())
    relationship_with_head = st.selectbox('RELATIONSHIP_WITH_HEAD', data['relationship_with_head'].unique())
    marital_status = st.selectbox('MARITAL_STATUS', data['marital_status'].unique())
    education_level = st.selectbox('EDUCATION_LEVEL', data['education_level'].unique())
    job_type = st.selectbox('JOB_TYPE', data['job_type'].unique())
    uniqueid = st.number_input('UNIQUEID', min_value=0.0)
    household_size = st.number_input('HOUSEHOLD_SIZE', min_value=0.0)
    age_of_respondent = st.number_input('AGE_OF_RESPONDENT', min_value=0.0)
    year = st.number_input('YEAR', min_value=0.0)

    if st.button('Predict'):
        # Encode the categorical inputs
        country_encoded = encoders['country'].transform([country])[0]
        bank_account_encoded = encoders['bank_account'].transform([bank_account])[0]
        location_type_encoded = encoders['location_type'].transform([location_type])[0]
        cellphone_access_encoded = encoders['cellphone_access'].transform([cellphone_access])[0]
        gender_of_respondent_encoded = encoders['gender_of_respondent'].transform([gender_of_respondent])[0]
        relationship_with_head_encoded = encoders['relationship_with_head'].transform([relationship_with_head])[0]
        marital_status_encoded = encoders['marital_status'].transform([marital_status])[0]
        education_level_encoded = encoders['education_level'].transform([education_level])[0]
        job_type_encoded = encoders['job_type'].transform([job_type])[0]

        # Construct input features array
        input_features = np.array([[
            country_encoded, year, uniqueid,
            location_type_encoded, cellphone_access_encoded,
            household_size, age_of_respondent, gender_of_respondent_encoded,
            relationship_with_head_encoded, marital_status_encoded,
            education_level_encoded, job_type_encoded
        ]])

        # Make prediction
        prediction = model.predict(input_features)

        # Display results
        st.header('Prediction Results')
        if prediction[0] == 1:
            st.error('⚠ High Risk of Prediction')
        else:
            st.success('✅ Low Risk of Prediction')

# Run the Streamlit app
if __name__ == '__main__':
    run_streamlit_app()