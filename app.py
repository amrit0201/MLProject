# app.py (updated with predict_proba)

import streamlit as st
import joblib
import pandas as pd

# Load model pipeline (preprocessor + model)
pipeline = joblib.load('treatment_prediction_pipeline.pkl')

st.title(" Mental Health Treatment Prediction App (Tech Survey)")

st.markdown("Fill in your details based on the Mental Health in Tech Survey.")

# Input fields
age = st.slider("How old are you?", min_value=15, max_value=80, value=30)
country = st.selectbox("What country do you live in?", ['United States', 'India', 'Canada', 'Other'])
gender = st.selectbox("What is your gender?", ['Male', 'Female', 'Other'])

ordinal_inputs = {
    'family_history': st.selectbox("Do you have a family history of mental illness?", ['No', 'Yes']),
    'work_interfere': st.selectbox(
        "If you have a mental health condition, do you feel that it interferes with your work?",
        ['unknown', 'Never', 'Rarely', 'Sometimes', 'Often']
    ),
    'benefits': st.selectbox("Does your employer provide mental health benefits?", ['No', "Don't know", 'Yes']),
    'care_options': st.selectbox("Do you know the options for mental health care your employer provides?", ['No', "Not sure", 'Yes']),
    'wellness_program': st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?", ['No', "Don't know", 'Yes']),
    'seek_help': st.selectbox("Does your employer provide resources to learn more about mental health issues and how to seek help?", ['No', "Don't know", 'Yes']),
    'anonymity': st.selectbox("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?", ['No', "Don't know", 'Yes']),
    'leave': st.selectbox("How easy is it for you to take medical leave for a mental health condition?", ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"]),
    'mental_health_consequence': st.selectbox("Do you think that discussing a mental health issue with your employer would have negative consequences?", ['No', 'Maybe', 'Yes']),
    'coworkers': st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?", ['No', 'Some of them', 'Yes']),
    'mental_health_interview': st.selectbox("Would you bring up a mental health issue with a potential employer in an interview?", ['No', 'Maybe', 'Yes']),
    'mental_vs_physical': st.selectbox("Do you feel that your employer takes mental health as seriously as physical health?", ['No', "Don't know", 'Yes']),
    'obs_consequence': st.selectbox("Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?", ['No', 'Yes']),
}

# Prepare input dataframe
input_data = {
    'Age': [age],
    'Country': [country],
    'Gender': [gender],
    **{key: [val] for key, val in ordinal_inputs.items()}
}
input_df = pd.DataFrame(input_data)

# Prediction & Probability
if st.button("Predict Treatment"):
    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]  # Probability for class '1' (treatment sought)
    st.success(f"Prediction: {' Likely to seek treatment' if prediction == 1 else ' Not likely to seek treatment'}")
    st.info(f"Probability of seeking treatment: {proba*100:.2f}%")
