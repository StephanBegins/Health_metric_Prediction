import streamlit as st
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define a function to create HTML with CSS for transitions and rounded rectangles
def transition_html(risk_level):
    # Determine which risk level to show prominently
    low_class = "show" if risk_level == "Low risk of diabetes" else "hidden"
    high_class = "show" if risk_level == "High risk of diabetes" else "hidden"
    
    html_code = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&family=Open+Sans:wght@400&display=swap');
    
    .container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 300px;
        position: relative;
        background-color: #121212; /* Dark background color */
        color: white;
    }}
    .box {{
        position: absolute;
        top: 50%;
        width: 45%;
        height: 100px;
        font-size: 24px;
        text-align: center;
        line-height: 100px;
        border-radius: 15px;
        transition: transform 1s ease, opacity 1s ease;
        opacity: 0;
    }}
    .low {{
        left: 10%;
        background-color: #2e7d32; /* Dark green for low risk */
        color: white;
        font-family: 'Roboto', sans-serif; /* Font for low risk */
        font-weight: bold;
    }}
    .high {{
        right: 10%;
        background-color: #d32f2f; /* Dark red for high risk */
        color: white;
        font-family: 'Open Sans', sans-serif; /* Font for high risk */
    }}
    .show {{
        transform: translateX(0);
        opacity: 1;
    }}
    .hidden {{
        transform: translateX(0);
        opacity: 0.5;
        font-size: 18px;
        text-decoration: line-through;
    }}
    .low.hidden {{
        background-color: #1b5e20; /* Shaded dark green */
    }}
    .high.hidden {{
        background-color: #c62828; /* Shaded dark red */
    }}
    </style>
    <div class="container">
        <div class="box low {low_class}">Low risk of diabetes</div>
        <div class="box high {high_class}">High risk of diabetes</div>
    </div>
    """
    return html_code

# Streamlit app
st.sidebar.title('Diabetes Risk Prediction')
pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0)
glucose = st.sidebar.number_input('Plasma Glucose Concentration', min_value=0)
bp = st.sidebar.number_input('Diastolic Blood Pressure', min_value=0)
skin_thickness = st.sidebar.number_input('Triceps Skin Fold Thickness', min_value=0)
insulin = st.sidebar.number_input('2-Hour Serum Insulin', min_value=0)
bmi = st.sidebar.number_input('Body Mass Index', min_value=0)
diabetes_pedigree = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.sidebar.number_input('Age', min_value=0)

# Add a 'Proceed' button
if st.sidebar.button('Proceed'):
    # Create a DataFrame with the same column names used during training
    features_df = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })

    # Scale the features
    features_scaled = scaler.transform(features_df)

    # Predict
    prediction = model.predict(features_scaled)
    risk_level = "Low risk of diabetes" if prediction[0] == 0 else "High risk of diabetes"

    # Display result with transition effect
    st.markdown(transition_html(risk_level), unsafe_allow_html=True)
