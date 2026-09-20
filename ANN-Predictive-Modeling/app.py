import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.set_page_config(page_title='Customer Churn Prediction', layout='centered')
st.markdown(
    '''
    <style>
        .block-container {max-width: 900px; padding-top: 2rem; padding-bottom: 3rem;}
        .stButton > button {font-weight: 600; min-height: 2.75rem;}
    </style>
    ''',
    unsafe_allow_html=True,
)

st.title('Customer Churn Prediction')
st.caption('Enter the customer details below to estimate churn probability.')

with st.container(border=True):
    st.subheader('Customer information')
    customer_col1, customer_col2, customer_col3 = st.columns(3)
    with customer_col1:
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    with customer_col2:
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
    with customer_col3:
        age = st.slider('Age', 18, 92)

with st.container(border=True):
    st.subheader('Financial information')
    financial_col1, financial_col2, financial_col3 = st.columns(3)
    with financial_col1:
        credit_score = st.number_input('Credit Score')
    with financial_col2:
        balance = st.number_input('Balance')
    with financial_col3:
        estimated_salary = st.number_input('Estimated Salary')

with st.container(border=True):
    st.subheader('Account information')
    account_col1, account_col2, account_col3, account_col4 = st.columns(4)
    with account_col1:
        tenure = st.slider('Tenure', 0, 10)
    with account_col2:
        num_of_products = st.slider('Number of Products', 1, 4)
    with account_col3:
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    with account_col4:
        is_active_member = st.selectbox('Is Active Member', [0, 1])

st.write('')
predict_clicked = st.button('Predict Churn', type='primary', use_container_width=True)

if predict_clicked:
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    with st.container(border=True):
        st.subheader('Prediction result')
        st.metric('Churn Probability', f'{prediction_proba:.2f}')
        if prediction_proba > 0.5:
            st.warning('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')
