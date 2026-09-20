# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import os


# Get the folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# Load the pre-trained model with ReLU activation
model = load_model(
    os.path.join(BASE_DIR, 'simple_rnn_imdb.h5')
)

max_features = model.layers[0].input_dim


# Step 2: Helper Functions

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([
        reverse_word_index.get(i - 3, '?')
        for i in encoded_review
    ])


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    encoded_review = [
        index if index < max_features else 2
        for index in encoded_review
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


import streamlit as st

st.set_page_config(
    page_title='IMDB Movie Review Sentiment Analysis',
    page_icon='🎬',
    layout='centered',
)


st.markdown(
    '''
    <style>
    .block-container { max-width: 820px; padding-top: 3rem; padding-bottom: 3rem; }
    .app-subtitle { color: #5f6b7a; margin-top: -0.75rem; margin-bottom: 2rem; }
    .result-label { color: #5f6b7a; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }
    </style>
    ''',
    unsafe_allow_html=True,
)


st.title('IMDB Movie Review Sentiment Analysis')

st.markdown(
    '<p class="app-subtitle">Enter a movie review to classify it as positive or negative.</p>',
    unsafe_allow_html=True,
)


with st.container(border=True):

    st.subheader('Review input')

    user_input = st.text_area(
        'Movie Review',
        placeholder='Type or paste a movie review here...',
        height=180,
    )


    if st.button(
        'Classify Review',
        type='primary',
        use_container_width=True
    ):

        preprocessed_input = preprocess_text(user_input)

        ## MAke prediction
        prediction = model.predict(preprocessed_input)

        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'


        with st.container(border=True):

            st.markdown(
                '<div class="result-label">Prediction result</div>',
                unsafe_allow_html=True
            )

            st.subheader(sentiment)

            st.write(
                f'Prediction Score: {prediction[0][0]}'
            )

    else:

        st.info(
            'Your sentiment result will appear here after classification.'
        )
