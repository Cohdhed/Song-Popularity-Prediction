import pandas as pd
import numpy as np
import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os


#Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

#Load preprocessor
with open('preprocessor.pkl', 'rb') as p_file:
    preprocessor = pickle.load(p_file)


# DATA INPUT PAGE
st.title('Song Popularity Prediction App')

# Numeric Input for Song duration
song_duration = st.number_input("Song duration(minute)")

# Numeric Input for Acouticness
acousticness = st.number_input("Acousticness")

# Numeric Input for Danceabilty
danceability = st.number_input("Danceability")

# Numeric Input for Energy
energy = st.number_input("Energy")

# Numeric Input for Instrumentalness
instrumentalness = st.number_input("Instrumentalness")

# Dropdown for Key
key_options = ['', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
key = st.selectbox("Key", key_options)

# Numeric Input for Liveness
liveness = st.number_input("Liveness")

# Numeric Input for Loudness
loudness = st.number_input("Loudness")

# Dropdown for Audio Mode
audio_mode_options = ['', "Major", "Minor"]
audio_mode = st.selectbox("Audio mode", audio_mode_options)
# Convert audio_mode to numerical values
audio_mode_mapping = {'Major': 1, 'Minor': 0}
audio_mode_numeric = audio_mode_mapping.get(audio_mode, None)

speechiness = st.number_input("Speechiness")

# Numeric Input for Tempo
tempo = st.number_input("Tempo")

# Dropdown for Time signature
time_sig_options = ['', 2,3,4,5]
time_signature = st.selectbox("Time signature", time_sig_options)

# Numeric Input for Audio Valence
audio_valence = st.number_input("Audio Valence")



user_input = {
        'song_duration_minutes': song_duration,
        'acousticness': acousticness,
        'danceability': danceability,
        'energy': energy,
        'instrumentalness': instrumentalness,
        'key': key,
        'liveness': liveness,
        'loudness': loudness,
        'audio_mode': audio_mode_numeric,
        'speechiness': speechiness,
        'tempo': tempo,
        'time_signature': time_signature,
        'audio_valence': audio_valence,

    }


# Function to check missing inputs
def check_missing_inputs(data):
    return any(value == '' for value in data.values())


# Function to make predictions
def predict(data):
    df = pd.DataFrame([data])
    
    # Apply the preprocessor to preprocess the new data
    new_data_preprocessed = preprocessor.transform(df)
    # Convert back to DataFrame with feature names
    new_data_preprocessed_df = pd.DataFrame(new_data_preprocessed, columns=preprocessor.get_feature_names_out())

    # Use the preprocessed data as input to the model for predictions
    predictions = model.predict_proba(new_data_preprocessed_df)[:, 1][0].item()
    decision = 'Low Popularity' if predictions < 0.5 else 'High Popularity'
    return f"Probability Of Popularity: {predictions:.2f} - {decision}"


# Button to make prediction
if st.button('Predict Popularity'):
    if check_missing_inputs(user_input):
        st.error("Please fill in all the fields.")
    else:
        result = predict(user_input)
        st.success(result)
