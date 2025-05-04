import streamlit as st
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_WORDS = 10000
MAX_LEN = 50

# Load tokenizer, label encoder, model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("emotion_model.h5")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction
def predict_emotion(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0]

# Streamlit UI
st.title("Emotion Detection from Text")
st.write("Enter a sentence and get the predicted emotion.")

user_input = st.text_area("Your sentence:", height=100)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        emotion = predict_emotion(user_input)
        st.success(f"**Predicted Emotion:** {emotion}")
