import streamlit as st
import joblib


st.set_page_config(page_title="Email Spam Detection", page_icon="☠️")

model = joblib.load("sentiment_pipeline.pkl")


st.title("This is Email spam Detection")


user_input = st.text_area("Write your preview")

if st.button("Predict"):
    if user_input.strip() != "":
        predict = model.predict([user_input])[0]
        sentiment = "It's Not Spam" if predict == 0 else "It's Spam"
        
        if sentiment == "It's Spam":
            st.warning(f"Prediction: {sentiment}")
        else:
            st.success(f"Prediction: {sentiment}")
    else:
        st.warning("Please write something")