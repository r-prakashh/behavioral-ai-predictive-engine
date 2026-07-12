import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('behavioral_model.pkl')

model = load_model()

st.set_page_config(page_title="Behavioral AI Engine", layout="centered", page_icon="🧠")
st.title("🧠 Behavioral AI: Intent-to-Action Predictor")
st.write("Adjust your daily micro-habits below to predict the probability of achieving your macro-goals over a 6-month timeline.")

st.markdown("---")
st.markdown("### 📊 Your Daily Micro-Habits")

# UI Sliders
sleep = st.slider("Hours of Sleep", 2.0, 12.0, 7.0, 0.5)
screen = st.slider("Entertainment Screen Time (Hours)", 0.0, 12.0, 4.0, 0.5)
focus = st.slider("Deep Work / Focus (Hours)", 0.0, 10.0, 2.0, 0.5)
water = st.slider("Water Intake (Liters)", 0.0, 5.0, 2.0, 0.5)

if st.button("Predict Goal Success Rate"):
    with st.spinner("Analyzing behavioral metrics..."):
        user_data = pd.DataFrame({
            'sleep_hours': [sleep],
            'screen_time': [screen],
            'focus_hours': [focus],
            'water_intake': [water]
        })
        
        # Calculate probability
        probability = model.predict_proba(user_data)[0][1] * 100
        
        st.markdown("---")
        st.subheader("🎯 AI Prediction")
        
        if probability > 70:
            st.success(f"**{probability:.1f}% Probability of Success** - Your habits are highly aligned with your goals.")
        elif probability > 40:
            st.warning(f"**{probability:.1f}% Probability of Success** - You are in the friction zone. Adjust habits to improve odds.")
        else:
            st.error(f"**{probability:.1f}% Probability of Success** - High risk of goal failure based on current behavioral metrics.")