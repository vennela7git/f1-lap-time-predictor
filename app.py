import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import load_lap_data, train_model

st.set_page_config(page_title="F1 Lap Time Predictor", layout="centered")

# Title and intro
st.title("🏁 Formula 1 Lap Time Predictor")
st.markdown("Built with FastF1, scikit-learn, and Streamlit")

# Load and train
with st.spinner("Training model..."):
    df = load_lap_data()
    min_lap_time = df['LapTime'].min()
    max_lap_time = df['LapTime'].max()
    
    model, X_test, y_test = train_model(df)

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("🎛️ Race Conditions")

driver = st.sidebar.selectbox("🏎️ Driver", df['Driver'].unique())
compound = st.sidebar.selectbox("🛞 Tyre Compound", df['Compound'].unique())
stint = st.sidebar.slider("🔁 Stint Number", int(df['Stint'].min()), int(df['Stint'].max()), 1)
air_temp = st.sidebar.slider("🌡️ Air Temperature (°C)", float(df['AirTemp'].min()), float(df['AirTemp'].max()), 26.0)
track_temp = st.sidebar.slider("🔥 Track Temperature (°C)", float(df['TrackTemp'].min()), float(df['TrackTemp'].max()), 35.0)
rainfall = st.sidebar.radio("🌧️ Rainfall (mm)", sorted(df['Rainfall'].unique()))

# Input DataFrame
input_df = pd.DataFrame([{
    "Driver": driver,
    "Compound": compound,
    "Stint": stint,
    "AirTemp": air_temp,
    "TrackTemp": track_temp,
    "Rainfall": rainfall
}])

# Prediction
predicted_time = model.predict(input_df)[0]

st.markdown("### 🧠 Predicted Lap Time")
st.success(f"⏱️ **{predicted_time:.2f} seconds**")

st.markdown("---")

# predicted vs actual plot
st.markdown("### 📊 Model Validation: Predicted vs Actual Lap Times")
y_pred = model.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.7, color = 'orange', label='Validation Data')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')

ax.scatter(predicted_time, predicted_time, color = 'red', marker='^', s= 200, label='Your Prediction')
ax.set_xlabel('Actual Lap Time (s)')
ax.set_ylabel('Predicted Lap Time (s)')
ax.set_title('Prediction accuracy')
ax.legend()
st.pyplot(fig)

st.markdown("---")

# Add a visual reaction to live inputs
st.markdown("### 🧪 Prediction Summary")

fig2, ax2 = plt.subplots(figsize=(6, 1))
ax2.barh(['Current Lap'], [predicted_time], color='green')
ax2.set_xlim(min_lap_time, max_lap_time)  # full lap time range for context
ax2.set_xlabel("Lap Time (s)")
st.pyplot(fig2)


st.markdown("---")

# 🗺️ Track Map
st.markdown("### 🗺️ Circuit: Silverstone (British GP)")
st.image("Silverstone.png", caption="Silverstone Circuit", use_container_width=True)

st.markdown("---")
st.caption("⚙️ Model: Random Forest | Data: 2023 race sessions | Built with ❤️ by Vennela")