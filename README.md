# 🏁 Formula 1 Lap Time Predictor

An interactive web app that predicts a Formula 1 driver's lap time based on tyre compound, stint number, air/track temperature, and rainfall conditions. Built using FastF1, scikit-learn, and Streamlit.

---

## 📦 Features

- Predicts lap time based on user input
- Live visual feedback via dynamic plots
- Circuit image for the selected race
- Powered by real 2023 race data (British GP)

---

## 🛠️ Tech Stack

- `FastF1` – Data loading from real race sessions
- `scikit-learn` – ML pipeline + Random Forest model
- `pandas` – Data cleaning and manipulation
- `Streamlit` – Web UI
- `matplotlib` – Visual plots

---



---

## ▶️ How to Run

```bash
git clone https://github.com/yourusername/f1-lap-time-predictor.git
cd f1-lap-time-predictor
pip install -r requirements.txt
streamlit run app.py


🚀 Future Ideas
Add support for multiple circuits
Deploy on Streamlit Cloud
Include telemetry (speed, throttle, braking)
Predict pit stop strategies