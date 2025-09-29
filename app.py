import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load saved models, scaler, and encoder
rf_model = pickle.load(open("injury_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("position_encoder.pkl", "rb"))

# Load dataset to calculate accuracy
df = pd.read_csv("ml mini data football.csv")
X = df.drop("Injury_Next_Season", axis=1)
y = df["Injury_Next_Season"]

# Safe encoding for Position
def safe_transform_position(pos_series, le):
    mapped = []
    for val in pos_series:
        if val in le.classes_:
            mapped.append(le.transform([val])[0])
        else:
            mapped.append(-1)  # unknown positions mapped to -1
    return np.array(mapped)

X["Position"] = safe_transform_position(X["Position"], le)

# Scale features
X[X.columns] = scaler.transform(X[X.columns])

# Train-test split for accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Accuracy of models
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

# Streamlit UI
st.set_page_config(page_title="Football Injury Prediction", page_icon="⚽")
st.title("🏈 University Football Injury Prediction")
st.markdown("Fill the player details below to predict the risk of injury for the next season.")

# Input fields
age = st.number_input("Age", 15, 40, 20)
height = st.number_input("Height (cm)", 150, 210, 175)
weight = st.number_input("Weight (kg)", 40, 120, 70)
position = st.selectbox("Position", le.classes_)
training_hours = st.number_input("Training Hours/Week", 0.0, 20.0, 10.0)
matches = st.number_input("Matches Played Last Season", 0, 100, 10)
previous_injuries = st.number_input("Previous Injury Count", 0, 10, 1)
knee_strength = st.slider("Knee Strength Score", 0.0, 100.0, 75.0)
hamstring_flex = st.slider("Hamstring Flexibility", 0.0, 100.0, 80.0)
reaction_time = st.number_input("Reaction Time (ms)", 150.0, 500.0, 250.0)
balance = st.slider("Balance Test Score", 0.0, 100.0, 85.0)
sprint_speed = st.number_input("Sprint Speed (10m/s)", 4.0, 8.0, 6.0)
agility = st.slider("Agility Score", 0.0, 100.0, 80.0)
sleep = st.number_input("Sleep Hours/Night", 4.0, 12.0, 8.0)
stress = st.slider("Stress Level Score", 0.0, 100.0, 50.0)
nutrition = st.slider("Nutrition Quality Score", 0.0, 100.0, 70.0)
warmup = st.selectbox("Warmup Routine Adherence", [0, 1])
bmi = st.number_input("BMI", 15.0, 40.0, 22.0)

if st.button("Predict Injury Risk"):
    # Encode position
    position_encoded = le.transform([position])[0]

    # Arrange input in same order as training dataset
    input_data = np.array([[age, height, weight, position_encoded,
                            training_hours, matches, previous_injuries,
                            knee_strength, hamstring_flex, reaction_time,
                            balance, sprint_speed, agility, sleep,
                            stress, nutrition, warmup, bmi]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predictions
    pred_rf = rf_model.predict(input_scaled)[0]
    pred_lr = lr_model.predict(input_scaled)[0]
    pred_xgb = xgb_model.predict(input_scaled)[0]
    
    st.subheader("🔍 Predictions & Accuracies")
    
    # Random Forest
    if pred_rf == 1:
        st.error(f"🌲 Random Forest: ⚠️ High Risk of Injury")
    else:
        st.success(f"🌲 Random Forest: ✅ Low Risk of Injury")
    st.caption(f"Accuracy: {rf_acc:.2f}")
    
    # Logistic Regression
    if pred_lr == 1:
        st.error(f"📉 Logistic Regression: ⚠️ High Risk of Injury")
    else:
        st.success(f"📉 Logistic Regression: ✅ Low Risk of Injury")
    st.caption(f"Accuracy: {lr_acc:.2f}")
    
    # XGBoost
    if pred_xgb == 1:
        st.error(f"🔥 XGBoost: ⚠️ High Risk of Injury")
    else:
        st.success(f"🔥 XGBoost: ✅ Low Risk of Injury")
    st.caption(f"Accuracy: {xgb_acc:.2f}")

