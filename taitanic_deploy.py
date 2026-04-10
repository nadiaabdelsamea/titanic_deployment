import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction")
st.write("أدخل بيانات الراكب لمعرفة هل كان سينجو أم لا!")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)

sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict Survival 🔍"):
    input_dict = {
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    }

    input_df = pd.DataFrame(input_dict)

    input_df = pd.get_dummies(input_df, columns=["Sex", "Embarked"])

    model_cols = model.feature_names_in_
    for col in model_cols:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[model_cols]

    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("✅ الناكب نجا (Survived)")
    else:
        st.error("❌ للأسف، الراكب لم ينجُ (Not Survived)")
