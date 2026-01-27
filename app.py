import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="CardioGuard", page_icon="🫀", layout="wide")

st.title("🫀 CardioGuard")
st.subheader("Interpretable AI for Cardiovascular Disease Risk Prediction")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("heart_processed.csv")

df = load_data()

# 🔴 CHANGED HERE
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)

# Train model
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
else:
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# Feature importance
st.markdown("## 🔍 Risk Factor Importance")

if model_choice == "Logistic Regression":
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Impact": model.coef_[0]
    }).sort_values(by="Impact", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=coef_df, x="Impact", y="Feature", ax=ax)
    st.pyplot(fig)

else:
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

# Patient input
st.markdown("## 🧪 Patient Risk Prediction")
user_input = {}

cols = st.columns(3)
for i, feature in enumerate(X.columns):
    user_input[feature] = cols[i % 3].number_input(
        feature,
        float(X[feature].min()),
        float(X[feature].max())
    )

input_df = pd.DataFrame([user_input])

if st.button("🔮 Predict Risk"):
    if model_choice == "Logistic Regression":
        risk = model.predict_proba(scaler.transform(input_df))[0][1]
    else:
        risk = model.predict_proba(input_df)[0][1]

    st.success(f"Predicted CVD Risk: **{risk*100:.2f}%**")

    if risk > 0.7:
        st.error("⚠️ High Risk – Medical attention recommended")
    elif risk > 0.4:
        st.warning("⚠️ Moderate Risk – Lifestyle changes advised")
    else:
        st.info("✅ Low Risk – Maintain healthy habits")

st.caption("CardioGuard | Hackathon Project | Explainable Medical AI")
