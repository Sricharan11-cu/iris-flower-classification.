
import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Iris AI Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classification")
st.write("AI-powered flower prediction using Python and Streamlit.")

@st.cache_resource
def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target
    )
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, iris.target_names, accuracy

model, names, accuracy = train_model()

st.success(f"AI model trained! Test accuracy: {accuracy:.2%}")
st.subheader("Enter flower measurements (cm)")

sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width", 0.1, 2.5, 0.2)

if st.button("Predict Flower", type="primary"):
    features = [[sepal_length, sepal_width,
                 petal_length, petal_width]]

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    st.subheader("Prediction Result")
    st.success(f"🌺 Species: {names[prediction].title()}")

    chart_data = pd.DataFrame({
        "Species": [n.title() for n in names],
        "Probability (%)": probabilities * 100
    }).set_index("Species")

    st.bar_chart(chart_data)
    st.write("Input measurements:", features[0])

st.caption("AIP Project | K-Nearest Neighbors (KNN)")
