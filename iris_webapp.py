import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title = "Iris Identification")
st.title("Iris Flower Classification")
st.image("iris_types.jpg", caption = "Iris Image")

def get_model():
    return joblib.load('iris.joblib')

spl = st.text_input("Enter Sepal Length: ", "")
spw = st.text_input("Enter Sepal Width: ", "")
pel = st.text_input("Enter Petal Length: ", "")
pew = st.text_input("Enter Petal Width: ", "")

if st.button('Classify'):
    values = [spl, spw, pel, pew]
    float_values = []
    for x in values:
        float_values.append(float(x))

    input_values = np.asanyarray(float_values).reshape(1, -1)

    mymodel = get_model()
    pred = mymodel.predict(input_values)
    pred = int(pred[0])

    if pred == 0:
        st.write("Your flower is Setosa")
    elif pred == 1:
        st.write("Your flower is Versicolor")
    elif pred == 2:
        st.write("Your flower is Virginica")