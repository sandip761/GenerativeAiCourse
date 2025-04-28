import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data#it dies not load the data every time it loads the cache data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['specis'] = iris.target
    return df, iris.target_names

df, target_names = load_data()
model =RandomForestClassifier()
model.fit(df.iloc[:,:-1], df['specis'])
print(df)

sepal_length = st.sidebar.slider("Sepal legth", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_legth = st.sidebar.slider("Petal Length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))


input_data = [[sepal_length, sepal_width, petal_legth,  petal_width]]

##Prediction
predicion = model.predict(input_data)
predicted_species = target_names[predicion[0]]

st.write("Prediction")
st.write(f"The prediction species is: {predicted_species}")


