import streamlit as st
import pandas as pd
import numpy as np

# Carregando o dataset
data_model = pd.read_csv("data_num.csv")
x = data_model.drop('crop_label', axis=1)
y = data_model['crop_label']

# Separando os dados de treino e teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Treinando o modelo de Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rforest = RandomForestClassifier()
model = rforest.fit(x_train, y_train)

# Criando o app
with st.sidebar:
    st.image("Brainize Tech (1).png", width=250)
    st.title("Crop Recommendation")
    choice = st.radio("**Navigation:**", ("About", "Prediction", "Batch Prediction"))
    st.info('**Note:** Please be aware that this application is intended solely for educational purposes. It is strongly advised against utilizing this tool for making any real life decisions.')

if choice == "About":
    st.write("""
        # Crop Recommendation
        This app predicts the best crop to be cultivated by an agricultural company!
        """)
    st.write('---')
    st.write('**About the App:**')
    st.write(
        'Utilizing a Random Forest Clasified augorithm, the aforementioned approach employs a meticulously trained model encompassing 8 distinct features. Its primary objective is to predict the best crop for an agricultural company.')
    st.info(
        '**Note:** Please be aware that this application is intended solely for educational purposes. It is strongly advised against utilizing this tool for making any real life decisions.')
    st.write('---')
    st.write('**About the Data:**')
    st.write(
        'Precision agriculture is in trend nowadays. It helps the farmers to get informed decision about the farming strategy. Here, we present to you a dataset which would allow the users to build a predictive model to recommend the most suitable crops to grow in a particular farm based on various parameters.'
        'For more information, please visit the [**DataSet**](https://www.kaggle.com/datasets/agriinnovate/agricultural-crop-dataset)')
    st.write('---')

if choice == "Prediction":
    nitrogen = st.number_input('Nitrogen levels', min_value=0, max_value=1000, value=0)
    phosphor = st.number_input('Phosphor levels', min_value=0, max_value=1000, value=0)
    potassium = st.number_input('Potassium levels', min_value=0, max_value=1000, value=0)
    temperature = st.number_input('Temperature (C°)', min_value=0, max_value=1000, value=0)
    humidity = st.number_input('Humidity (g.m³)', min_value=0, max_value=1000, value=0)
    col1, col2 = st.columns(2)
    with col1:
        ph = st.number_input('Soild pH', min_value=0, max_value=14, value=0)
    with col2:
        rainfall = st.number_input('Rainfall (mm)', min_value=0, max_value=300, value=0)

    if st.button('Predict'):
        prediction = model.predict([[nitrogen, phosphor, potassium, temperature, humidity, ph, rainfall]])
        st.write('The best crop for the given conditions is **{}**'.format(prediction[0]))

        st.success('The demand for the cement company in the given period is {}'.format(prediction[0]))

if choice == "Batch Prediction":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_pred = pd.read_csv(uploaded_file)
        st.write('---')
        st.write('**Dataset:**')
        st.write(df_pred)
        if st.button("Predict"):
            # Predict the probability of churn using the model and create new column for the besst crop
            df_pred['crop_suggestion'] = model.predict(df_pred)
            # Create a success message
            st.success('The best crop for the given conditions was predicted successfully!')
            # Show the dataframe
            st.write(df_pred)
            # Create a button to download the dataset with the predicted probability of bankruptcy
            if st.download_button(label='Download Predicted Dataset', data=df_pred.to_csv(index=False),
                                  file_name='predicted.csv', mime='text/csv'):
                pass
        else:
            st.write('---')
            st.info('Click the button to predict the best crop for the given codition!')

st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')