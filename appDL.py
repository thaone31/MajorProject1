import tensorflow as tf
import streamlit as st
import DLmodel.models.CNN as PhoBert_CNN
import MLmodel.PhoBERTWE.model.Logistic as PhoBert_Logistic


import DLmodel.preprocess_data as preprocess_data 
from DLmodel.preprocess_data import vocab
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import base64

# @st.experimental_memo
# def getImgAsBase64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # Tạo phần tử video và thiết lập làm nền
# video_path =  getImgAsBase64("background-video.mp4")

# video_html = f"""
# <video autoplay muted loop style="position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%;">
#     <source src="data:video:{video_path}" type="video/mp4">
# </video>
# """


# # Sử dụng HTML để chèn phần tử video vào ứng dụng
# st.markdown(video_html, unsafe_allow_html=True)

# Set up the Streamlit app
st.title("Hotel Sentiment Analysis")

# User input area
review_title = st.text_input("Review Title")
review_text = st.text_area("Review Text")

# Preprocess input
title_ids, text_ids = preprocess_data.preprocess_single_sentence(review_title, review_text)

tensor_title = tf.convert_to_tensor(title_ids)
tensor_text = tf.convert_to_tensor(text_ids)

# Model selection
model_choice = st.selectbox("Select Model", ["Deep Learning Model - CNN", "Machine Learning Model - Logistic Regression"])
# Predict button
if st.button("Predict"):
    if model_choice == "Deep Learning Model - CNN":
        output = PhoBert_CNN.predict([tensor_title, tensor_text])
        if isinstance(output, tuple) and len(output) == 3:
            # If the output is a tuple with 3 elements
            sizes = np.array(list(output))

    elif model_choice == "Machine Learning Model - Logistic Regression":
        output = PhoBert_Logistic.predict([tensor_title, tensor_text])
        if isinstance(output, tuple) and len(output) == 3:
            # If the output is a tuple with 3 elements
            sizes = np.array(list(output))

      
    percentages_list = output.tolist()
    percentages = percentages_list[0]

    print("percentage 5: ", percentages)

    # Tạo biểu đồ tròn
    st.plotly_chart(px.pie(names=['Negative', 'Neutral', 'Positive'], values=percentages, 
                        color_discrete_sequence=['#b9fbc0', '#f1c0e8 ', '#fbf8cc' ]))

    