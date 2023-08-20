# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model


# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://plus.unsplash.com/premium_photo-1682309676673-392c56015c5c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80')
    st.title('Lip Reading')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet using StreamLit ✌🏻') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('👀 This is all the machine learning model sees when making a prediction')
        video, annotations,image_data = load_data(tf.convert_to_tensor(file_path))
        # st.text(video.shape)
        imageio.mimsave('animation.gif',np.squeeze((video * 50).astype(np.uint8)) , duration=100)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
