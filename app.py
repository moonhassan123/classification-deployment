import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np



#Title
st.title("Image Classification")

#load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('modelvgg.h5')
    return model

with st.spinner("Loading Model...."):
    model=load_model()


# image preprocessing
def image_processing(inp_img):
    # img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    
    img_array = inp_img / 255
    new_array = cv2.resize(img_array, (416,416))
    return new_array.reshape(-1, 416, 416, 3)



# upload image from local system
file_img = st.file_uploader("Please choose an image file",type=['jpeg','jpg','PNG'])

class_dict = {'nCT': 0, 'pCT': 1}

def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction) == clss:
             if key == 'nCT':
                 return 'Negative CT'
             else:
                 return 'Positive CT'
             
if file_img is None:
    st.write("You haven't upload any image")
else:
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            # pathfile = file_img.name
            input_img = Image.open(file_img)
            prediction = model.predict(image_processing(np.array(input_img)))
            pred_final = prediction_cls(prediction)
            st.write("The image is classified as: ", pred_final)
            st.image(file_img, use_column_width=True)
            st.balloons()
        