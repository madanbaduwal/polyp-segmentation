"""
    Polyp segmentation
    @Madan Baduwal
"""

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
# from predict import predict
# from unet import model



# import pretrained model
@st.cache_resource
def predict(img):
    model_path = "./models/best.pt"
    model = YOLO(model_path)
    results = model(img)
    return results


# model option
st.markdown("<h1 style='text-align: center;'>Polyp Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Madan Baduwal<sup>1</sup>, Kishor Karki<sup>2</sup>, Usha Shrestha<sup>3</sup>, Halimat Popoola<sup>4</sup>, Priyanka Kumar<sup>1</sup></h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Department of Computer Science, University of Texas Permian Basin</h5>", unsafe_allow_html=True)
st.markdown("<h9 style='text-align: center;'>(baduwal_m63609, karki_k65395, shrestha_u53095, popoola_h51572, kumar_p)@utpb.edu</h9>", unsafe_allow_html=True)



# file uploader
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'tif'])

col1, col2 = st.columns(2)
with col1:
    # image cache
    image_cache = st.container()
    if uploaded_file is not None:
        # convert image into np array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, 1)  # 1 means loading the image in color (BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        image_cache.subheader("Your uploaded file:")
        print(img_array)
        image_cache.image(img_array)

        # store in streamlit session
        st.session_state.img_array = img_array
    elif 'img_array' in st.session_state:
        img_array = st.session_state.img_array
        # display image
        image_cache.subheader("Your uploaded file:")
        print(img_array)
        image_cache.image(img_array)
    img_pred_button = st.button('Predict')

with col2:
    if img_pred_button:
        if "img_array" not in st.session_state:
            st.write("You haven't upload any file!")
        else:
            st.subheader("Model Prediction:")
            pred_img = st.session_state.img_array 
            results = predict(pred_img)
            
            
            for result in results:
                for j, mask in enumerate(result.masks.data):

                    mask = mask.cpu().numpy()
                    st.session_state.pred_mask = mask
                    st.image(mask)
            clear = st.button("clear prediction")

            if clear:
                del st.session_state.pred_maskl
                img_pred_button = False

# Citation
st.markdown("""

<p>Baduwal, Madan, et al.<i>Polyp Segmentation.</i>. 
""", unsafe_allow_html=True)

