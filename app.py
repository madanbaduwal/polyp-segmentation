"""
    Polyp segmentation
    Madan Baduwal
"""

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# from predict import predict
# from unet import model



# import pretrained model
@st.cache_resource
def predict(img):
    model_path = "./models/best.pt"
    model = YOLO(model_path)
    results = model(img)
    print(f"Image tpye{results}")
    return results


# model option
st.markdown("<h1 style='text-align: center;'>DataScience: Polyp Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Department of Computer Science, University of Texas Permian Basin</h5>", unsafe_allow_html=True)
# st.markdown("<h5 style='text-align: center;'>baduwal_m63609@utpb.edu</h5>", unsafe_allow_html=True)



# file uploader
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'tif'])

col1, col2 = st.columns(2)
with col1:
    # image cache
    image_cache = st.container()
    if uploaded_file is not None:
        # convert image into np array
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img)

        # # check if the file is valid (3 * 256 * 256)
        # if img_array.shape != (256, 256, 3):
        #     st.write("Image size should be 256*256")
        # else:
        # display image
        image_cache.subheader("Your uploaded file:")
        image_cache.image(img_array)

        # store in streamlit session
        st.session_state.img_array = img_array
        img_array = img_array / 255
    elif 'img_array' in st.session_state:
        img_array = st.session_state.img_array
        # display image
        image_cache.subheader("Your uploaded file:")
        image_cache.image(img_array)
    img_pred_button = st.button('Predict')

with col2:
    if img_pred_button:
        if "img_array" not in st.session_state:
            st.write("You haven't upload any file!")
        else:
            # get predicted mask image
            st.subheader("Model Prediction:")
            pred_img = st.session_state.img_array / 255
            results = predict(pred_img)
            # pred_mask = pred_mask[0].permute(1, 2, 0)
            
            
            clear = st.button("clear prediction")
            for result in results:
                for j, mask in enumerate(result.masks.data):

                    mask = mask.numpy() * 255
                    st.session_state.pred_mask = mask
                    st.image(mask.numpy())

                # mask = cv2.resize(mask, (W, H))
            # clear prediction and content
            if clear:
                del st.session_state.pred_maskl
                img_pred_button = False