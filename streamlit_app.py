import streamlit as st
import pickle
import tensorflow as tf
import os

with open('model_pic_clf.pkl', 'rb') as pkl_file:
    my_model = pickle.load(pkl_file)

path_to_zip = tf.keras.utils.get_file('Pets.zip', extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'Pets_filtered')

BATCH_SIZE = 5
IMG_SIZE = (224, 224)

dataset = tf.keras.utils.image_dataset_from_directory(PATH, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

def main():
    page = st.sidebar.selectbox("Choose a page", ["Description", "Model"])

    if page == "Description":
        st.header("Cats and dogs classifier")
        st.write("Please select a page on the left.")
        st.write("The dataset contains 12501 photos of cats and 12501 photos of dogs. The model predicts the class the photo belongs to.")
    elif page == "Model":
        st.title("Model")
        if st.button("Predict"):
            st.write(my_model.predict(dataset))

if __name__ == "__main__":
    main()
