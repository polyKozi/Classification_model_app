import streamlit as st
import pickle
import numpy as np

def main():
    X_test = np.array([[-4.04, -7.29, -3.77,  3.34, -1.43, -1.44, -5.50,  1.71, 9.54, -9.07,  3.66, -7.24, -8.07],
    [ 1.04, -8.91, -3.41, -1.02, -1.55,  1.05,  1.10, -1.12, 4.94,  8.57,  2.56,  1.36, 8.96],
    [-1.74, -8.91,  1.21,  1.32, -4.40,  6.80,  8.80, -5.73, 1.60, -1.07,  3.66,  9.55, -2.34],
    [ 3.51, -5.58, -8.11, -7.35, -4.40,  1.48,  1.65, -7.32, -3.73, -5.26,  3.00,  2.69, 1.63],
    [ 1.77,  1.63,  1.10, -2.72, 5.87,  7.74,  1.19, -4.94, 2.16,  2.05,  3.44,  8.29, 1.33],
    [ 1.70, -4.24,  3.10, -1.42, -2.97,  3.04,  4.88, -4.94, 7.20,  2.37,  3.00,  1.40, 1.65]])
    y_test = np.array([2, 1, 2, 1, 1, 1])


    with open('model_pic_clf.pkl', 'rb') as pkl_file:
        clf = pickle.load(pkl_file)
    page = st.sidebar.selectbox("Choose a page", ["Description", "Model"])

    if page == "Description":
        st.header("Cats and dogs classifier")
        st.write("Please select a page on the left.")
        st.write("The dataset contains 12501 photos of cats and 12501 photos of dogs. The model predicts the class the photo belongs to.")
    elif page == "Model":
        st.title("Model")
        if st.button("Predict"):
            st.write("Prediction:", clf.predict(X_test))
            st.write("True", y_test)

if __name__ == "__main__":
    main()