import streamlit as st

def main():
    page = st.sidebar.selectbox("Choose a page", ["Description", "Model"])

    if page == "Description":
        st.header("Cats and dogs classifier")
        st.write("Please select a page on the left.")
        st.write("The dataset contains 12501 photos of cats and 12501 photos of dogs. The model predicts the class the photo belongs to.")
    elif page == "Model":
        st.title("Model")
        if st.button("Predict"):
            st.write("This is a cat =^._.^=")

if __name__ == "__main__":
    main()
