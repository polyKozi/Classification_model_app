import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    df_win_pr = pd.DataFrame()
    df_win_pr = pd.read_csv('wine_preprocessed.csv', sep=',')

    x_useful_col = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "OD280/OD315_of_diluted wines", "proline"]
    X = df_win_pr[x_useful_col]
    y = df_win_pr.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf = clf.fit(X_train, y_train)

    page = st.sidebar.selectbox("Choose a page", ["Description", "Model"])

    if page == "Description":
        st.header("Wine")
        st.write("Please select a page on the left.")
        st.write("The dataset contains 178 objects. Each object is a wine. There are 3 types of wine. The task is to predict what class each wine belongs to.")
    elif page == "Model":
        st.title("Model")
        if st.button("Predict"):
            st.write("Prediction:", clf.predict(X_test))
            st.write("True", y_test)

if __name__ == "__main__":
    main()