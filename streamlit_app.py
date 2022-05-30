import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_win_pr = pd.DataFrame()
df_win_pr = pd.read_csv('data/wine_preprocessed.csv', sep=',')

x_useful_col = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "OD280/OD315_of_diluted wines", "proline"]
X = df_win_pr[x_useful_col]
y = df_win_pr.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

def main():
    page = st.sidebar.selectbox("Choose a page", ["Description", "Model"])

    if page == "Description":
        st.header("Cats and dogs classifier")
        st.write("Please select a page on the left.")
        st.write("The dataset contains 12501 photos of cats and 12501 photos of dogs. The model predicts the class the photo belongs to.")
    elif page == "Model":
        st.title("Model")
        if st.button("Predict"):
            st.write(clf.predict(X_test))

if __name__ == "__main__":
    main()