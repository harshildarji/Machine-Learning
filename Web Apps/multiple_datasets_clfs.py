# Video link: https://www.youtube.com/watch?v=Klqn--Mu2pE&ab_channel=PythonEngineer
import numpy as np
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("Beginner ML WebApp")
st.write(
    "_You can change the dataset and the classifier from the left sidebar._\n\n---"
)

dataset = st.sidebar.selectbox(
    "Choose a dataset:", ("Iris", "Breast cancer", "Wine dataset")
)
clf = st.sidebar.selectbox("Choose a classifier:", ("KNN", "SVM", "Random Forest"))

_dataset = {
    "Iris": datasets.load_iris(),
    "Breast cancer": datasets.load_breast_cancer(),
    "Wine dataset": datasets.load_wine(),
}


def get_dataset(dataset_name):
    _data = _dataset[dataset_name]
    return _data.data, _data.target


def get_clf(clf_name):
    if clf_name == "KNN":
        k = st.sidebar.slider("Nearest neighbours to consider:", 1, 15)
        clf = KNeighborsClassifier(n_neighbors=k)
    elif clf_name == "SVM":
        c = st.sidebar.slider("Regularization value:", 0.01, 10.0)
        clf = SVC(C=c)
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("Maximum depth:", 2, 15)
        n_estimators = st.sidebar.slider("Number of estimators:", 1, 100)
        clf = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, random_state=42
        )

    return clf


clf = get_clf(clf)
X, y = get_dataset(dataset)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)
f1_macro = f1_score(y_test, preds, average="macro")
f1_micro = f1_score(y_test, preds, average="micro")

st.write(
    f"Dataset:\n"
    f"#### **{dataset}**\n"
    f"- Shape: **{X.shape}**\n"
    f"- Number of classes: **{len(np.unique(y))}**\n---"
)
st.write("Classifier (*with params*):\n" "#### **{}**\n---".format(clf))
st.write(
    f"Performance:\n\n"
    f"Accuracy: **{acc}**\n\n"
    f"F1 Score:\n"
    f"- Macro: **{f1_macro}**\n"
    f"- Micro: **{f1_micro}**"
)
