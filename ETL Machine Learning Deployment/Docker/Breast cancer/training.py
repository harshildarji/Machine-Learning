import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset = pd.read_csv("breast_cancer.csv")
y = dataset["target"]
X = dataset.drop("target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("[+] Training and saving the model...")
svc = SVC()
svc.fit(X_train, y_train)
pred = svc.predict(X_test)

file = open("svc_model.pkl", "wb")
pickle.dump(svc, file)
file.close()
