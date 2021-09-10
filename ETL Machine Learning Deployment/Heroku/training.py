import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

print("[+] Preprocessing the dataset...")
dataset = pd.read_csv("cardata.csv")

final_dataset = dataset[
    [
        "Year",
        "Selling_Price",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
    ]
]
final_dataset["Current_Year"] = 2021
final_dataset["age"] = final_dataset["Current_Year"] - final_dataset["Year"]
final_dataset.drop(["Year", "Current_Year"], axis=1, inplace=True)
final_dataset = pd.get_dummies(final_dataset, drop_first=True)

X = final_dataset.iloc[:, 1:]
y = final_dataset.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ["auto", "sqrt"]
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
}

print("[+] Training and saving the model...")
model = RandomForestRegressor()
rf_random = RandomizedSearchCV(
    estimator=model,
    param_distributions=random_grid,
    scoring="neg_mean_squared_error",
    n_iter=10,
    cv=5,
    verbose=0,
    random_state=42,
    n_jobs=1,
)
rf_random.fit(X_train, y_train)

file = open("rf_regression_model.pkl", "wb")
pickle.dump(rf_random, file)
file.close()
