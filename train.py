import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import joblib

housing = fetch_california_housing(as_frame=True)

X= housing.data
y = housing.target
model = RandomForestRegressor()

model.fit(X,y)

joblib.dump(model,"model/model.pkl")
