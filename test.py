import numpy as np
import pickle
import pandas as pd
import sys
import xgboost as xgb

def rmspe(preds, actuals):
      preds = preds.reshape(-1)
      actuals = actuals.reshape(-1)
      assert preds.shape == actuals.shape
      error = 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
      return error

print("Loading test data.")
test = pd.read_csv("data/test.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
print(str(test.shape[0]) + " rows have been found.")
print("Loading store data.")
store = pd.read_csv("data/store.csv", header=0)
print(str(store.shape[0]) + " rows have been found.")
print("Filter rows where Sales is bigger than zero")
test = test[test.Sales > 0]
print("Join train and store")
X_test = pd.merge(test, store, on="Store")
print("Loaded the model")
model = pickle.load(open("rossman.dat", "rb"))
print("Start prediction")
predictions = model.predict(X_test)
test_pred = test["Prediction"].values
test_actuals = test["Sales"].values
rmspe_test = rmspe(test_pred, test_actuals)

print(rmspe_test)