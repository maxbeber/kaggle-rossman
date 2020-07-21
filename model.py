import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def metric(preds, actuals):
	preds = preds.reshape(-1)
	actuals = actuals.reshape(-1)
	assert preds.shape == actuals.shape
	return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


print("Loading training data.")
train = pd.read_csv("data/train.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
print(str(train.shape[0]) + " rows have been found.")

print("Loading store data.")
store = pd.read_csv("data/store.csv", header=0)
print(str(store.shape[0]) + " rows have been found.")

print("Loading test data")
test = pd.read_csv("data/test.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
print(str(test.shape[0]) + " rows have been found.")

print("Filter rows where Sales is bigger than zero")
train = train[train.Sales > 0]
test = test[test.Sales > 0]

print("Filter rows where Open is true")
train = train[train.Open != 0]

print("Join train and store")
train = pd.merge(train, store, on="Store")
test = pd.merge(test, store, on="Store")

print("Clean up data")
#train.dropna(inplace=True)
#clean_up_data()

print("Train Random Forrest")
features = ["Store", "Open"]

naive_sales_per_store = train.groupby("Store")["Sales"].mean()
test["Prediction"] = test.Store.map(naive_sales_per_store)

# Train a Random Forest model
rf = RandomForestRegressor()
#rf.fit(train[features], train.Sales)
#test["Prediction"] = rf.predict(test[features])

# Measure the error
predictions = test["Prediction"].values
actuals = test["Sales"].values
rmspe = metric(predictions, actuals)
print('Validation RMSPE for Baseline I model: {:.3f}'.format(rmspe))
