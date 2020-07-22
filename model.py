import datetime
import numpy as np
import sys
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

features = [ \
    "Store", "DayOfWeek", "Customers", "Open", "Promo", \
    "CompetitionDistance","Comp_since_days", \
    "Promo2", 'Promo_since_days', \
    "StateHoliday_0", "StateHoliday_a", \
    "StateHoliday_b", "StateHoliday_c", "StoreType_a", "StoreType_b", \
    "StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", \
    "Assortment_c", "PromoInterval_Feb,May,Aug,Nov", \
    "PromoInterval_Jan,Apr,Jul,Oct", "PromoInterval_Mar,Jun,Sept,Dec", \
    "Year", "Month", "WeekOfYear","MedianSalesByStore", "MedianSalesByDayOfWeek", \
    "Monetary" , "Recency", "Customer_avg"]
params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "nthread": 4
}
num_boost_round = 130
test_features_eng = {}


def rmspe(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    error = 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
    return error


def data_cleaning(df):
    missing_value_numeric_features = ["Promo", "CompetitionDistance", "CompetitionOpenSinceMonth", \
    "CompetitionOpenSinceYear", "Promo2SinceYear", "Promo2SinceWeek", "SchoolHoliday"]
    integer_features = ["Open", "DayOfWeek", "Promo", "Store", "Customers", "CompetitionOpenSinceYear", \
    "CompetitionOpenSinceMonth", "Promo2SinceYear", "Promo2SinceWeek", "CompetitionDistance"]
    df.loc[:, missing_value_numeric_features] = df.loc[:, missing_value_numeric_features].fillna(value=-999)
    df.loc[:, "Open"].fillna(value="1", inplace=True)
    mean_imputer = SimpleImputer(strategy="median")
    df[["Customers"]] = mean_imputer.fit_transform(df[["Customers"]])
    df.loc[:, "DayOfWeek"] = df.Date.dt.dayofweek + 1
    categories = ["StateHoliday", "StoreType", "Assortment"]
    df.loc[:, categories].fillna(value="Missing", inplace=True)
    df.loc[:, "StateHoliday"] = df.StateHoliday.replace({"0.0": "0"})
    df[integer_features] = df[integer_features].astype("int64", copy=False)
    one_hot_encoding_features = ["SchoolHoliday", "StateHoliday", "StoreType", "Assortment", "PromoInterval"]
    df = pd.get_dummies(df, columns=one_hot_encoding_features)
    return df


def time_distance(df,sinceWorM,sinceYr,distance_name,week=None):
    # sinceWorM : string of column name with ...sinceMonth, if sinceWeek => week=1
    # sinceYr : string of column name with ... SinceYear
    # distance_name : ex. competition_dist or promo_dist
    m_999 = df[sinceWorM] != -999
    y_999 = df[sinceYr] != -999
    glob_999 = m_999 + y_999
    df_mask = df.loc[glob_999,:]
    if week == 1:
        df_mask.loc[:,sinceWorM]=df_mask.loc[:,sinceWorM] // 4
        df_mask.loc[:,sinceWorM] = df_mask.loc[:,sinceWorM].apply(str).replace({"0":"1","13":"12"})
    df_mask.loc[:,"temp_MY"] = pd.to_datetime(df_mask.loc[:,sinceYr].apply(str) + "-" + df_mask.loc[:,sinceWorM].apply(str) + "-01")
    df_mask.loc[:,distance_name] = (df_mask.loc[:,"Date"] - df_mask.loc[:,"temp_MY"]).dt.days
    # concat back columns with distance in days to original df
    new_df = pd.concat([df,df_mask.loc[:,distance_name]],axis=1, sort=False)
    new_df.loc[:,distance_name] = new_df.loc[:,distance_name].fillna(value=-9999).astype("int",copy=False)
    return new_df


def feature_engineering(df, test_set=None):
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["WeekOfYear"] = df.Date.dt.weekofyear
    df = time_distance(df, 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Comp_since_days')
    df = time_distance(df, 'Promo2SinceWeek', 'Promo2SinceYear','Promo_since_days', 1)
    median_sales_by_store = df.groupby("Store")["Sales"].median()
    # Convert to dictionnaire to use in TEST SET and Save it in global Dictionnary
    test_features_eng["MedianSalesByStore"] = median_sales_by_store.to_dict()
    df["MedianSalesByStore"] = df.Store.map(median_sales_by_store)
    median_sales_by_day_of_week = df.groupby("DayOfWeek")["Sales"].median()
    # Convert to dictionnaire to use in TEST SET and Save it in global Dictionnary
    test_features_eng["MedianSalesByDayOfWeek"] = median_sales_by_day_of_week.to_dict()
    df["MedianSalesByDayOfWeek"] = df.DayOfWeek.map(median_sales_by_store)
    ## RFM
    today = datetime.datetime(2014,8,1)
    agg_rule = {'Date': lambda x: (today - x.max()).days, 'Customers': lambda x: x.median(), 'Sales': lambda x: x.sum()}
    rfm_score = df.groupby('Store').agg(agg_rule)
    rfm_score.columns = ["R", "C", "M"]
    quantiles = rfm_score.quantile(q=[0.25, 0.5, 0.75])
    ## Recency Feature
    test_features_eng["Recency"] = rfm_score["R"].to_dict()
    df["Recency"] = df.Store.map(test_features_eng["Recency"])
    # Convert to dictionnaire to use in TEST SET and Save it in global Dictionnary
    ## Monetary Feature
    rfm_score["M_segments"] = rfm_score['M'].apply(m_group, args=('M',quantiles))
    # Convert to dictionnaire to use in TEST SET and Save it in global Dictionnary
    test_features_eng["Monetary"] = rfm_score["M_segments"].to_dict()
    df["Monetary"] = df.Store.map(test_features_eng["Monetary"])
    test_features_eng["Customer_avg"] = rfm_score["C"].to_dict()
    df["Customer_avg"] = df.Store.map(test_features_eng["Customer_avg"])
    pickle.dump(test_features_eng, open("augmented_features.dat", "wb"))
    return df


def load_data():
    train = pd.read_csv("data/train.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
    print(str(train.shape[0]) + " rows have been found.")
    print("Loading store data.")
    store = pd.read_csv("data/store.csv", header=0)
    print(str(store.shape[0]) + " rows have been found.")
    print("Filter rows where Sales is bigger than zero")
    train = train[train.Sales > 0]
    print("Filter rows where Open is true")
    train = train[train.Open != 0]
    print("Join train and store")
    train = pd.merge(train, store, on="Store")
    return train


def m_group(x, type, quantiles):
    if x <= quantiles[type].iloc[0]:
        return 4
    elif x <= quantiles[type].iloc[1]:
        return 3
    elif x <= quantiles[type].iloc[2]:
        return 2
    else:
        return 1


def display_features_by_importance(features):
    features_dict = {}
    feature_importance_sorted = sorted(features.items(), key=lambda kv: kv[1], reverse=True)
    for i in range(len(feature_importance_sorted)):
        feature, importance_rate = feature_importance_sorted[i]
        features_dict[feature] = importance_rate
    print(features_dict)


def display_metrics(x_train, y_train, x_holdout, y_holdout):
    print("")
    print("#################################################")
    train_pred = x_train["Prediction"].values
    train_actuals = y_train.values
    rmspe_train = rmspe(train_pred, train_actuals)
    print('RMSPE on TRAIN SET: {:.3f}'.format(rmspe_train))
    predictions = x_holdout["Prediction"].values
    actuals = y_holdout.values
    rmspe_test = rmspe(predictions, actuals)
    print('RMSPE on TEST SET: {:.3f}'.format(rmspe_test))
    print("#################################################")
    print("")


print("Loading training data.")
data = load_data()

print("Clean up data")
data = data_cleaning(data)

print("Feature engineering")
data = feature_engineering(data)

Y = data.Sales
X = data.drop(columns=["Sales"], axis=1)
X_train, X_holdout, Y_train, Y_holdout = train_test_split(X, Y, test_size=0.30, random_state=42, shuffle=True)

print("Train Random Forrest")
dtrain = xgb.DMatrix(X_train[features].values, Y_train.values)
dvalid = xgb.DMatrix(X_holdout[features].values, Y_holdout.values)
watchlist = [(dtrain, "train"), (dvalid, "eval")]
gbm = xgb.XGBRegressor()
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=25, verbose_eval=True)

print("Computing predictions")
yhat_holdout = gbm.predict(xgb.DMatrix(X_holdout[features].values))
X_holdout["Prediction"] = yhat_holdout
yhat_train = gbm.predict(xgb.DMatrix(X_train[features].values))
X_train["Prediction"] = yhat_train

print("Display the list of features by importance")
feature_score = gbm.get_score(importance_type="gain")
display_features_by_importance(feature_score)

# save model to file
print("Save the model to rossman.dat")
gbm.save_model("rossman.bin")
pickle.dump(gbm, open("rossman.dat", "wb"))

display_metrics(X_train, Y_train, X_holdout, Y_holdout)