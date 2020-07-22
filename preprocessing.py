import pandas as pd
from sklearn.impute import SimpleImputer

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