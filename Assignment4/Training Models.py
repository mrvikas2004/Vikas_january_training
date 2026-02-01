import pandas as pd

df = pd.read_csv("data.csv")
df.head()

#understanding data
df.info()
df.isnull().sum()

#Data Cleaning and Preprocessing
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df.drop("date", axis=1, inplace=True)

#remove irrelevents
df.drop(["street", "country"], axis=1, inplace=True)

#remove duplicates
df.drop_duplicates(inplace=True)

#Outliers
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

df = df[(df["price"] >= Q1 - 1.5*IQR) &
        (df["price"] <= Q3 + 1.5*IQR)]

X = df.drop("price", axis=1)
y = df["price"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(model):
    preds = model.predict(X_test)
    return {
        "R2": r2_score(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds)
    }

from sklearn.linear_model import LinearRegression

lr = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression())
])

lr.fit(X_train, y_train)
lr_res = evaluate(lr)

from sklearn.tree import DecisionTreeRegressor

dt = Pipeline([
    ("prep", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42))
])

dt.fit(X_train, y_train)
dt_res = evaluate(dt)

from sklearn.ensemble import RandomForestRegressor

rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

rf.fit(X_train, y_train)
rf_res = evaluate(rf)

from sklearn.neighbors import KNeighborsRegressor

knn = Pipeline([
    ("prep", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=5))
])

knn.fit(X_train, y_train)
knn_res = evaluate(knn)

from sklearn.svm import SVR

svm = Pipeline([
    ("prep", preprocessor),
    ("model", SVR())
])

svm.fit(X_train, y_train)
svm_res = evaluate(svm)

results = pd.DataFrame(
    [lr_res, dt_res, rf_res, knn_res, svm_res],
    index=["Linear Regression", "Decision Tree", "Random Forest", "KNN", "SVM"]
)


df.drop(["street", "country"], axis=1, inplace=True, errors="ignore")


print(results)
import matplotlib.pyplot as plt

y_pred = rf.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

results

df_output = X_test.copy()
df_output["Actual_Price"] = y_test
df_output["Predicted_Price"] = rf.predict(X_test)

df_output.to_csv("output.csv", index=False)

