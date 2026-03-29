import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)

print("▶ Wczytywanie data")
df = pd.read_csv("data/data_processed.csv", index_col=0)

X = df.drop(columns=["amount"])
y = df["amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)
print(f"   Train: {X_train.shape[0]} wierszy | Test: {X_test.shape[0]} wierszy\n")


models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        random_state=40,
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,       
        random_state=40,
        verbosity=0,
    ),
}

results = {}

for name, model in models.items():
    print(f"▶ Trenuję {name}...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    rmse_test  = root_mean_squared_error(y_test,  y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)
    r2_train    = r2_score(y_train, y_pred_train)

    cv_scores  = cross_val_score(model, X_train, y_train,
                                 cv=5, scoring="neg_root_mean_squared_error")
    cv_rmse    = -cv_scores.mean()

    results[name] = {
        "model":       model,
        "rmse_train":  rmse_train,
        "rmse_test":   rmse_test,
        "r2_test":     r2_test,
        "r2_train":     r2_train,
        "y_pred_test": y_pred_test,
    }
    print(f"   Train RMSE: {rmse_train:.2f} | R²: {r2_train:.4f}")


print("\n▶ Zapisywanie modeli...")
with open("models/model_linear_regression.pkl", "wb") as f:
    pickle.dump(results["Linear Regression"]["model"], f)
with open("models/model_random_forest.pkl", "wb") as f:
    pickle.dump(results["Random Forest"]["model"], f)
with open("models/model_xgboost.pkl", "wb") as f:
    pickle.dump(results["XGBoost"]["model"], f)
print("✅ Modele zapisane w folderze models/\n")

