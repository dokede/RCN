import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Folder na modele ───────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# ── 1. Wczytanie danych ────────────────────────────────────────────────────────
print("▶ Wczytywanie data_processed.csv...")
df = pd.read_csv(r"data\data_processed.csv")

X = df.drop(columns=["amount"])
y = df["amount"]

# ── 2. Podział na train/test ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)
print(f"   Train: {X_train.shape[0]} wierszy | Test: {X_test.shape[0]} wierszy")

# ── 3. Trenowanie modeli ───────────────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=40),
    "XGBoost":           XGBRegressor(n_estimators=100, random_state=40, verbosity=0),
}

for name, model in models.items():
    print(f"▶ Trenuję {name}...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    r2_train   = r2_score(y_train, y_pred_train)

    print(f"   Train RMSE: {rmse_train:.2f} | R²: {r2_train:.4f}")

# ── 4. Zapis modeli ────────────────────────────────────────────────────────────
print("\n▶ Zapisywanie modeli do folderu models/...")

with open("models/model_linear_regression.pkl", "wb") as f:
    pickle.dump(models["Linear Regression"], f)

with open("models/model_random_forest.pkl", "wb") as f:
    pickle.dump(models["Random Forest"], f)

with open("models/model_xgboost.pkl", "wb") as f:
    pickle.dump(models["XGBoost"], f)

print("✅ Gotowe! Zapisano modele w folderze models/")