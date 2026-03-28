import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# ── 1. Wczytanie danych ────────────────────────────────────────────────────────
print("▶ Wczytywanie data_processed.csv...")
df = pd.read_csv(r"data\data_processed.csv", index_col=0)

X = df.drop(columns=["amount"])
y = df["amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)
print(f"   Train: {X_train.shape[0]} wierszy | Test: {X_test.shape[0]} wierszy")

# ── 2. Wczytanie modeli ────────────────────────────────────────────────────────
print("\n▶ Wczytywanie modeli z folderu models/...")

models = {}
for name, filename in [
    ("Linear Regression", "models/model_linear_regression.pkl"),
    ("Random Forest",     "models/model_random_forest.pkl"),
    ("XGBoost",           "models/model_xgboost.pkl"),
]:
    with open(filename, "rb") as f:
        models[name] = pickle.load(f)
    print(f"   Wczytano: {filename}")

# ── 3. Ewaluacja ───────────────────────────────────────────────────────────────
print("\n▶ Wyniki:")
print(f"{'Model':<25} {'Train RMSE':>12} {'Train R²':>10} {'Test RMSE':>12} {'Test R²':>10}")
print("-" * 72)

for name, model in models.items():
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    rmse_test  = root_mean_squared_error(y_test,  y_pred_test)
    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)

    print(f"{name:<25} {rmse_train:>12.2f} {r2_train:>10.4f} {rmse_test:>12.2f} {r2_test:>10.4f}")

print("\n✅ Testowanie zakończone.")