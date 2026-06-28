"""
train_model.py
Trains a Ridge Regression model to predict hospital bed occupancy.

Mirrors the project on your resume: benchmarks Linear/Ridge/Lasso/ElasticNet
with repeated 5-fold cross-validation + hyperparameter tuning, selects Ridge,
and persists the winning model to model.pkl.

Run once locally:  python train_model.py
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Build a synthetic, but realistic, hospital dataset.
#    Features: population_density, elderly_ratio, admissions_last_week,
#              avg_length_of_stay, staff_ratio, season_index
#    Target:   bed_occupancy (bed-days)
# ---------------------------------------------------------------------------
def make_dataset(n=2000):
    population_density   = np.random.normal(50, 15, n).clip(5, 120)
    elderly_ratio        = np.random.normal(0.18, 0.05, n).clip(0.05, 0.40)
    admissions_last_week = np.random.normal(120, 35, n).clip(20, 300)
    avg_length_of_stay   = np.random.normal(4.5, 1.2, n).clip(1, 12)
    staff_ratio          = np.random.normal(0.8, 0.15, n).clip(0.3, 1.5)
    season_index         = np.random.uniform(0, 1, n)  # 0=summer .. 1=winter peak

    noise = np.random.normal(0, 1.5, n)

    bed_occupancy = (
        0.04 * population_density
        + 18.0 * elderly_ratio
        + 0.09 * admissions_last_week
        + 0.85 * avg_length_of_stay
        - 3.2 * staff_ratio
        + 4.0 * season_index
        + noise
    )

    return pd.DataFrame({
        "population_density": population_density,
        "elderly_ratio": elderly_ratio,
        "admissions_last_week": admissions_last_week,
        "avg_length_of_stay": avg_length_of_stay,
        "staff_ratio": staff_ratio,
        "season_index": season_index,
        "bed_occupancy": bed_occupancy,
    })


def main():
    df = make_dataset()
    X = df.drop(columns=["bed_occupancy"])
    y = df["bed_occupancy"]

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    # Benchmark four models with a scaler in front (good practice for linear models)
    candidates = {
        "Linear":     LinearRegression(),
        "Ridge":      Ridge(),
        "Lasso":      Lasso(max_iter=10000),
        "ElasticNet": ElasticNet(max_iter=10000),
    }

    print("Cross-validated RMSE (lower is better):")
    for name, est in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        neg_mse = cross_val_score(pipe, X, y, cv=cv,
                                  scoring="neg_mean_squared_error")
        rmse = np.sqrt(-neg_mse)
        print(f"  {name:12s}  RMSE = {rmse.mean():.3f} (+/- {rmse.std():.3f})")

    # Tune Ridge's alpha (the winning family on your resume)
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    grid = GridSearchCV(
        ridge_pipe,
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=cv, scoring="neg_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X, y)
    best = grid.best_estimator_

    preds = best.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"\nSelected Ridge (alpha={grid.best_params_['model__alpha']})")
    print(f"  Train RMSE = {rmse:.3f} bed-days   R2 = {r2:.3f}")

    joblib.dump({
        "model": best,
        "features": list(X.columns),
    }, "app/model.pkl")
    print("\nSaved -> app/model.pkl")


if __name__ == "__main__":
    main()
