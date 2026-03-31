"""
FPL AI Optimizer - Machine Learning Module
==========================================
Trains Random Forest and Gradient Boosting (XGBoost-style) regressors
to predict each player's expected points for the next gameweek.

Models are evaluated with MAE and RMSE; the best model's predictions
are attached to the DataFrame as `predicted_points`.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Proxy target: weighted combination of performance metrics
    that mimics what a player's 'next gameweek points' would look like.

    In production you would shift this by one gameweek.
    Here we use a smoothed, feature-derived estimate as ground truth.
    """
    pts = df["total_points"].clip(0, 200)
    minutes_frac = (df["minutes"] / df["minutes"].max()).clip(0, 1)

    # Normalise key metrics
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    target = (
        0.50 * norm(pts) +
        0.20 * norm(df["ict_index"]) +
        0.15 * norm(df["form_score"]) +
        0.15 * norm(minutes_frac)
    ) * 15   # scale to ~point range

    return target.rename("expected_pts")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

class FPLPredictor:
    """
    Trains Random Forest and Gradient Boosting regressors,
    selects the best one, and exposes predict().
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.models: dict = {}
        self.metrics: dict = {}
        self.best_name: str = ""
        self.best_model = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> dict:
        """
        Train on the provided DataFrame.

        Returns
        -------
        dict  : evaluation metrics for each model
        """
        X = df[self.feature_cols].values
        y = build_target(df).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        # ── Random Forest ─────────────────────────────────────────────
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_s, y_train)
        self.models["Random Forest"] = rf

        # ── Gradient Boosting (XGBoost-style) ─────────────────────────
        gb = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
        gb.fit(X_train_s, y_train)
        self.models["Gradient Boosting"] = gb

        # ── Evaluate ──────────────────────────────────────────────────
        for name, model in self.models.items():
            preds = model.predict(X_test_s)
            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            self.metrics[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}

        # Best = lowest RMSE
        self.best_name = min(self.metrics, key=lambda k: self.metrics[k]["RMSE"])
        self.best_model = self.models[self.best_name]

        return self.metrics

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted points for every row in df."""
        X = df[self.feature_cols].values
        X_s = self.scaler.transform(X)
        return self.best_model.predict(X_s)

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.Series:
        """Return feature importances of the winning model."""
        imp = self.best_model.feature_importances_
        return (
            pd.Series(imp, index=self.feature_cols)
            .sort_values(ascending=False)
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_ml_pipeline(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, FPLPredictor, dict]:
    """
    Full ML pipeline.

    Returns
    -------
    df_out      : original df with `predicted_points` column attached
    predictor   : fitted FPLPredictor instance
    metrics     : {model_name: {MAE, RMSE}}
    """
    predictor = FPLPredictor(feature_cols)
    metrics   = predictor.fit(df)
    df = df.copy()
    df["predicted_points"] = predictor.predict(df).clip(0)
    return df, predictor, metrics
