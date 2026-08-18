from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass
class ModelBundle:
    model: object
    feature_columns: list[str]
    train_rows: int


def make_primary_model() -> object:
    try:
        from xgboost import XGBClassifier

        base = XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        base = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, random_state=42)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CalibratedClassifierCV(base, method="isotonic", cv=3)),
        ]
    )


def make_baseline_model() -> object:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> ModelBundle:
    train = clean_labeled_frame(frame)
    if len(train) < 50:
        raise ValueError("Need at least 50 labeled games to train a model.")
    columns = feature_columns or FEATURE_COLUMNS
    model = make_primary_model()
    X = train[columns]
    y = train["target_home_win"].astype(int)
    model.fit(X, y)
    return ModelBundle(model=model, feature_columns=columns.copy(), train_rows=len(train))


def train_baseline(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> object:
    train = clean_labeled_frame(frame)
    columns = feature_columns or FEATURE_COLUMNS
    model = make_baseline_model()
    model.fit(train[columns], train["target_home_win"].astype(int))
    return model


def predict_home_prob(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([])
    return bundle.model.predict_proba(frame[bundle.feature_columns])[:, 1]


def evaluate_predictions(y_true: pd.Series, home_probs: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    clipped = np.clip(home_probs, 0.001, 0.999)
    return {
        "games": float(len(y)),
        "accuracy": float(accuracy_score(y, clipped >= 0.5)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, clipped)),
        "home_win_rate": float(np.mean(y)),
        "avg_home_prob": float(np.mean(clipped)),
    }


def calibration_table(y_true: pd.Series, home_probs: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"actual": y_true.astype(int), "prob": home_probs})
    df["bucket"] = pd.cut(df["prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return (
        df.groupby("bucket", observed=False)
        .agg(games=("actual", "size"), avg_prob=("prob", "mean"), actual_home_win_rate=("actual", "mean"))
        .reset_index()
    )


def clean_labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["target_home_win"].notna()].copy()


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ModelBundle:
    return joblib.load(path)
