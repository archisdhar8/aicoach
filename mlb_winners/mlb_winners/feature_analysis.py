from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .db import upsert_df, write_csv
from .features import FEATURE_COLUMNS
from .moneyline_diagnostics import feature_subset_experiment_rows, persist_evaluation_rows
from .modeling import evaluate_predictions, predict_home_prob, train_model


LEAKAGE_PATTERNS = ("target", "score", "home_won", "result", "final")


@dataclass(frozen=True)
class FeatureAnalysisResult:
    summary: pd.DataFrame
    metrics: dict[str, float]


def analyze_features(
    con,
    frame: pd.DataFrame,
    train_through: int,
    test_year: int,
    report_dir,
) -> FeatureAnalysisResult:
    train = frame[(frame["season"] <= train_through) & frame["target_home_win"].notna()].copy()
    test = frame[(frame["season"] == test_year) & frame["target_home_win"].notna()].copy()
    if train.empty or test.empty:
        raise ValueError("Need non-empty train and test frames for feature analysis.")
    bundle = train_model(train)
    probs = predict_home_prob(bundle, test)
    metrics = evaluate_predictions(test["target_home_win"], probs)

    perm = permutation_importance(
        bundle.model,
        test[FEATURE_COLUMNS],
        test["target_home_win"].astype(int),
        n_repeats=8,
        random_state=42,
        scoring="neg_log_loss",
    )
    rows = [
        {
            "feature_name": name,
            "importance_type": "permutation_log_loss",
            "importance": float(value),
            "risk_flag": risk_flag(name, value, perm.importances_mean),
        }
        for name, value in zip(FEATURE_COLUMNS, perm.importances_mean)
    ]
    rows.extend(shap_rows(bundle, test))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return FeatureAnalysisResult(summary=summary, metrics=metrics)
    summary["rank"] = summary.groupby("importance_type")["importance"].rank(ascending=False, method="first").astype(int)
    summary = summary.sort_values(["importance_type", "rank"]).reset_index(drop=True)
    persist_feature_report(con, summary, train_through, test_year)
    write_csv(summary, report_dir / f"feature_importance_{train_through}_{test_year}.csv")
    write_csv(pd.DataFrame([metrics]), report_dir / f"feature_importance_metrics_{train_through}_{test_year}.csv")
    subset_report = feature_subset_experiment_rows(train, test, summary)
    write_csv(subset_report, report_dir / f"feature_subset_experiments_{train_through}_{test_year}.csv")
    persist_evaluation_rows(con, "feature_subset_importance", subset_report.assign(year=test_year))
    return FeatureAnalysisResult(summary=summary, metrics=metrics)


def shap_rows(bundle, test: pd.DataFrame) -> list[dict[str, object]]:
    try:
        import shap
    except Exception:
        return []
    sample = test[FEATURE_COLUMNS].head(min(len(test), 500))
    try:
        explainer = shap.Explainer(lambda x: bundle.model.predict_proba(pd.DataFrame(x, columns=FEATURE_COLUMNS))[:, 1], sample)
        values = explainer(sample)
        means = np.abs(values.values).mean(axis=0)
    except Exception:
        return []
    return [
        {
            "feature_name": name,
            "importance_type": "shap_abs_mean",
            "importance": float(value),
            "risk_flag": risk_flag(name, value, means),
        }
        for name, value in zip(FEATURE_COLUMNS, means)
    ]


def risk_flag(feature_name: str, importance: float, all_importances) -> str:
    lower = feature_name.lower()
    if any(pattern in lower for pattern in LEAKAGE_PATTERNS):
        return "leakage_name_risk"
    values = np.array(all_importances, dtype=float)
    if len(values) and importance > max(np.percentile(values, 95), values.mean() + 3 * values.std()):
        return "dominates_model"
    if importance <= 0:
        return "noise_candidate"
    if "last7" in lower or "ewma" in lower:
        return "recent_form_watch"
    return ""


def persist_feature_report(con, summary: pd.DataFrame, train_through: int, test_year: int) -> int:
    rows = []
    for row in summary.to_dict("records"):
        report_id = hashlib.sha1(
            f"{train_through}:{test_year}:{row['importance_type']}:{row['feature_name']}".encode()
        ).hexdigest()
        rows.append(
            {
                "report_id": report_id,
                "train_through": train_through,
                "test_year": test_year,
                "feature_name": row["feature_name"],
                "importance_type": row["importance_type"],
                "importance": row["importance"],
                "rank": int(row["rank"]),
                "risk_flag": row.get("risk_flag") or "",
                "raw_payload": json.dumps(row, default=str),
            }
        )
    return upsert_df(con, "feature_importance_reports", pd.DataFrame(rows))
