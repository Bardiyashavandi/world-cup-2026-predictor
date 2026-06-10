import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")


# ─────────────────────────────────────────
# 1. FEATURE COLUMNS
# ─────────────────────────────────────────

FEATURE_COLS = [
    "is_neutral",
    "home_elo", "away_elo", "elo_diff",
    "home_form5_points", "home_form5_avg_scored", "home_form5_avg_conceded",
    "away_form5_points", "away_form5_avg_scored", "away_form5_avg_conceded",
    "home_form10_points", "home_form10_avg_scored", "home_form10_avg_conceded",
    "away_form10_points", "away_form10_avg_scored", "away_form10_avg_conceded",
    "form5_points_diff", "form10_points_diff", "avg_scored_diff",
    "h2h_matches", "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "h2h_avg_goals",
]

REFERENCE_DATE = pd.Timestamp("2026-06-11")


# ─────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────

def load_data():
    train = pd.read_csv("data/processed/train_features.csv")
    predict = pd.read_csv("data/processed/predict_features.csv")

    train = train.dropna(subset=["home_goals", "away_goals"])
    train[FEATURE_COLS] = train[FEATURE_COLS].fillna(0)
    predict[FEATURE_COLS] = predict[FEATURE_COLS].fillna(0)

    train["result"] = train.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
        else ("D" if r["home_goals"] == r["away_goals"] else "A"),
        axis=1
    )

    return train, predict


# ─────────────────────────────────────────
# 3. TIME WEIGHTS
# ─────────────────────────────────────────

def compute_sample_weights(train: pd.DataFrame) -> np.ndarray:
    train = train.copy()
    train["date"] = pd.to_datetime(train["date"])
    weights = train["date"].apply(
        lambda d: np.exp(-0.001 * (REFERENCE_DATE - d).days)
    )
    weights = weights / weights.sum() * len(train)
    return weights.values


# ─────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────

def train_logistic(train: pd.DataFrame) -> tuple:
    """
    Train Logistic Regression to predict W/D/L outcome.

    This model does ONE thing well:
    Outputs calibrated win/draw/loss probabilities.

    These probabilities feed directly into the ensemble
    as a calibration signal alongside scoreline models.

    Note: No class_weight='balanced' — we want the model
    to reflect the true distribution of outcomes in football
    (home wins ~48%, away wins ~29%, draws ~23%)
    """
    X = train[FEATURE_COLS].values
    y = train["result"].values
    sample_weights = compute_sample_weights(train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Logistic Regression — Result classifier...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_scaled, y, sample_weight=sample_weights)

    # Cross validation
    print("Cross-validating with TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
        X_scaled, y,
        cv=tscv,
        scoring="accuracy"
    )
    print(f"  Result accuracy: {scores.mean():.3f} "
          f"(±{scores.std():.3f})")

    # Print coefficients for interpretability
    classes = model.classes_
    print(f"\n  Classes: {classes}")

    print(f"\n  Top features → Home Win:")
    h_idx = list(classes).index("H")
    coefs = pd.Series(
        model.coef_[h_idx], index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, coef in coefs.head(5).items():
        print(f"    {feat:35} {coef:+.3f}")

    print(f"\n  Top features → Away Win:")
    a_idx = list(classes).index("A")
    coefs_away = pd.Series(
        model.coef_[a_idx], index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, coef in coefs_away.head(5).items():
        print(f"    {feat:35} {coef:+.3f}")

    return model, scaler


# ─────────────────────────────────────────
# 5. PREDICT OUTCOMES
# ─────────────────────────────────────────

def predict_outcomes(model, scaler,
                      predict_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict W/D/L probabilities for all fixtures.
    This is what logistic regression is actually good at.
    No scoreline conversion — just clean probabilities.
    """
    X = predict_df[FEATURE_COLS].fillna(0).values
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)
    classes = list(model.classes_)

    h_idx = classes.index("H")
    d_idx = classes.index("D")
    a_idx = classes.index("A")

    results = []
    for i, (_, row) in enumerate(predict_df.iterrows()):
        hw_prob = round(proba[i][h_idx], 3)
        d_prob = round(proba[i][d_idx], 3)
        aw_prob = round(proba[i][a_idx], 3)

        # Predicted result = highest probability outcome
        predicted = max(
            [("H", hw_prob), ("D", d_prob), ("A", aw_prob)],
            key=lambda x: x[1]
        )[0]

        results.append({
            "match_id": row.get("match_id", i),
            "group": row.get("group", ""),
            "matchday": row.get("matchday", 1),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win_prob": hw_prob,
            "draw_prob": d_prob,
            "away_win_prob": aw_prob,
            "predicted_result": predicted,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────
# 6. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_logistic():
    print("=" * 60)
    print("  LOGISTIC REGRESSION MODEL — WC 2026 Predictor")
    print("=" * 60)
    print()

    print("Loading data...")
    train, predict = load_data()
    print(f"  Training set: {len(train)} matches")
    print(f"  Prediction set: {len(predict)} fixtures")
    print()

    print("Result distribution in training data:")
    dist = train["result"].value_counts()
    total = len(train)
    for result, count in dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {result}: {count:5} ({pct:.1f}%) {bar}")
    print()

    model, scaler = train_logistic(train)

    print("\nGenerating WC 2026 outcome predictions...")
    out_df = predict_outcomes(model, scaler, predict)

    os.makedirs("data/predictions", exist_ok=True)
    out_df.to_csv("data/predictions/logistic_all.csv", index=False)
    out_df[out_df["matchday"] == 1].to_csv(
        "data/predictions/logistic_md1.csv", index=False
    )

    # Print
    print()
    print("Logistic Regression — WC 2026 Outcome Predictions")
    print("=" * 60)
    for _, row in out_df.iterrows():
        result_str = (
            "HOME WIN" if row["predicted_result"] == "H"
            else "DRAW" if row["predicted_result"] == "D"
            else "AWAY WIN"
        )
        print(
            f"Group {row['group']} MD{row['matchday']} | "
            f"{row['home_team']:20} vs "
            f"{row['away_team']:20} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']} "
            f"→ {result_str}"
        )

    # Summary
    print()
    results_dist = out_df["predicted_result"].value_counts()
    print("Predicted outcome distribution:")
    for r, count in results_dist.items():
        label = (
            "Home wins" if r == "H"
            else "Draws" if r == "D"
            else "Away wins"
        )
        pct = count / len(out_df) * 100
        bar = "█" * int(pct / 3)
        print(f"  {label:12} {count:3} ({pct:.1f}%) {bar}")

    print()
    print("Saved to data/predictions/logistic_all.csv")
    print("Saved to data/predictions/logistic_md1.csv")


if __name__ == "__main__":
    run_logistic()