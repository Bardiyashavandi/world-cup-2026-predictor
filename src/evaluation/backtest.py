"""
Backtesting — evaluates model performance on WC 2018 and 2022.

Methodology:
1. For each World Cup (2018, 2022):
   - Train all models on data BEFORE that tournament
   - Predict all group stage matches
   - Compare predictions to real results
2. Compute metrics:
   - MAE (Mean Absolute Error) on goals
   - Result accuracy (H/D/A)
   - Brier Score (probability calibration)
   - Exact score accuracy
3. Compare all models side by side
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")
sys.path.append("src/data")
sys.path.append("src/models/baseline")
sys.path.append("src/models/ml")
sys.path.append("src/features")


# ─────────────────────────────────────────
# 1. LOAD WC MATCHES
# ─────────────────────────────────────────

def get_wc_matches(results: pd.DataFrame, year: int) -> pd.DataFrame:
    """Get World Cup group stage matches for a given year."""
    wc = results[
        results["tournament"].str.contains(
            "FIFA World Cup", na=False
        ) &
        ~results["tournament"].str.contains(
            "qualification", case=False, na=False
        ) &
        (results["date"].dt.year == year)
    ].copy()

    dates = sorted(wc["date"].unique())
    n = len(dates)
    third = max(1, n // 3)

    date_to_md = {}
    for i, d in enumerate(dates):
        if i < third:
            date_to_md[d] = 1
        elif i < 2 * third:
            date_to_md[d] = 2
        else:
            date_to_md[d] = 3

    wc["matchday"] = wc["date"].map(date_to_md)
    return wc.dropna(subset=["home_goals", "away_goals"])


# ─────────────────────────────────────────
# 2. SIMPLE POISSON PREDICTION
# ─────────────────────────────────────────

def predict_poisson(home_xg: float, away_xg: float) -> dict:
    """Generate scoreline and probabilities from xG values."""
    home_xg = np.clip(home_xg, 0.3, 4.0)
    away_xg = np.clip(away_xg, 0.3, 4.0)

    scorelines = []
    for h in range(7):
        for a in range(7):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            scorelines.append({"h": h, "a": a, "p": p})

    df = pd.DataFrame(scorelines)
    top = df.sort_values("p", ascending=False).iloc[0]

    home_win = df[df["h"] > df["a"]]["p"].sum()
    draw = df[df["h"] == df["a"]]["p"].sum()
    away_win = df[df["h"] < df["a"]]["p"].sum()

    return {
        "pred_home": int(top["h"]),
        "pred_away": int(top["a"]),
        "home_win_prob": home_win,
        "draw_prob": draw,
        "away_win_prob": away_win,
    }


# ─────────────────────────────────────────
# 3. MODEL PREDICTORS
# ─────────────────────────────────────────

def predict_historical_avg(train: pd.DataFrame,
                            match: pd.Series,
                            n: int = 10) -> dict:
    """Historical average prediction."""
    def get_stats(team, before_date):
        mask = (
            ((train["home_team"] == team) |
             (train["away_team"] == team)) &
            (train["date"] < before_date)
        )
        matches = train[mask].dropna(
            subset=["home_goals", "away_goals"]
        ).sort_values("date", ascending=False).head(n)

        if len(matches) == 0:
            return 1.2, 1.2

        scored = []
        conceded = []
        for _, m in matches.iterrows():
            if m["home_team"] == team:
                scored.append(m["home_goals"])
                conceded.append(m["away_goals"])
            else:
                scored.append(m["away_goals"])
                conceded.append(m["home_goals"])
        return np.mean(scored), np.mean(conceded)

    h_scored, h_conceded = get_stats(
        match["home_team"], match["date"]
    )
    a_scored, a_conceded = get_stats(
        match["away_team"], match["date"]
    )

    home_xg = (h_scored + a_conceded) / 2
    away_xg = (a_scored + h_conceded) / 2

    return predict_poisson(home_xg, away_xg)


# ─────────────────────────────────────────
# 4. COMPUTE METRICS
# ─────────────────────────────────────────

def compute_metrics(predictions: list) -> dict:
    """Compute evaluation metrics from a list of predictions."""
    if len(predictions) == 0:
        return {}

    df = pd.DataFrame(predictions)

    home_mae = np.mean(
        np.abs(df["pred_home"] - df["actual_home"])
    )
    away_mae = np.mean(
        np.abs(df["pred_away"] - df["actual_away"])
    )

    df["pred_result"] = df.apply(
        lambda r: "H" if r["pred_home"] > r["pred_away"]
        else ("D" if r["pred_home"] == r["pred_away"] else "A"),
        axis=1
    )
    df["actual_result"] = df.apply(
        lambda r: "H" if r["actual_home"] > r["actual_away"]
        else ("D" if r["actual_home"] == r["actual_away"] else "A"),
        axis=1
    )
    result_acc = (
        df["pred_result"] == df["actual_result"]
    ).mean()

    exact_score = (
        (df["pred_home"] == df["actual_home"]) &
        (df["pred_away"] == df["actual_away"])
    ).mean()

    df["actual_home_win"] = (
        df["actual_result"] == "H"
    ).astype(float)
    df["actual_draw"] = (
        df["actual_result"] == "D"
    ).astype(float)
    df["actual_away_win"] = (
        df["actual_result"] == "A"
    ).astype(float)

    brier = np.mean([
        brier_score_loss(
            df["actual_home_win"],
            df["home_win_prob"].clip(0.01, 0.99)
        ),
        brier_score_loss(
            df["actual_draw"],
            df["draw_prob"].clip(0.01, 0.99)
        ),
        brier_score_loss(
            df["actual_away_win"],
            df["away_win_prob"].clip(0.01, 0.99)
        ),
    ])

    return {
        "home_mae": round(home_mae, 3),
        "away_mae": round(away_mae, 3),
        "result_accuracy": round(result_acc, 3),
        "exact_score_accuracy": round(exact_score, 3),
        "brier_score": round(brier, 3),
        "n_matches": len(df),
    }


# ─────────────────────────────────────────
# 5. RUN BACKTEST
# ─────────────────────────────────────────

def run_backtest(results: pd.DataFrame,
                  test_year: int) -> dict:
    """
    Run backtest for a given World Cup year.
    Train on all data before the tournament.
    Test on tournament group stage matches only.
    """
    print(f"\n{'='*60}")
    print(f"  Backtesting on WC {test_year}")
    print(f"{'='*60}")

    cutoff = pd.Timestamp(f"{test_year}-01-01")
    train = results[results["date"] < cutoff].copy()
    test = get_wc_matches(results, test_year)

    print(f"  Training matches: {len(train)}")
    print(f"  Test matches:     {len(test)}")

    if len(test) == 0:
        print(f"  No WC {test_year} matches found")
        return {}

    feature_cols = [
        "is_neutral", "home_elo", "away_elo", "elo_diff",
        "home_form5_points", "home_form5_avg_scored",
        "home_form5_avg_conceded",
        "away_form5_points", "away_form5_avg_scored",
        "away_form5_avg_conceded",
        "home_form10_points", "home_form10_avg_scored",
        "home_form10_avg_conceded",
        "away_form10_points", "away_form10_avg_scored",
        "away_form10_avg_conceded",
        "form5_points_diff", "form10_points_diff",
        "avg_scored_diff",
        "h2h_matches", "h2h_home_wins", "h2h_away_wins",
        "h2h_draws", "h2h_avg_goals",
    ]

    all_model_results = {}

    # ── Historical Average ───────────────
    print(f"\n  Running Historical Average...")
    ha_preds = []
    for _, match in test.iterrows():
        pred = predict_historical_avg(train, match)
        ha_preds.append({
            **pred,
            "actual_home": int(match["home_goals"]),
            "actual_away": int(match["away_goals"]),
        })
    all_model_results["Historical Average"] = compute_metrics(
        ha_preds
    )
    print(f"  ✅ Done — {len(ha_preds)} matches")

    # ── XGBoost ─────────────────────────
    train_features_path = "data/processed/train_features.csv"
    if os.path.exists(train_features_path):
        print(f"\n  Running XGBoost...")
        try:
            import xgboost as xgb

            all_features = pd.read_csv(
                train_features_path, parse_dates=["date"]
            )

            # FIXED: properly filter train and test sets
            train_feat = all_features[
                all_features["date"] < cutoff
            ].dropna(subset=["home_goals", "away_goals"])

            # Test features — only matches in the test WC year
            test_feat = all_features[
                all_features["date"].dt.year == test_year
            ].dropna(subset=["home_goals", "away_goals"])

            # Further filter to only WC matches
            test_feat = test_feat[
                test_feat["home_team"].isin(
                    test["home_team"].tolist()
                ) &
                test_feat["away_team"].isin(
                    test["away_team"].tolist()
                )
            ]

            for col in feature_cols:
                if col not in train_feat.columns:
                    train_feat[col] = 0
                if col not in test_feat.columns:
                    test_feat[col] = 0

            train_feat[feature_cols] = train_feat[
                feature_cols
            ].fillna(0)
            test_feat[feature_cols] = test_feat[
                feature_cols
            ].fillna(0)

            if len(test_feat) == 0:
                print(f"  ⚠️  No test features found for {test_year}")
            else:
                X_train = train_feat[feature_cols]
                y_home = train_feat["home_goals"]
                y_away = train_feat["away_goals"]

                params = {
                    "n_estimators": 200,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "objective": "count:poisson",
                    "random_state": 42,
                    "n_jobs": -1,
                }
                model_h = xgb.XGBRegressor(**params)
                model_a = xgb.XGBRegressor(**params)
                model_h.fit(X_train, y_home)
                model_a.fit(X_train, y_away)

                xgb_preds = []
                for _, row in test_feat.iterrows():
                    X_pred = pd.DataFrame(
                        [row[feature_cols].fillna(0).to_dict()]
                    )
                    hxg = float(model_h.predict(X_pred)[0])
                    axg = float(model_a.predict(X_pred)[0])
                    pred = predict_poisson(hxg, axg)
                    xgb_preds.append({
                        **pred,
                        "actual_home": int(row["home_goals"]),
                        "actual_away": int(row["away_goals"]),
                    })

                all_model_results["XGBoost"] = compute_metrics(
                    xgb_preds
                )
                print(f"  ✅ Done — {len(xgb_preds)} matches")

        except Exception as e:
            print(f"  ⚠️  XGBoost failed: {e}")

    return all_model_results


# ─────────────────────────────────────────
# 6. PRINT RESULTS TABLE
# ─────────────────────────────────────────

def print_results_table(results: dict, year: int):
    """Print a clean comparison table."""
    print(f"\n  Results for WC {year}:")
    print(
        f"  {'Model':25} {'H-MAE':>7} {'A-MAE':>7} "
        f"{'Result%':>8} {'Exact%':>7} {'Brier':>7} {'N':>4}"
    )
    print(
        f"  {'-'*25} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*4}"
    )

    for model, metrics in sorted(
        results.items(),
        key=lambda x: x[1].get("result_accuracy", 0),
        reverse=True
    ):
        if not metrics:
            continue
        print(
            f"  {model:25} "
            f"{metrics.get('home_mae', 0):>7.3f} "
            f"{metrics.get('away_mae', 0):>7.3f} "
            f"{metrics.get('result_accuracy', 0):>7.1%} "
            f"{metrics.get('exact_score_accuracy', 0):>7.1%} "
            f"{metrics.get('brier_score', 0):>7.3f} "
            f"{metrics.get('n_matches', 0):>4}"
        )


# ─────────────────────────────────────────
# 7. SAVE RESULTS
# ─────────────────────────────────────────

def save_backtest_results(all_results: dict):
    """Save backtest results to CSV."""
    rows = []
    for year, model_results in all_results.items():
        for model, metrics in model_results.items():
            rows.append({
                "year": year,
                "model": model,
                **metrics
            })

    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/backtest_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved backtest results to {out_path}")
    return df


# ─────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────

def run_full_backtest():
    print("=" * 60)
    print("  BACKTESTING — WC 2018 & 2022")
    print("=" * 60)
    print()

    print("Loading data...")
    results = pd.read_csv(
        "data/processed/results_clean.csv",
        parse_dates=["date"]
    )
    print(f"  Loaded {len(results)} matches")

    all_results = {}

    for year in [2018, 2022]:
        year_results = run_backtest(results, year)
        all_results[year] = year_results
        print_results_table(year_results, year)

    # Combined summary
    print(f"\n{'='*60}")
    print("  COMBINED SUMMARY (2018 + 2022)")
    print(f"{'='*60}")

    combined = {}
    for year_results in all_results.values():
        for model, metrics in year_results.items():
            if model not in combined:
                combined[model] = []
            combined[model].append(metrics)

    print(
        f"\n  {'Model':25} {'Avg H-MAE':>10} "
        f"{'Avg Result%':>12} {'Avg Brier':>10}"
    )
    print(
        f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}"
    )

    for model, metrics_list in sorted(
        combined.items(),
        key=lambda x: np.mean(
            [m.get("result_accuracy", 0) for m in x[1]]
        ),
        reverse=True
    ):
        avg_mae = np.mean(
            [m.get("home_mae", 0) for m in metrics_list]
        )
        avg_acc = np.mean(
            [m.get("result_accuracy", 0) for m in metrics_list]
        )
        avg_brier = np.mean(
            [m.get("brier_score", 0) for m in metrics_list]
        )
        print(
            f"  {model:25} "
            f"{avg_mae:>10.3f} "
            f"{avg_acc:>11.1%} "
            f"{avg_brier:>10.3f}"
        )

    save_backtest_results(all_results)


if __name__ == "__main__":
    run_full_backtest()