import pandas as pd
import os
from datetime import datetime


def save_model_predictions(predictions_df: pd.DataFrame, 
                           model_name: str,
                           matchday: int = None):
    """
    Save model predictions with timestamp and model name.
    Creates a versioned predictions file so we can compare
    models against each other and against real results later.
    """
    os.makedirs("data/predictions", exist_ok=True)

    # Add metadata
    predictions_df = predictions_df.copy()
    predictions_df["model"] = model_name
    predictions_df["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Filter by matchday if specified
    if matchday:
        predictions_df = predictions_df[predictions_df["matchday"] == matchday]
        out_path = f"data/predictions/{model_name}_md{matchday}.csv"
    else:
        out_path = f"data/predictions/{model_name}_all.csv"

    predictions_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    return out_path


def load_all_predictions() -> pd.DataFrame:
    """Load and combine all saved predictions for comparison."""
    pred_dir = "data/predictions"
    if not os.path.exists(pred_dir):
        print("No predictions found.")
        return pd.DataFrame()

    dfs = []
    for f in os.listdir(pred_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(f"{pred_dir}/{f}")
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def compare_with_results(predictions_path: str,
                         results_path: str = "data/raw/wc_2026_results.csv") -> pd.DataFrame:
    """
    Compare saved predictions against real results.
    Only compares matches where played=True.
    """
    preds = pd.read_csv(predictions_path)
    results = pd.read_csv(results_path)

    # Only played matches
    played = results[results["played"] == True].copy()

    if len(played) == 0:
        print("No matches played yet.")
        return pd.DataFrame()

    # Merge predictions with real results
    merged = preds.merge(
        played[["match_id", "home_goals", "away_goals"]],
        on="match_id",
        suffixes=("_pred", "_actual")
    )

    # Evaluate
    merged["exact_score"] = (
        (merged["predicted_home_goals"] == merged["home_goals_actual"]) &
        (merged["predicted_away_goals"] == merged["away_goals_actual"])
    )

    merged["actual_result"] = merged.apply(
        lambda r: "H" if r["home_goals_actual"] > r["away_goals_actual"]
        else ("D" if r["home_goals_actual"] == r["away_goals_actual"] else "A"),
        axis=1
    )

    merged["predicted_result"] = merged.apply(
        lambda r: "H" if r["predicted_home_goals"] > r["predicted_away_goals"]
        else ("D" if r["predicted_home_goals"] == r["predicted_away_goals"] else "A"),
        axis=1
    )

    merged["correct_result"] = merged["actual_result"] == merged["predicted_result"]

    # Summary
    print(f"\nModel Evaluation — {predictions_path}")
    print("=" * 50)
    print(f"Matches evaluated  : {len(merged)}")
    print(f"Correct result     : {merged['correct_result'].sum()} / {len(merged)} "
          f"({merged['correct_result'].mean():.1%})")
    print(f"Exact score        : {merged['exact_score'].sum()} / {len(merged)} "
          f"({merged['exact_score'].mean():.1%})")

    return merged


if __name__ == "__main__":
    # Save ELO predictions
    elo_preds = pd.read_csv("data/processed/predictions_elo.csv")
    save_model_predictions(elo_preds, model_name="elo", matchday=1)
    save_model_predictions(elo_preds, model_name="elo")

    print("\nAll current predictions:")
    all_preds = load_all_predictions()
    print(all_preds[["home_team", "away_team", "matchday",
                      "predicted_home_goals", "predicted_away_goals",
                      "model"]].to_string())