import pandas as pd
import numpy as np
import os
import sys

sys.path.append("src/data")
from normalize_teams import normalize_team_name


def load_data():
    results = pd.read_csv("data/processed/results_clean.csv", parse_dates=["date"])
    fixtures = pd.read_csv("data/processed/fixtures_clean.csv", parse_dates=["date"])
    elo = pd.read_csv("data/processed/elo_clean.csv")
    return results, fixtures, elo


def get_team_form(results: pd.DataFrame, team: str, before_date, n: int = 5) -> dict:
    """
    Get a team's form in their last n matches before a given date.
    Looks at matches where team played either home or away.
    """
    # Get all matches for this team before the date
    mask = (
        ((results["home_team"] == team) | (results["away_team"] == team)) &
        (results["date"] < before_date)
    )
    team_matches = results[mask].sort_values("date", ascending=False).head(n)

    if len(team_matches) == 0:
        return {
            "form_points": 0,
            "form_matches": 0,
            "avg_goals_scored": 0,
            "avg_goals_conceded": 0,
            "form_wins": 0,
            "form_draws": 0,
            "form_losses": 0,
        }

    points = 0
    goals_scored = []
    goals_conceded = []
    wins = draws = losses = 0

    for _, match in team_matches.iterrows():
        if match["home_team"] == team:
            scored = match["home_goals"]
            conceded = match["away_goals"]
            result = match["result"]
            win = result == "H"
            draw = result == "D"
        else:
            scored = match["away_goals"]
            conceded = match["home_goals"]
            result = match["result"]
            win = result == "A"
            draw = result == "D"

        goals_scored.append(scored)
        goals_conceded.append(conceded)

        if win:
            points += 3
            wins += 1
        elif draw:
            points += 1
            draws += 1
        else:
            losses += 1

    return {
        "form_points": points,
        "form_matches": len(team_matches),
        "avg_goals_scored": np.mean(goals_scored),
        "avg_goals_conceded": np.mean(goals_conceded),
        "form_wins": wins,
        "form_draws": draws,
        "form_losses": losses,
    }


def get_h2h(results: pd.DataFrame, home_team: str, away_team: str, before_date, n: int = 5) -> dict:
    """
    Get head-to-head record between two teams before a given date.
    """
    mask = (
        (
            ((results["home_team"] == home_team) & (results["away_team"] == away_team)) |
            ((results["home_team"] == away_team) & (results["away_team"] == home_team))
        ) &
        (results["date"] < before_date)
    )
    h2h = results[mask].sort_values("date", ascending=False).head(n)

    if len(h2h) == 0:
        return {
            "h2h_matches": 0,
            "h2h_home_wins": 0,
            "h2h_away_wins": 0,
            "h2h_draws": 0,
            "h2h_avg_goals": 0,
        }

    home_wins = draws = away_wins = 0
    total_goals = []

    for _, match in h2h.iterrows():
        total_goals.append(match["total_goals"])
        if match["home_team"] == home_team:
            if match["result"] == "H": home_wins += 1
            elif match["result"] == "D": draws += 1
            else: away_wins += 1
        else:
            if match["result"] == "A": home_wins += 1
            elif match["result"] == "D": draws += 1
            else: away_wins += 1

    return {
        "h2h_matches": len(h2h),
        "h2h_home_wins": home_wins,
        "h2h_away_wins": away_wins,
        "h2h_draws": draws,
        "h2h_avg_goals": np.mean(total_goals),
    }


def get_elo_for_team(elo: pd.DataFrame, team: str) -> float:
    """Get current ELO rating for a team."""
    row = elo[elo["country"] == team]
    if len(row) == 0:
        return 1500.0  # default neutral ELO
    return float(row["rating"].values[0])


def build_match_features(results: pd.DataFrame, elo: pd.DataFrame, 
                          home_team: str, away_team: str, 
                          match_date, is_neutral: int = 1) -> dict:
    """
    Build the full feature vector for a single match.
    """
    # ELO features
    home_elo = get_elo_for_team(elo, home_team)
    away_elo = get_elo_for_team(elo, away_team)
    elo_diff = home_elo - away_elo

    # Form features - last 5 and last 10
    home_form5 = get_team_form(results, home_team, match_date, n=5)
    away_form5 = get_team_form(results, away_team, match_date, n=5)
    home_form10 = get_team_form(results, home_team, match_date, n=10)
    away_form10 = get_team_form(results, away_team, match_date, n=10)

    # H2H features
    h2h = get_h2h(results, home_team, away_team, match_date)

    features = {
        # Team identifiers
        "home_team": home_team,
        "away_team": away_team,
        "date": match_date,
        "is_neutral": is_neutral,

        # ELO
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,

        # Form last 5
        "home_form5_points": home_form5["form_points"],
        "home_form5_avg_scored": home_form5["avg_goals_scored"],
        "home_form5_avg_conceded": home_form5["avg_goals_conceded"],
        "away_form5_points": away_form5["form_points"],
        "away_form5_avg_scored": away_form5["avg_goals_scored"],
        "away_form5_avg_conceded": away_form5["avg_goals_conceded"],

        # Form last 10
        "home_form10_points": home_form10["form_points"],
        "home_form10_avg_scored": home_form10["avg_goals_scored"],
        "home_form10_avg_conceded": home_form10["avg_goals_conceded"],
        "away_form10_points": away_form10["form_points"],
        "away_form10_avg_scored": away_form10["avg_goals_scored"],
        "away_form10_avg_conceded": away_form10["avg_goals_conceded"],

        # Form differential (home minus away) — useful single features
        "form5_points_diff": home_form5["form_points"] - away_form5["form_points"],
        "form10_points_diff": home_form10["form_points"] - away_form10["form_points"],
        "avg_scored_diff": home_form5["avg_goals_scored"] - away_form5["avg_goals_scored"],

        # H2H
        "h2h_matches": h2h["h2h_matches"],
        "h2h_home_wins": h2h["h2h_home_wins"],
        "h2h_away_wins": h2h["h2h_away_wins"],
        "h2h_draws": h2h["h2h_draws"],
        "h2h_avg_goals": h2h["h2h_avg_goals"],
    }

    return features


def build_training_dataset(results: pd.DataFrame, elo: pd.DataFrame,
                            min_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Build the full training dataset from historical results.
    Each row = one match with all features + target variables.
    Only uses matches from min_date onwards to keep training fast.
    """
    training_matches = results[results["date"] >= min_date].copy()
    print(f"Building features for {len(training_matches)} training matches...")

    rows = []
    for i, (_, match) in enumerate(training_matches.iterrows()):
        if i % 500 == 0:
            print(f"  Processing match {i}/{len(training_matches)}...")

        features = build_match_features(
            results=results,
            elo=elo,
            home_team=match["home_team"],
            away_team=match["away_team"],
            match_date=match["date"],
            is_neutral=int(match["neutral"]) if "neutral" in match else 1,
        )

        # Add target variables
        features["home_goals"] = match["home_goals"]
        features["away_goals"] = match["away_goals"]
        features["result"] = match["result"]
        features["total_goals"] = match["total_goals"]
        features["is_world_cup"] = match["is_world_cup"]

        rows.append(features)

    df = pd.DataFrame(rows)
    return df


def build_prediction_dataset(fixtures: pd.DataFrame, results: pd.DataFrame,
                              elo: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the 2026 WC fixtures we want to predict.
    """
    print(f"Building features for {len(fixtures)} WC 2026 fixtures...")
    rows = []
    for _, fixture in fixtures.iterrows():
        features = build_match_features(
            results=results,
            elo=elo,
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            match_date=fixture["date"],
            is_neutral=1,  # all WC matches on neutral ground
        )
        features["match_id"] = fixture["match_id"]
        features["group"] = fixture["group"]
        features["matchday"] = fixture["matchday"]
        features["city"] = fixture["city"]
        rows.append(features)

    return pd.DataFrame(rows)


def save_datasets():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading data...")
    results, fixtures, elo = load_data()

    print("\nBuilding training dataset...")
    train_df = build_training_dataset(results, elo, min_date="2010-01-01")
    train_df.to_csv("data/processed/train_features.csv", index=False)
    print(f"Saved {len(train_df)} rows to train_features.csv")

    print("\nBuilding prediction dataset...")
    pred_df = build_prediction_dataset(fixtures, results, elo)
    pred_df.to_csv("data/processed/predict_features.csv", index=False)
    print(f"Saved {len(pred_df)} rows to predict_features.csv")

    print("\nFeature columns:")
    feature_cols = [c for c in train_df.columns if c not in
                    ["home_team", "away_team", "date", "home_goals",
                     "away_goals", "result", "total_goals", "is_world_cup"]]
    for col in feature_cols:
        print(f"  {col}")


if __name__ == "__main__":
    save_datasets()