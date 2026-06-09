import pandas as pd
import os
from normalize_teams import normalize_dataframe

def process_results(path: str = "data/raw/results.csv") -> pd.DataFrame:
    """
    Clean and filter the main international results dataset.
    Keeps only matches from 1990 onwards and relevant tournaments.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Filter from 1990 onwards — older data is less relevant
    df = df[df["date"] >= "2006-01-01"].copy()

    # Normalize team names
    df = normalize_dataframe(df, ["home_team", "away_team"])

    # Add useful derived columns
    df["home_goals"] = df["home_score"]
    df["away_goals"] = df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_difference"] = df["home_score"] - df["away_score"]

    df["result"] = df["goal_difference"].apply(
        lambda x: "H" if x > 0 else ("A" if x < 0 else "D")
    )

    # Flag tournament type
    df["is_world_cup"] = df["tournament"].str.contains(
        "FIFA World Cup", na=False
    ).astype(int)

    df["is_friendly"] = df["tournament"].str.contains(
        "Friendly", na=False
    ).astype(int)

    df = df.sort_values("date").reset_index(drop=True)

    return df


def process_fixtures(path: str = "data/raw/wc_2026_fixtures.csv") -> pd.DataFrame:
    """Clean the 2026 WC fixtures."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = normalize_dataframe(df, ["home_team", "away_team"])
    return df


def process_elo(path: str = "data/processed/elo_latest.csv") -> pd.DataFrame:
    """Load the latest ELO ratings."""
    df = pd.read_csv(path)
    df = normalize_dataframe(df, ["country"])
    return df


def save_all(out_dir: str = "data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    print("Processing results...")
    results = process_results()
    results.to_csv(f"{out_dir}/results_clean.csv", index=False)
    print(f"  Saved {len(results)} matches to results_clean.csv")

    print("Processing fixtures...")
    fixtures = process_fixtures()
    fixtures.to_csv(f"{out_dir}/fixtures_clean.csv", index=False)
    print(f"  Saved {len(fixtures)} fixtures to fixtures_clean.csv")

    print("Processing ELO...")
    elo = process_elo()
    elo.to_csv(f"{out_dir}/elo_clean.csv", index=False)
    print(f"  Saved {len(elo)} teams to elo_clean.csv")

    print("\nAll done. Processed files:")
    for f in os.listdir(out_dir):
        print(f"  data/processed/{f}")


if __name__ == "__main__":
    save_all()