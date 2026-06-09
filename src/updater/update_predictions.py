"""
Update predictions after each matchday.

Usage:
    python3 src/updater/update_predictions.py --matchday 1
    python3 src/updater/update_predictions.py --matchday 2
    python3 src/updater/update_predictions.py --matchday 3

Run this after filling in real results in data/raw/wc_2026_results.csv
"""

import argparse
import subprocess
import sys
import os
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def run_step(description: str, command: list) -> bool:
    """Run a script and print status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")

    result = subprocess.run(
        command,
        capture_output=False,
        text=True
    )

    if result.returncode == 0:
        print(f"\n  ✅ Done: {description}")
        return True
    else:
        print(f"\n  ❌ Failed: {description}")
        print(f"  Return code: {result.returncode}")
        return False


def check_results_entered(matchday: int) -> bool:
    """
    Check that results have been entered for the given matchday.
    Returns True if at least some results are marked as played=True.
    """
    results_path = "data/raw/wc_2026_results.csv"

    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return False

    try:
        results = pd.read_csv(results_path)
        if "played" not in results.columns:
            print("❌ Results file missing 'played' column")
            return False

        played = results[
            (results["matchday"] == matchday) &
            (results["played"] == True)
        ]

        if len(played) == 0:
            print(f"❌ No MD{matchday} results marked as played=True")
            print(f"   Fill in data/raw/wc_2026_results.csv first")
            print(f"   Set played=True and fill home_goals/away_goals")
            return False

        print(f"✅ Found {len(played)} MD{matchday} results entered")
        return True

    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return False


def print_results_summary(matchday: int):
    """Print a summary of entered results."""
    results = pd.read_csv("data/raw/wc_2026_results.csv")
    played = results[
        (results["matchday"] == matchday) &
        (results["played"] == True)
    ]

    print(f"\n  MD{matchday} Results Summary:")
    print(f"  {'-'*50}")
    for _, row in played.sort_values("group").iterrows():
        result_str = (
            f"{int(row['home_goals'])}-{int(row['away_goals'])}"
        )
        print(
            f"  Group {row['group']} | "
            f"{row['home_team']:20} "
            f"{result_str:5} "
            f"{row['away_team']}"
        )


def print_standings_summary(matchday: int):
    """Print current group standings."""
    standings_path = f"data/processed/standings_after_md{matchday}.csv"
    if not os.path.exists(standings_path):
        return

    standings = pd.read_csv(standings_path)
    if len(standings) == 0:
        return

    print(f"\n  Standings After MD{matchday}:")
    print(f"  {'-'*50}")
    for group in sorted(standings["group"].unique()):
        print(f"\n  Group {group}:")
        group_df = standings[
            standings["group"] == group
        ].sort_values("group_position")
        for _, row in group_df.iterrows():
            qualified = " ✅" if row.get("points", 0) >= 6 else ""
            print(
                f"    {row['group_position']}. "
                f"{row['team']:22} "
                f"Pts:{int(row['points'])} "
                f"GD:{int(row['goal_difference']):+d} "
                f"GF:{int(row['goals_for'])}"
                f"{qualified}"
            )


def print_predictions_summary(matchday: int):
    """Print a summary of final ensemble predictions for a matchday."""
    pred_path = f"data/predictions/ensemble_md{matchday}.csv"

    if not os.path.exists(pred_path):
        # Fall back to XGBoost predictions
        pred_path = f"data/predictions/xgboost_all.csv"
        if not os.path.exists(pred_path):
            return

        preds = pd.read_csv(pred_path)
        preds = preds[preds["matchday"] == matchday]
        source = "XGBoost"
    else:
        preds = pd.read_csv(pred_path)
        source = "Ensemble"

    if len(preds) == 0:
        return

    print(f"\n  MD{matchday} Predictions ({source}):")
    print(f"  {'-'*60}")
    for _, row in preds.sort_values("group").iterrows():
        flags = ""
        if row.get("home_must_win", 0):
            flags += f" 🔥"
        if row.get("away_must_win", 0):
            flags += f" 🔥"
        if row.get("home_already_qualified", 0):
            flags += f" ✅"
        if row.get("away_already_qualified", 0):
            flags += f" ✅"

        print(
            f"  Group {row['group']} | "
            f"{row['home_team']:20} "
            f"{row['predicted_home_goals']}-"
            f"{row['predicted_away_goals']} "
            f"{row['away_team']:20} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']}"
            f"{flags}"
        )


# ─────────────────────────────────────────
# MAIN UPDATE PIPELINE
# ─────────────────────────────────────────

def run_update(matchday: int):
    """
    Full update pipeline after a matchday's results come in.
    Predicts ALL remaining matchdays after the update.
    """
    start_time = datetime.now()
    remaining_matchdays = [md for md in [1, 2, 3] if md > matchday]

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║   WC 2026 PREDICTOR — POST MD{matchday} UPDATE               ║")
    print(f"║   Will predict: MD{remaining_matchdays}                          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║   Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}                        ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ─── VALIDATE ───────────────────────────────────────────
    print(f"\n📋 Step 0 — Validating MD{matchday} results...")
    if not check_results_entered(matchday):
        print(f"\nHow to enter results:")
        print(f"  1. Open data/raw/wc_2026_results.csv")
        print(f"  2. For each MD{matchday} match fill in:")
        print(f"     home_goals, away_goals, played=True")
        print(f"  3. Run this script again")
        sys.exit(1)

    print_results_summary(matchday)

    # ─── STEP 1: REPROCESS DATA ─────────────────────────────
    success = run_step(
        f"Step 1 — Reprocessing data with MD{matchday} results",
        ["python3", "src/data/process_data.py"]
    )
    if not success:
        print("Pipeline stopped at Step 1")
        sys.exit(1)

    # ─── STEP 2: REBUILD FEATURES ───────────────────────────
    success = run_step(
        "Step 2 — Rebuilding features with updated form/H2H",
        ["python3", "src/features/build_features.py"]
    )
    if not success:
        print("Pipeline stopped at Step 2")
        sys.exit(1)

    # ─── STEP 3: COMPUTE STANDINGS ──────────────────────────
    success = run_step(
        f"Step 3 — Computing group standings after MD{matchday}",
        ["python3", "-c",
         f"import sys; sys.path.append('src/features'); "
         f"from group_standings import save_standings; "
         f"save_standings({matchday})"]
    )
    if not success:
        print("Pipeline stopped at Step 3")
        sys.exit(1)

    print_standings_summary(matchday)

    # ─── STEP 4: RUN ALL BASE MODELS ────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 4 — Running all prediction models")
    print(f"  Predicting remaining matchdays: MD{remaining_matchdays}")
    print(f"{'='*60}")

    base_models = [
        ("ELO Model",
         ["python3", "src/models/baseline/elo_model.py"]),
        ("Dixon-Coles",
         ["python3", "src/models/baseline/dixon_coles.py"]),
        ("Historical Average",
         ["python3", "src/models/baseline/historical_avg.py"]),
        ("Dynamic ELO",
         ["python3", "src/models/baseline/dynamic_elo.py"]),
        ("XGBoost",
         ["python3", "src/models/ml/xgboost_model.py"]),
        ("LightGBM",
         ["python3", "src/models/ml/lightgbm_model.py"]),
        ("Neural Network",
         ["python3", "src/models/ml/neural_network.py"]),
    ]

    failed_models = []
    for model_name, command in base_models:
        print(f"\n  Running {model_name}...")
        result = subprocess.run(
            command, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ {model_name} complete")
        else:
            print(f"  ⚠️  {model_name} had issues — continuing...")
            failed_models.append(model_name)

    # Stakes model runs for each remaining matchday separately
    for remaining_md in remaining_matchdays:
        if remaining_md == 1:
            continue  # stakes model not for MD1
        print(f"\n  Running Stakes Model for MD{remaining_md}...")
        result = subprocess.run(
            ["python3", "src/models/ml/stakes_model.py",
             str(remaining_md)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ Stakes Model MD{remaining_md} complete")
        else:
            print(
                f"  ⚠️  Stakes Model MD{remaining_md} had issues"
            )
            failed_models.append(f"Stakes MD{remaining_md}")

    if failed_models:
        print(
            f"\n  ⚠️  Models with issues: {', '.join(failed_models)}"
        )
        print("  Ensemble will use available models only")

    # ─── STEP 5: RUN ENSEMBLE FOR ALL REMAINING MATCHDAYS ───
    ensemble_path = "src/ensemble/ensemble.py"
    for remaining_md in remaining_matchdays:
        if os.path.exists(ensemble_path):
            run_step(
                f"Step 5 — Running ensemble for MD{remaining_md}",
                ["python3", ensemble_path, str(remaining_md)]
            )
        else:
            print(f"\n⚠️  Ensemble not built yet.")
            print(
                f"   Individual predictions in data/predictions/"
            )

    # ─── DONE ───────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).seconds

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║   ✅ UPDATE COMPLETE — MD{matchday} processed                  ║")
    print(f"║   ⏱  Time elapsed: {elapsed}s                              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║   📁 Predictions saved to data/predictions/          ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Print predictions for all remaining matchdays
    for remaining_md in remaining_matchdays:
        print()
        print_predictions_summary(remaining_md)

    print()
    print("─" * 60)
    print("Next steps:")
    for remaining_md in remaining_matchdays:
        print(
            f"  MD{remaining_md} predictions → "
            f"data/predictions/ensemble_md{remaining_md}.csv"
        )
    if matchday < 3:
        print(f"\n  After MD{matchday+1} matches run:")
        print(
            f"  python3 src/updater/update_predictions.py "
            f"--matchday {matchday+1}"
        )
    else:
        print("\n  Group stage complete! Check final standings.")
    print("─" * 60)


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update WC 2026 predictions after a matchday"
    )
    parser.add_argument(
        "--matchday",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Which matchday just finished (1, 2, or 3)"
    )
    args = parser.parse_args()
    run_update(args.matchday)