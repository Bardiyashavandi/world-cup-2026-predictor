# ⚽ FIFA World Cup 2026 — Group Stage Predictor

> A multi-model machine learning system for predicting the 2026 FIFA World Cup group stage. Combines statistical models, gradient boosting, and neural networks into a weighted ensemble with live Bayesian updating after each matchday.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Models](https://img.shields.io/badge/Models-9-green)
![Accuracy](https://img.shields.io/badge/Backtest%20Accuracy-50.5%25-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## 🎯 Project Overview

This project predicts scorelines and win/draw/loss probabilities for all 72 group stage matches of the 2026 FIFA World Cup. What makes it distinctive:

- **Multi-model ensemble** — 9 models from different paradigms combined with learned weights
- **Iterative updating** — predictions improve after each matchday using real results
- **Stakes-aware** — a specialist model activates for MD2/MD3 capturing squad rotation and must-win scenarios
- **Bayesian updating** — team strength estimates update after each match
- **Backtested** — validated on WC 2018 and 2022 before making 2026 predictions

---

## 🏗️ Architecture

    Raw Data (6 sources)
           ↓
    Feature Engineering (24 features per match)
           ↓
    ┌─────────────────────────────────────────┐
    │           9 Prediction Models           │
    │                                         │
    │  Statistical:    ML:        Specialist: │
    │  ├ Static ELO   ├ XGBoost  └ Stakes    │
    │  ├ Dixon-Coles  ├ LightGBM             │
    │  ├ Hist. Avg    ├ Neural Net           │
    │  └ Dynamic ELO  └ Logistic Reg         │
    └─────────────────────────────────────────┘
           ↓
    Weighted Ensemble (matchday-aware weights)
           ↓
    Final Predictions + Confidence Scores
           ↓
    Streamlit Dashboard

---

## 📊 Models

### Statistical Models

| Model | Description |
|-------|-------------|
| **Static ELO** | ELO ratings converted to Poisson xG then scoreline probabilities |
| **Dixon-Coles** | Attack/defense parameters fitted via MLE with time decay and low-score correction |
| **Historical Average** | Recent form-based xG using last N matches |
| **Dynamic ELO** | Live ELO recomputed by replaying all historical matches with variable K-factors |

### ML Models

| Model | Description |
|-------|-------------|
| **XGBoost** | Gradient boosting on 24 features with Poisson loss and exponential time decay weights |
| **LightGBM** | Leaf-wise gradient boosting, faster alternative to XGBoost |
| **Neural Network** | 3-layer FootballNet (24→128→64→32→2) with BatchNorm, Dropout and early stopping |
| **Logistic Regression** | W/D/L classifier providing calibrated outcome probabilities |

### Specialist Model

| Model | Description |
|-------|-------------|
| **Stakes Model** | XGBoost trained only on MD2/MD3 historical WC matches with 18 additional group context features including must_win, already_qualified and elimination_risk |

---

## 🔧 Features

24 features engineered per match:

| Category | Features |
|----------|----------|
| **ELO** | home_elo, away_elo, elo_diff |
| **Form last 5** | points, avg_scored, avg_conceded × 2 teams |
| **Form last 10** | points, avg_scored, avg_conceded × 2 teams |
| **Differentials** | form5_diff, form10_diff, avg_scored_diff |
| **Head-to-Head** | matches, home_wins, away_wins, draws, avg_goals |
| **Context** | is_neutral |

Stakes features added for MD2/MD3: points, position, can_qualify, already_qualified, must_win, elimination_risk, points_needed × 2 teams + matchday

---

## 📈 Performance

Backtested on WC 2018 and 2022 group stage matches:

| Model | Result Accuracy | Exact Score | Brier Score |
|-------|----------------|-------------|-------------|
| XGBoost | **50.5%** | 12.9% | **0.190** |
| Historical Avg | 32.0% | 5.5% | 0.211 |
| Random baseline | 33.3% | — | — |
| Betting markets | ~55% | — | — |

XGBoost achieves 50.5% result accuracy — within 5% of professional betting markets.

---

## 🔄 Live Updating Pipeline

After each matchday:

    # 1. Fill in results
    # Edit data/raw/wc_2026_results.csv

    # 2. Run automated pipeline
    python3 src/updater/update_predictions.py --matchday 1

    # This automatically:
    # Rebuilds features with new results
    # Computes group standings
    # Runs Bayesian strength updater
    # Reruns all 9 models
    # Regenerates ensemble for all remaining matchdays
    # Activates stakes model with real group context

---

## 📁 Repository Structure

    world-cup-2026-predictor/
    ├── data/
    │   ├── raw/                    ← source data + live results
    │   ├── processed/              ← cleaned features + standings
    │   └── predictions/            ← per-model + ensemble outputs
    ├── src/
    │   ├── data/                   ← ingestion + normalization
    │   ├── features/               ← feature engineering + standings
    │   ├── models/
    │   │   ├── baseline/           ← ELO, Dixon-Coles, Hist. Avg, Dynamic ELO
    │   │   └── ml/                 ← XGBoost, LightGBM, NN, Logistic, Stakes
    │   ├── ensemble/               ← weighted ensemble
    │   ├── evaluation/             ← backtesting
    │   └── updater/                ← matchday pipeline + Bayesian updater
    ├── dashboard/
    │   └── app.py                  ← Streamlit dashboard
    ├── notebooks/
    │   └── eda.ipynb               ← exploratory analysis
    └── README.md

---

## 🚀 Getting Started

### 1. Clone and install

    git clone https://github.com/Bardiyashavandi/world-cup-2026-predictor.git
    cd world-cup-2026-predictor
    python3 -m pip install -r requirements.txt

### 2. Download data

Download from Kaggle and place CSVs in data/raw/:

- International football results (martj42/international-football-results)
- WC 2026 Historical ELO ratings (afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings)

### 3. Run pipeline

    python3 src/data/process_data.py
    python3 src/features/build_features.py
    python3 src/models/baseline/elo_model.py
    python3 src/models/baseline/dixon_coles.py
    python3 src/models/baseline/historical_avg.py
    python3 src/models/baseline/dynamic_elo.py
    python3 src/models/ml/xgboost_model.py
    python3 src/models/ml/lightgbm_model.py
    python3 src/models/ml/neural_network.py
    python3 src/models/ml/logistic_regression.py
    python3 src/ensemble/ensemble.py 1
    streamlit run dashboard/app.py

---

## 📦 Requirements

    pandas
    numpy
    scipy
    scikit-learn
    xgboost
    lightgbm
    torch
    streamlit
    plotly
    matplotlib
    seaborn
    requests
    jupytext

---

## 🧠 Key Design Decisions

**Why Poisson for goals?**
Goals are rare independent events — exactly what Poisson models. Validated empirically in the EDA notebook.

**Why time-weighted training?**
Germany 2014 is irrelevant to Germany 2026. Exponential decay weights recent matches up to 5.8x more than oldest.

**Why TimeSeriesSplit for CV?**
Standard k-fold would train on future data to predict the past — data leakage. TimeSeriesSplit always trains on past, validates on future.

**Why a specialist Stakes Model?**
General models ignore group context. France with 6 points in MD3 rotates their squad. Panama needing a win plays completely differently. A specialist trained only on MD2/MD3 historical WC data captures this.

**Why ensemble over single model?**
Each model captures something different — ELO captures strength, Dixon-Coles captures attack/defense, XGBoost captures non-linear feature interactions. The ensemble is more robust than any individual model.

---

## 📜 References

- Dixon, M. and Coles, S. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market
- Elo, A. (1978). The Rating of Chessplayers, Past and Present
- Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System

---

## 👤 Author

**Bardiya Shavandi**

Built as a portfolio project demonstrating end-to-end ML system design across statistics, machine learning, and MLOps.