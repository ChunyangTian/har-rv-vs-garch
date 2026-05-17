# Forecasting S&P 500 Realized Volatility: HAR-RV vs GARCH(1,1)

**Author:** Chunyang Tian
**Date:** May 2026
**Goal:** Compare two volatility forecasting models, HAR-RV (Corsi 2009) and GARCH(1,1) (Bollerslev 1986), to find out which model is better at predicting next-day realized volatility on S&P 500 data.

## Purpose & Outcome

This project want to answer one question: when we forecast tomorrow's market volatility, should we use the new approach (HAR-RV) or the classic approach (GARCH)?

Both models predict the same thing — tomorrow's volatility — but they use the data in very different ways. HAR-RV directly average the past volatility over 1, 5, and 21 days. GARCH instead use raw returns and update through a recursive formula. I run a strict out-of-sample test on SPY data from 2004 to 2026 to see which one give smaller error.

The main finding is HAR-RV win in both in-sample and out-of-sample test. The difference is about 10% out-of-sample. Both models overfit by similar amount (around 15%), so HAR-RV's advantage is from model structure, not from parameter stability.

## Data and Features

- SPY daily close price from Yahoo Finance, 2004-01-01 to 2026-05-01 (5595 days)
- Daily log return: log(Close_today / Close_yesterday)
- Realized volatility (RV): abs(Return) × sqrt(252)
- This is a simple proxy. The "true" RV need intraday data (5-minute returns), which I don't have free access to. This is a known limitation — see below.

## Methodology

### HAR-RV model

Use three features to predict today's RV:
- RV_d = yesterday's RV
- RV_w = past 5 days' RV average
- RV_m = past 21 days' RV average

All three features use .shift(1) to make sure we don't use today's RV to predict today's RV (this is called "look-ahead bias", a deadly mistake in time series forecasting).

The model is just a linear regression:

RV_today ≈ β_d × RV_d + β_w × RV_w + β_m × RV_m + β₀

The three β weights are learned by ordinary least squares (OLS).

### GARCH(1,1) model

GARCH have a totally different idea. It assume today's variance can be written as:

σ²_today = ω + α × ε²_yesterday + β × σ²_yesterday

where ε is the return shock. This is recursive — yesterday's σ² contain the information from the day before, which contain the day before that, and so on. So GARCH actually use the entire history through this recursion, although older days have very small weight (β^N decay).

The three parameters (ω, α, β) are estimated by maximum likelihood. I use the arch python library.

### Out-of-sample test

For fair comparison, both models use exact same setup:
- Sliding window of 1000 days (about 4 years)
- Refit every single day (drop oldest day, add newest day)
- Predict only the next day (day t+1)

This design make sure the only difference between two models is the model structure itself, not the refit frequency or training data.

## Result

### In-sample MSE (model fit on all data, predict same data)

| Model | MSE |
|---|---|
| HAR-RV | 0.01491 |
| GARCH | 0.01655 |

HAR-RV win by about 10%. But this is in-sample — the model already see all the data when fitting, so this is not a fair prediction test.

### Out-of-sample MSE (the real test)

| Model | MSE |
|---|---|
| HAR-RV | 0.01717 |
| GARCH | 0.01887 |

HAR-RV still win by 9.9%. Since both models use exactly same training setup (1000-day sliding window, refit every day), this 9.9% difference can only come from the model itself, not from the experiment design.

### Overfit analysis

| Model | In-sample | Out-of-sample | Overfit % |
|---|---|---|---|
| HAR-RV | 0.01491 | 0.01717 | 15.2% |
| GARCH | 0.01655 | 0.01887 | 14.0% |

Both models overfit by similar amount. This is interesting because in earlier experiment (when I refit GARCH only every 21 days), GARCH look like it overfit much more (33%). After I switch to daily refit for GARCH, the overfit gap disappear. So part of what look like "GARCH is unstable" actually was "refit not often enough". This is a methodology lesson — refit frequency can hide the true model behavior.

## Interpretation

Why HAR-RV win on SPY data:

1. SPY volatility have strong long memory — last month's vol still matter today. HAR-RV directly use 21-day moving average to catch this.

2. GARCH's α coefficient is only 0.125, meaning only 12.5% weight on yesterday's shock. The rest is β (85%) on yesterday's σ², which itself decay exponentially. So GARCH effectively forget the 21-day-ago information very fast.

3. HAR-RV's regression coefficients (β_d=-0.068, β_w=0.556, β_m=0.353) show that weekly and monthly average dominate the prediction, with daily even having negative weight. This match the idea that one-day-ago vol is too noisy to be useful, but rolling averages are more stable.

## Limitations

1. **RV proxy is rough.** I use abs(daily return) to approximate realized volatility. The standard in literature is to use 5-minute intraday returns, which is more accurate. With true intraday RV, HAR-RV's advantage will probably be even larger (Corsi 2009 report R² around 0.5, mine is only 0.28).

2. **Only one-day horizon tested.** I only predict t+1. For multi-day forecast (like one-week ahead), GARCH's recursive structure may have advantage because it can iterate forward, while HAR-RV need to be re-formulated.

3. **No statistical significance test.** I report MSE difference but don't test whether it is statistically significant (like Diebold-Mariano test). This need future work.

4. **No comparison with simpler baseline.** A trivial baseline like "tomorrow's RV = today's RV" might already be hard to beat. I should compare HAR-RV and GARCH against this baseline too.

## Future Work

- Try with intraday data to get more accurate RV measure
- Test multi-day horizon (k=5, k=21) where GARCH may catch up
- Run Diebold-Mariano test to check if MSE difference is significant
- Combine with regime detection (my other project) — see if HAR-RV's advantage hold in different market regime (calm, volatile, crisis)

## How to Run

pip install yfinance pandas matplotlib numpy arch scikit-learn
python HAR_RV.py

The script will print regression coefficients, GARCH parameters, in-sample MSE, out-of-sample MSE, and show two plots (in-sample and out-of-sample volatility forecast comparison).