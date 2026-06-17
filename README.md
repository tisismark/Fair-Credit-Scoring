# Fair-Credit-Scoring
Fair Credit Scoring — Home Credit Default Risk
Replicating and extending Kozodoi, Jacob & Lessmann (2022) on the Home Credit Default Risk dataset, with a focus on reducing age-based bias in credit decisions.
Overview
A LightGBM credit scoring model was audited and found to produce a 0.32 Equalized Odds Difference (EOD) against young applicants (under 25) — driven by proxy features like EXT_SOURCE_1 and DAYS_BIRTH that encode credit-history disadvantages correlated with age. This project implements and compares two post-processing fairness strategies to reduce that disparity.
Key Features

LightGBM baseline with 5-fold stratified CV and 700+ engineered features from 5 Home Credit tables (bureau, bureau_balance, previous_application, installments_payments)
Proxy analysis via SHAP to identify the root cause of age bias before intervening
Strategy 1 — Per-group Threshold Calibration: grid search over separate decision thresholds for young/old groups to equalize TPR
Strategy 2 — AIF360 EqOddsPostprocessing (Hardt, Price & Srebro 2016): LP-based score remixing on a calibration split, avoiding the minority-group convergence issues seen with in-processing methods
Multi-metric evaluation: AUC, Profit (Verbraken et al. 2014), EOD, DPD, SP, SF, and per-group TPR/FPR breakdowns
Full fairness–performance trade-off visualisation (EOD vs AUC, EOD vs Profit)

Stack
Python · LightGBM · AIF360 · Fairlearn · SHAP · scikit-learn · pandas · matplotlib
