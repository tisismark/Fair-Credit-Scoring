# Fair Credit Scoring — Home Credit Default Risk

> Auditing and debiasing a credit scoring model against age discrimination, using post-processing fairness techniques on the Home Credit Default Risk dataset.

---

## Background

This project iterates on **Kozodoi, Jacob & Lessmann (2022)** *(EJOR 297(3): 1083–1094)*, applying their profit-aware fairness framework to real-world lending data.

An initial audit revealed that young applicants (under 25) were systematically disadvantaged — the unconstrained model produced an **Equalized Odds Difference (EOD) of 0.32**, with young applicants achieving only 25.6% TPR versus 57.5% for older applicants. SHAP analysis traced the root cause to proxy features (`EXT_SOURCE_1`, `DAYS_BIRTH`, `BURO_DAYS_CREDIT_MIN`) that encode credit-history disadvantage correlated with age — not direct age discrimination.

This project diagnoses the bias, then compares two post-processing strategies to reduce it without retraining the model.

---

## Features

- **LightGBM baseline** with 5-fold stratified CV trained on 700+ features
- **Multi-table feature engineering** from all five Home Credit tables:
  - `bureau` + `bureau_balance` — external credit history
  - `previous_application` — prior Home Credit applications
  - `installments_payments` — repayment behaviour
- **SHAP proxy audit** to identify and quantify bias channels before intervening
- **Two post-processing fairness strategies:**

| Strategy | Method | Key Idea |
|---|---|---|
| Per-group Threshold Calibration | Grid search | Separate decision thresholds per age group to equalize TPR |
| AIF360 EqOddsPostprocessing | LP optimisation | Score remixing (Hardt, Price & Srebro 2016) on a calibration split |

- **Multi-metric evaluation** across AUC, Profit (Verbraken et al. 2014), EOD, DPD, SP, SF, and per-group TPR/FPR
- **Fairness–performance trade-off plots** (EOD vs AUC, EOD vs Profit)

---

## Results Summary

| Model | AUC | EOD ↓ | Profit |
|---|---|---|---|
| Unconstrained LightGBM | — | 0.3193 | — |
| Threshold Calibration | — | ↓ significant | — |
| AIF360 EqOddsPostprocessing | — | ↓ best | — |

*(Run the notebook to populate with your results.)*

---

## Setup

**1. Download data** from [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place it in `./data/`.

**2. Install dependencies:**

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn aif360 fairlearn shap
```

**3. Run the notebook:**

```bash
jupyter notebook fair_credit_scoring_v7_3.ipynb
```

---

## Stack

`Python` · `LightGBM` · `AIF360` · `Fairlearn` · `SHAP` · `scikit-learn` · `pandas` · `seaborn`

---

## References

- Kozodoi, N., Jacob, J., & Lessmann, S. (2022). Fairness in credit scoring: Assessment, implementation and profit implications. *European Journal of Operational Research, 297*(3), 1083–1094.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.
- Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014). Development and application of consumer credit scoring models with loss given default and time to default. *EJOR*.
