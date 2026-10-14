# Reproducibility package

Regenerates every table and figure in the manuscript from two public datasets
and `config.json`. No value is read back from the paper.

## 1. Get the data
- **SmartConvert / Customer Propensity to Purchase** (Powis, Kaggle):
  https://www.kaggle.com/datasets/benpowis/customer-propensity-to-purchase-data
  Use `training_sample.csv`. The competition test file is not used (labels withheld).
- **Balabit Mouse Dynamics Challenge** (Fülöp et al., 2016):
  `git clone https://github.com/balabit/Mouse-Dynamics-Challenge`
  Only `training_files/` is used (10 users, 65 sessions).

## 2. Run
```
pip install -r requirements.txt
python run_pipeline.py --smartconvert training_sample.csv \
    --balabit Mouse-Dynamics-Challenge/training_files --out results
python make_figures.py --smartconvert training_sample.csv \
    --balabit Mouse-Dynamics-Challenge/training_files --out results
```
Runtime is a few minutes on a laptop.

## 3. Declared settings (config.json)
| Setting | Value |
|---|---|
| Seed | 42 |
| Holdout | stratified 20% |
| Out-of-fold scores (calibration, sensitivity) | 5-fold stratified |
| Reward-prior weighting | unweighted (class-weighted reported for comparison) |
| Friction flags | closed_minibasket_click, checked_returns_detail, sort_by, list_size_dropdown, promo_banner_click |
| Tier c -> percentile band | c/6 to (c+1)/6, c = 0..5 (non-overlapping, declared) |
| Friction score | F = 0.6 J + 0.4 V, J and V normalised by pooled 99th percentile |
| Trigger | F above its 70th percentile |
| Balabit filter | Move/Drag events; dt > 0; displacement > 0; velocity <= 10,000 px/s |
| Assumed-effect grid | 0 to 0.30 in steps of 0.05 |
| Bootstrap | 2,000 session resamples (sensitivity); 1,000 (metric CIs) |

## 4. Outputs (`results/`)
| File | Manuscript use |
|---|---|
| `table_discrimination.*` | Discrimination under four availability assumptions, with 95% CIs |
| `leave_one_out_auc.csv` | Leave-one-feature-out audit of the final specification |
| `feature_lift_audit.csv` | Lift for every binary column, flagging any not in the register |
| `fig_reliability.png`, `results.json["calibration"]` | Brier, ECE, slope, intercept, in-the-large gap |
| `deployment_coefficients.csv` | Full-data refit used as the candidate initial score model |
| `results.json["balabit"]` | Balabit summary statistics |
| `results.json["friction_check"]` | Trigger rate and empirical agreement between the Balabit score and the rank-only trigger |
| `table_sensitivity.*`, `fig_sensitivity.png` | Assumption-conditional sensitivity sweep |
| `fig_pipeline.png`, `fig_feature_lift.png`, `fig_coefficients.png`, `fig_balabit_users.png` | Figures 1, 3, 4, 6 (from `make_figures.py`) |
| `table_trigger_comparison.*` | Headroom targeted by friction, random, lowest-score and highest-score triggers |

## 5. Feature availability
`feature_availability_register.csv` records the availability basis for each
retained input. SmartConvert has no within-session timestamps, so event-derived
features are semantically screened, not timestamp-verified.
