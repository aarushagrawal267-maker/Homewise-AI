# HomeWise AI — Project Report

Every result in this report was produced by the code in this repository, in training run
`RUN-20261113-180324-422408` (13.8 min on a 4-core CPU; Python 3.13, scikit-learn 1.9.1, NumPy 2.4.6, pandas 3.0.5,
XGBoost 3.2.0, LightGBM 4.7.0, SHAP 0.51.0, Optuna 5.0.0). The source files are in `outputs/reports/`, and every
model run is logged in `outputs/experiments/experiments.jsonl`.

---

## 1. Dataset overview

* **Source:** Kaggle *House Prices: Advanced Regression Techniques*, residential sales in Ames, Iowa, 2006–2010.
* **Sizes:** `train.csv` has 1,460 rows × 81 columns; `test.csv` has 1,459 rows × 80 columns (no target).
* **Features:** 79, of which 35 are numerical and 44 categorical. `MSSubClass` is a numerically coded category and
  is treated as categorical.
* **Validation passed:**
  * all 79 expected columns are present and train and test share an identical schema;
  * Ids do not overlap (train 1–1460, test 1461–2919);
  * no duplicate Ids or records;
  * the target is numeric, positive and complete.
* **Missing data:** 7,829 missing cells in 19 training columns.
* **File integrity:** MD5 checksums are recorded and verified on download, in `data_validation_report.json`.

## 2. Exploratory findings

| Statistic | SalePrice |
|---|---|
| Mean / median | $180,921 / $163,000 |
| Min / max | $34,900 / $755,000 |
| Standard deviation | $79,443 |
| Q1 / Q3 (IQR) | $129,975 / $214,000 ($84,025) |
| Skewness (raw → log1p) | 1.88 → 0.12 |
| Kurtosis | 6.54 |

The following findings are generated automatically by `src/eda.py`:

* **Quality** has the strongest relationship with price (Spearman 0.81). The median rises from $50,150 at quality 1
  to $432,390 at quality 10.
* **Living area** has Pearson r = 0.71. A univariate linear fit implies about $10,713 per 100 sq ft, before
  accounting for anything else.
* **Age:** homes built in 2000 or later have a median of $223,750, against $122,000 for those built before 1950
  (Spearman 0.65). Age is confounded with size and quality.
* **Garage:** median price by capacity is $100,000 for 0 cars, $128,000 for 1, $177,750 for 2 and $295,000 for 3.
  The 4-car group (n = 5) has a median of $200,000, a small-sample reversal.
* **Neighbourhoods:** NridgHt ($315,000), NoRidge ($301,500) and StoneBr ($278,000) have the highest medians;
  BrDale ($106,000), IDOTRR ($103,000) and MeadowV ($88,000) the lowest.
* **Top correlations:** OverallQual 0.79, GrLivArea 0.71, GarageCars 0.64, GarageArea 0.62, TotalBsmtSF 0.61.
* **Outliers:**
  * 61 sales lie above the 1.5×IQR fence of $340,038. These are genuine expensive homes and are kept.
  * Ids 524 (4,676 sq ft, $184,750) and 1299 (5,642 sq ft, $160,000) are both *Partial* sales far below the trend.

These are associations in historical data, not causal effects.

## 3. Preprocessing decisions

| Decision | Rationale |
|---|---|
| NA → `"None"` for 15 descriptors (basement, garage, fireplace, pool, fence, alley, misc feature, masonry type) | The data dictionary defines NA as "feature absent" |
| NA → 0 for the matching areas and counts | No basement means 0 sq ft of basement |
| Missing `GarageYrBlt` → `YearBuilt`, plus a `HasGarage` flag | Keeps age features meaningful without inventing a garage |
| Quality and condition scales → ordered integers | Preserves the order Ex > Gd > TA > Fa > Po > None |
| Median / most-frequent imputation inside the pipeline | Real gaps (LotFrontage 17.7%, a few test-only gaps), fitted on training folds only |
| One-hot encoding with `handle_unknown="ignore"` | `MSSubClass` value 150 appears in test but never in train; future inputs may contain new categories too |
| `log1p` of features with \|skew\| > 0.75, plus standard scaling, for linear models only | Trees are scale-invariant; the skewed columns are learned on training data |
| `log1p` target, predictions back-transformed with `expm1` | Target skewness 1.88 → 0.12; the error metric (RMSLE) is on the log scale |
| Remove Ids 524 and 1299 from the training split only | Documented partial-sale anomalies; the validation split is never filtered |
| Stratified (price-decile) 80/20 split, seed 42 | Both sets span the full price range; reproducible |

Leakage is prevented structurally: every fitted transform lives inside the model `Pipeline`, and tests confirm that
the imputer statistics depend only on training rows.

## 4. Feature engineering decisions

Nineteen features in four groups: *size* (TotalSF, TotalBathrooms, TotalPorchSF, LivingLotRatio, AreaPerGarageCar),
*age* (HouseAge, YearsSinceRemodel, GarageAge, WasRemodeled), *quality* (QualityCondition, QualityArea,
QualityBand, ConditionBand) and *facilities* (six Has* flags). All are computed per row from information available at
sale time; none use the target.

Ablation, 5-fold CV RMSLE on the training split:

| Configuration | Ridge | Δ | LightGBM | Δ |
|---|---|---|---|---|
| no engineered features | 0.1154 | +0.0018 | 0.1201 | +0.0023 |
| all engineered features | 0.1136 | — | 0.1178 | — |
| without size | 0.1136 | −0.0000 | 0.1191 | +0.0013 |
| without age | 0.1136 | −0.0000 | 0.1184 | +0.0006 |
| without quality | 0.1154 | +0.0018 | 0.1190 | +0.0011 |
| without facilities | 0.1135 | −0.0001 | 0.1182 | +0.0003 |

* Engineered features reduce CV error for both model families.
* For Ridge the entire gain comes from the *quality* interactions (QualityArea, QualityCondition), which a linear
  model cannot otherwise form.
* LightGBM gains from every group, most from *size* and *quality*.
* The fold SD is about 0.012, so the per-group differences are indicative, not statistically conclusive.

## 5. Model comparison

The table separates **training fit**, **CV** (used for selection) and **validation** (scored once).

| Model | Train RMSLE | CV RMSLE ± SD | Val RMSLE | Val MAE | Val RMSE | Val R² | Val MedAE | Val MAPE | Train s | ms/row |
|---|---|---|---|---|---|---|---|---|---|---|
| **Weighted Ensemble** | 0.0756 | **0.1076 ± 0.0122** | 0.0964 | $12,483 | $19,262 | 0.9421 | $8,143 | 7.03% | 6.42 | 1.12 |
| Stacking Ensemble | 0.0733 | 0.1088 ± 0.0122 | 0.0974 | $12,551 | $19,310 | 0.9418 | $8,175 | 7.08% | 31.34 | 0.75 |
| Lasso | 0.0988 | 0.1113 ± 0.0130 | 0.0977 | $13,309 | $20,294 | 0.9358 | $8,746 | 7.31% | 0.17 | 0.25 |
| XGBoost | 0.0575 | 0.1132 ± 0.0109 | 0.1090 | $13,571 | $21,172 | 0.9301 | $8,200 | 7.80% | 0.73 | 0.18 |
| Gradient Boosting | 0.0611 | 0.1136 ± 0.0090 | 0.1046 | $12,833 | $19,723 | 0.9393 | $8,223 | 7.47% | 5.03 | 0.18 |
| Ridge | 0.0960 | 0.1136 ± 0.0138 | 0.0993 | $13,423 | $20,272 | 0.9359 | $8,890 | 7.39% | 0.10 | 0.17 |
| LightGBM | 0.0247 | 0.1145 ± 0.0121 | 0.1107 | $13,060 | $20,923 | 0.9317 | $7,682 | 7.65% | 1.58 | 0.36 |
| Linear Regression | 0.0879 | 0.1253 ± 0.0071 | 0.1289 | $15,410 | $23,095 | 0.9168 | $9,784 | 9.08% | 0.11 | 0.18 |
| Random Forest | 0.0472 | 0.1289 ± 0.0125 | 0.1219 | $14,410 | $23,477 | 0.9140 | $9,428 | 8.43% | 1.55 | 0.58 |
| Decision Tree | 0.1144 | 0.1813 ± 0.0118 | 0.1508 | $19,978 | $29,702 | 0.8624 | $13,340 | 11.55% | 0.11 | 0.16 |

**Observations**

* **Regularised linear models are very strong here.** Lasso is the best single model by CV. With a log target,
  quality interactions and one-hot neighbourhoods, much of the structure is close to log-linear.
* **Tree ensembles overfit the most.** LightGBM's train RMSLE is 0.025 against 0.115 in CV, yet they add
  complementary signal.
* **The OOF-optimised blend is best.** Lasso 0.503 + Gradient Boosting 0.371 + XGBoost 0.125 (Ridge 0) beats every
  member in CV by 0.0037 RMSLE over Lasso, and also on validation.
* **Stacking gives no further gain** and costs five times the training time, so the simpler weighted blend was
  selected.
* **The ensemble is not automatically better:** it was selected because it had the lowest CV RMSLE, under the same
  procedure as every other model.

## 6. Cross-validation results

Per-fold RMSLE of the selected ensemble: 0.1216, 0.0895, 0.1052, 0.1010, 0.1205, giving a mean of 0.1076 and an SD
of 0.0122. Its out-of-fold predictions over the 1,166 training homes give MAE $13,112, RMSE $20,788 and R² 0.931.

The weights were optimised on these same OOF predictions, so this CV figure is slightly optimistic. Validation
provides the unbiased check.

## 7. Validation results (held-out, 292 homes, scored once)

* **Headline metrics:** MAE $12,483, RMSE $19,262, R² 0.942, RMSLE 0.0964, median absolute error $8,143, MAPE 7.0%.
* **Accuracy bands:** 79.8% of homes are predicted within ±10% of their sale price and 93.5% within ±20%.
* **Residuals** (actual − predicted): mean $775, SD $19,280, range −$55,764 to +$134,513.

Validation RMSLE (0.096) is lower than CV RMSLE (0.108). That is plausible fold-to-fold variation (CV folds range
from 0.090 to 0.122), not evidence of tuning on validation, because validation was never used for any decision.

## 8. Uncertainty calibration

Split-conformal intervals use the selected model's OOF log residuals. These are the intervals the app shows:

| Target coverage | Validation coverage | Log half-width | Median relative width | Mean width |
|---|---|---|---|---|
| 80% | 81.8% | 0.110 | 22.0% (±11.0%) | $39,639 |
| 90% | 90.4% | 0.164 | 32.9% (±16.4%) | $59,139 |
| 95% | 95.2% | 0.216 | 43.6% (±21.8%) | $78,414 |

Achieved coverage matches the targets on unseen data. Widths scale with price because the intervals are built on the
log scale.

## 9. Important features

| Rank | Mean \|SHAP\| (validation, engineered features) | Permutation importance (validation, raw columns, ΔRMSLE) |
|---|---|---|
| 1 | TotalSF 0.046 | OverallQual 0.050 |
| 2 | GrLivArea 0.042 | GrLivArea 0.047 |
| 3 | QualityArea 0.032 | TotalBsmtSF 0.029 |
| 4 | QualityCondition 0.031 | YearBuilt 0.018 |
| 5 | OverallQual 0.029 | LotArea 0.016 |
| 6 | TotalBathrooms 0.025 | OverallCond 0.015 |
| 7 | HouseAge 0.025 | 1stFlrSF 0.015 |
| 8 | LotArea 0.022 | 2ndFlrSF 0.011 |
| 9 | TotalBsmtSF 0.018 | Functional 0.008 |
| 10 | Neighborhood 0.015 | MSZoning 0.007 |

Both methods agree that **size** (total and above-ground area, basement) and **quality** dominate, followed by
**age**, **lot size** and **condition**. SHAP spreads credit across correlated engineered features (TotalSF,
QualityArea), while permutation importance credits the raw source columns. Correlated features share credit in both,
so individual values understate the combined concept.

SHAP additivity was verified, with a maximum reconstruction error of 1.7 × 10⁻⁴ log units for the ensemble.

## 10. Error analysis

Validation error by actual-price quartile:

| Quartile | n | MAE | Median \|% error\| | Mean signed error |
|---|---|---|---|---|
| Q1 (lowest) | 73 | $7,937 | 5.1% | +$2,055 |
| Q2 | 73 | $8,383 | 4.1% | −$1,319 |
| Q3 | 73 | $12,605 | 5.2% | +$2,731 |
| Q4 (highest) | 73 | $21,006 | 5.7% | −$6,566 |

Validation error by sale condition:

| Condition | n | Median \|% error\| |
|---|---|---|
| Normal | 236 | 4.7% |
| Partial (new homes) | 26 | 5.9% |
| Family | 6 | 9.8% |
| Abnorml (foreclosure, short sale…) | 20 | 10.1% |
| Alloca / AdjLand | 2 / 2 | 12.7% / 4.3% |

* **Relative error is roughly constant across price levels**, so dollar errors grow with price. This is consistent
  with a log-scale model.
* **Expensive homes are slightly under-predicted on average** (−$6,566 in Q4): the model shrinks towards typical
  prices where data is sparse.
* **Non-market sales are hardest.** Abnormal and family sales have about twice the error of normal sales; their
  prices reflect circumstances the features do not capture.
* **Largest errors** (`validation_predictions.csv`):
  * Under-prediction: home #899, a brand-new 2009 quality-9 *Partial* sale in Northridge Heights, sold for $611,657
    against a prediction of $477,144 (−$134,513, −22%). Luxury new builds are rare in the data.
  * Over-prediction: home #1212, a quality-8 1988 home in Gilbert, predicted $219,764 against a sale of $164,000
    (+34%), with nothing in its recorded features to explain the low price.

## 11. Kaggle submission

`outputs/submission.csv` holds 1,459 predictions in the required `Id,SalePrice` format, made by the selected model.
The competition test set has no ground truth, so it was **not scored** locally and played no part in model selection.

## 12. Limitations

1. **Scope:** one US city, 2006–2010. The model is not valid elsewhere or for current prices.
2. **Sample size:** 1,460 sales. Rare configurations are poorly represented, and intervals are less reliable there.
3. **Causality:** SHAP, partial dependence and simulator outputs are associations, not effects of interventions.
4. **Optimism:** the ensemble's CV score is slightly optimistic because its weights were fitted on the same OOF
   predictions. Validation is the unbiased reference.
5. **Interval coverage** is marginal, not conditional on a particular home.
6. **App defaults:** fields not on the form come from a representative real sale (#716, 1,350 sq ft, quality 6,
   1974, recorded price $165,000) or from the dataset home the user starts from.

## 13. Future research directions

* Nested CV for unbiased ensemble scores.
* Conditional (locally weighted or Mondrian) conformal intervals by price band or sale condition.
* Monotonic constraints in the boosting models so that what-if responses are always directionally plausible.
* Time-aware validation (train on earlier years, test on later ones) to measure drift across the 2008 crisis.
* Geospatial and neighbourhood-amenity features, and transfer to larger, more recent datasets with proper licensing.
* Model cards and error audits by neighbourhood for fairness and robustness.
