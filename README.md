# 💰 Smart Finance Tracker (India) - ML-powered personal finance

A Streamlit app that **auto-categorises bank transactions, detects subscriptions, forecasts next month's spending,
recommends a budget and assigns a spending persona**. It uses classical ML only (scikit-learn, XGBoost, LightGBM),
with an MLOps setup of MLflow tracking, versioned model files and Docker.

> Built for the *Fundamentals of Computing for Data Science Lab* PBL / Mini Project.
> **All models are trained and evaluated on synthetic data** (see [Data](#-data-honest-statement)). The numbers below are
> *not* real-world performance.

## ✨ Features

| Area | Feature | Type |
|---|---|---|
| Transactions | Auto-categorisation with confidence + "Needs review" badge | **ML** - text classification (TF-IDF + Logistic Regression) |
| | Inline category correction saved to SQLite (kept as future training feedback) | Non-ML |
| | Subscription / recurring-payment detection | **ML** - binary classification (XGBoost) |
| | Autopay for subscriptions (**simulated**, no money moves) | Non-ML |
| Forecasting | Next-month spend per category with a 10-90% range | **ML** - tabular regression on lag/rolling features |
| | Savings forecast, recommended budget with reasons | ML-derived + rules |
| Insights | Spending persona | **ML** - clustering (K-Means) |
| | Weekend vs weekday **hypothesis test** (Mann-Whitney U), fastest-growing categories, top merchants | Statistics |
| | **Unusual transaction alerts** (Q3 + 3×IQR rule per category) | Statistics |
| App | Dashboard, goals with feasibility, CSV / bank-statement import | Non-ML |
| | Admin page: model versions, test metrics vs baseline, comparison tables, plots, user-feedback stats | MLOps |

**Imbalanced data → SMOTE.** Both classifiers face imbalanced classes (categorizer: 12.5 : 1 between the largest and
smallest category; subscription detector: ~6.6% positives). Every candidate model is trained three ways:
**no reweighting**, **`class_weight="balanced"`** (or `scale_pos_weight` for XGBoost) and **SMOTE**. SMOTE sits *inside* an
`imblearn` `Pipeline`, so synthetic samples are generated only from the training folds and never from validation or
test data. The comparison tables below show the outcome honestly: on this data **SMOTE did not beat the unweighted
models**, so it was not selected. It stays in the code as a tested option.

## 🧱 Architecture

```
finance-tracker/
├── app.py                  # Streamlit app (7 pages, one function per page)
├── src/
│   ├── data_prep.py        # public-data loaders, synthetic generator (HCES-calibrated), cleaning
│   ├── features.py         # features for all 4 models - shared by training AND serving
│   ├── train.py            # compare models, MLflow logging, single test evaluation, versioned saving
│   └── services.py         # app logic: DB, predictions, forecast, budget, autopay (simulated), patterns
├── notebooks/eda.ipynb     # 4 EDA sections, each with "Findings and Decisions"
├── data/SOURCES.md         # every dataset, URL, licence, how it is used
├── models/                 # <name>_v<N>.joblib + model_info.json (registry)   [generated]
├── results/                # model_comparison_*.csv, plots, summary.md
├── Dockerfile, docker-compose.yml
```

## 🚀 Quick start

### Local (Python 3.12+)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_prep        # generate + clean data  (~1 min)
python -m src.train            # train all 4 modules    (~12 min on 2 CPU cores)
streamlit run app.py           # http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # http://localhost:5000
```
In the app, type any username, click **Create account**, then **📦 Load demo data** to fill every page with 24
months of a generated user's history.

### Docker
```bash
docker compose build
docker compose run --rm app python -m src.data_prep && docker compose run --rm app python -m src.train
docker compose up            # app → http://localhost:8501, MLflow → http://localhost:5000
```
Models, data, results, MLflow runs and the SQLite DB (`./db/`) are mounted as volumes, so they survive restarts.
Train either locally or in Docker, not both: MLflow stores absolute artifact paths. To switch, delete `mlruns/`.

> Python 3.12 is used instead of 3.11 because the pinned NumPy 2.5 / XGBoost 3.4 / SHAP 0.54 releases require ≥ 3.12.

## 📊 Data (honest statement)

There is **no public dataset of real, labelled, multi-month personal transactions for Indian users**, because that data
is private. This project therefore uses a hybrid approach:

| Source | Kind | Used for |
|---|---|---|
| **MoSPI HCES 2023-24** factsheet / press note | **Official government data** | Calibrating the generator's category shares (urban): food 39.68%, processed food & beverages 11.09%, conveyance 8.46%, durables 6.87%, rent 6.58%. Other groups are approximate (marked in code). |
| `generate_data()` in `src/data_prep.py` | **Generated by us** | 300 users × 24 months (~377k transactions) with Indian merchants and narration styles (`UPI/ZOMATO/483921/Payment`, `ACH D- LIC OF INDIA …`, `NEFT-RENT-JUN`), noise (typos, truncation, case, reference IDs), ~5% ambiguous P2P payments, recurring streams with ±2-day jitter, and hard negatives/positives |
| HF Transaction Dataset, UPI 2024, Credit-card India | Public, **simulated** | Optional loaders and EDA comparison. **Not used in the reported runs**: network access was unavailable when this repo was built. Download them into `data/raw/` (see `data/SOURCES.md`) and re-run. |
| Indian Bank Statements (AgamiAI, Apache 2.0) | Public, synthetic | Style reference for narration formats only |
| Your own bank statement | **Real, private** | Optional: import via the app's CSV upload or `import_bank_csv()`; never committed |

Hidden `archetype` labels (Saver, Lifestyle Spender, ...) are **never given to any model**. They are used only for a
post-hoc sanity check of the clusters (Adjusted Rand Index).

Known deviations from HCES: generated rent is ~15% of spend vs 6.6% in HCES. HCES averages over homeowners who pay no
rent, while a renting household typically pays 15-28% of income. Groceries are correspondingly lower.

## 🧪 Methodology

* **No leakage:** each model is one scikit-learn/imblearn `Pipeline` (preprocessing [+ SMOTE] + model), fitted only on
  training folds. Forecasting uses a **chronological split** (train → 4 validation months → last 3 months test), and
  `check_forecast_leakage()` asserts that features for month *t* are identical with and without data after *t*.
  Subscription CV is **grouped by user** (`StratifiedGroupKFold`), and clustering is fit on 80% of users and checked on the held-out 20%.
* **Select on validation, report on test once**, always against a **baseline**. If two models are within noise (1 std of CV
  score, or 1% RMSE), the faster or simpler one is chosen and the script says so.
* Light tuning with `RandomizedSearchCV` (≤ 10 iterations). Fixed `SEED = 42` everywhere.
* **Quality gate:** `model_info.json` is updated only if the winner beats its baseline.

## 📈 Results (synthetic data, from the last `python -m src.train` run)

### 1. Auto-categorisation - primary metric macro F1 (stratified 5-fold CV on 19,200 rows, test 4,800)

| Model | CV macro F1 | ± std | CV accuracy |
|---|---|---|---|
| Baseline: majority class | 0.0201 | 0.0000 | 0.1778 |
| NaiveBayes | 0.9621 | 0.0026 | 0.9546 |
| NaiveBayes + SMOTE | 0.9603 | 0.0027 | 0.9552 |
| LogReg | 0.9732 | 0.0030 | 0.9648 |
| LogReg + balanced | 0.9596 | 0.0016 | 0.9607 |
| LogReg + SMOTE | 0.9627 | 0.0022 | 0.9617 |
| LinearSVM(cal) | 0.9734 | 0.0025 | 0.9649 |
| LinearSVM(cal) + balanced | 0.9716 | 0.0032 | 0.9634 |
| LinearSVM(cal) + SMOTE | 0.9628 | 0.0021 | 0.9613 |
| RandomForest + balanced | 0.9620 | 0.0035 | 0.9605 |
| LightGBM + balanced | 0.9655 | 0.0028 | 0.9616 |

**Winner: LogReg** (tuned C=10; LinearSVM was within noise and slower). **Test:** macro F1 **0.974**, accuracy 0.967, weighted F1 0.967 vs majority baseline macro F1 0.020 / accuracy 0.178.
Confidence threshold (chosen on out-of-fold validation predictions for ≥98% accuracy among auto-accepted rows): **0.3**.
**Generalisation to merchants never seen in training: macro F1 0.532**. This is the realistic expectation for new merchants, and why the *Needs review* flag and user corrections matter.
SMOTE and class weights both *lowered* macro F1 here (e.g. LogReg 0.973 → 0.963 with SMOTE): even the smallest class has a few hundred distinctive training examples, so oversampling mainly adds interpolated noise to the sparse TF-IDF space.

### 2. Subscription detection - primary metric F1 (positive class), CV grouped by user

| Model | CV F1 | ± std | CV PR-AUC |
|---|---|---|---|
| Baseline: rule (>=3 payments & gap std < 3d) | 0.3724 | 0.0188 | - |
| LogReg | 0.9530 | 0.0081 | 0.9918 |
| LogReg + balanced | 0.8901 | 0.0088 | 0.9893 |
| LogReg + SMOTE | 0.8949 | 0.0125 | 0.9891 |
| RandomForest | 0.9911 | 0.0022 | 0.9997 |
| RandomForest + balanced | 0.9898 | 0.0020 | 0.9996 |
| RandomForest + SMOTE | 0.9895 | 0.0022 | 0.9996 |
| GradientBoosting | 0.9878 | 0.0034 | 0.9993 |
| GradientBoosting + SMOTE | 0.9833 | 0.0044 | 0.9993 |
| XGBoost | 0.9904 | 0.0041 | 0.9996 |
| XGBoost + scale_pos_weight | 0.9898 | 0.0032 | 0.9996 |
| XGBoost + SMOTE | 0.9908 | 0.0026 | 0.9996 |

**Winner: XGBoost** (tuned). **Test:** F1 **0.985**, precision 0.995, recall 0.976, ROC-AUC 0.9999, PR-AUC 0.9989 vs rule baseline F1 0.418 (precision 0.839, recall 0.278).
⚠️ The first version scored F1 0.994. Following the project rule *"if it looks too good, investigate"*, harder cases were added (fixed-price but irregular LPG/wallet top-ups, postpaid bills with varying amounts, house-help salary to a person mixed with random P2P payments, missed months, newly started subscriptions). That dropped the rule baseline from 0.68 to 0.37 and logistic regression to 0.95, but tree models still separate the classes almost perfectly because on generated data the label is almost a deterministic function of "regular gaps + fixed amount". **Treat this as an upper bound.** Real statements will be noisier.

### 3. Next-month forecast - primary metric RMSE (₹), chronological split

| Model | Val RMSE | Val MAE | Val R² |
|---|---|---|---|
| Baseline: last month | 4504 | 1030 | 0.561 |
| Baseline: 3-month mean | 3755 | 1001 | 0.695 |
| Ridge | 3442 | 1070 | 0.744 |
| RandomForest | 3329 | 928 | 0.760 |
| GradientBoosting | 3386 | 938 | 0.752 |
| XGBoost | 3371 | 990 | 0.754 |
| LightGBM | 3351 | 977 | 0.757 |
| LightGBM (tuned) | 3296 | - | - |

**Winner: LightGBM** (within 1% of Random Forest and ~40× faster; tuned). **Test (last 3 months):** RMSE **₹3,443**, MAE ₹931, R² 0.760 vs 3-month-mean baseline RMSE ₹3,881 (R² 0.695) and last-month baseline RMSE ₹4,946, so **11% lower RMSE than the best baseline**.
Honest caveat: sMAPE is *worse* than the baselines (25.4% vs 21.8% on non-zero months), because the model is better on the big amounts that RMSE weights but slightly worse in relative terms on small ones. The largest errors are in lumpy categories (Travel, Health, Shopping; see `results/forecaster_error_by_category.png`). Forecast range = prediction + 10th/90th percentile of validation residuals (₹-1,217 / +₹1,005).

### 4. Spending persona - primary metric silhouette

| Algorithm | k | Silhouette | Davies-Bouldin | Smallest cluster | ARI vs hidden archetype |
|---|---|---|---|---|---|
| KMeans | 6 | 0.188 | 1.726 | 8.8% | 0.703 |
| KMeans | 7 | 0.185 | 1.704 | 7.5% | 0.664 |
| GaussianMixture | 7 | 0.179 | 1.750 | 8.3% | 0.667 |
| KMeans | 5 | 0.176 | 1.831 | 14.6% | 0.566 |
| KMeans | 8 | 0.176 | 1.773 | 6.2% | 0.581 |
| GaussianMixture | 6 | 0.172 | 1.823 | 13.3% | 0.709 |

**Winner: KMeans (k=7)**. k=7 was within 0.01 silhouette of k=6 with a better Davies-Bouldin score (the tie-break rule). Held-out users: silhouette 0.172, ARI vs hidden archetypes 0.560 (reporting only, never used for training). Silhouette ≈ 0.19 shows that spending styles overlap rather than form crisp groups, which is expected for people.
Personas (named automatically from cluster profiles): Debt-Heavy, Saver, Lifestyle Spender, Student / Early Career, Family Budgeter, Moderate Saver, Moderate Student / Early Career.

Full tables: `results/summary.md` and `results/model_comparison_*.csv`. Plots: `results/*.png`.

| | |
|---|---|
| ![](results/categorizer_confusion_matrix.png) | ![](results/forecaster_baseline_vs_model.png) |
| ![](results/subscription_roc_pr.png) | ![](results/persona_pca.png) |


## 🔁 MLOps

* **Git** for code; data and models are reproducible from code (`data_prep` → `train`).
* **MLflow** (`sqlite:///mlruns/mlflow.db`): one experiment per module. Every candidate run logs params, CV metrics,
  training time and git commit, and the winner run also logs test metrics, the model file and plots.
* **Model registry:** `models/<name>_v<N>.joblib` (full pipeline, version auto-increments) + `models/model_info.json`
  (active version, algorithm, metrics vs baseline, threshold, residual quantiles, persona names). The app loads the active versions once (`st.cache_resource`).
* **Same code for training and serving:** the app calls the feature functions in `src/features.py`.
* **Feedback:** user corrections are stored (`predicted_category` vs `category`), and the Admin page shows the correction
  rate. This is a basic model-health signal and the data a future retraining step would use.
* **Docker:** `app` and `mlflow` services.

## ⚠️ Limitations

* Synthetic data: the categorizer and subscription detector score very high because generated merchants and
  schedules are cleaner than real statements. Expect lower scores on real data, especially for merchants never seen before (see the unseen-merchant score).
* Cold start: forecasts need 3 complete months and personas need 3 months of data.
* No real bank integration, no real payments (autopay is simulated), no festival awareness (no reliable festival
  feature data), no drift monitoring or automated retraining (out of scope).
* Very simple login (username only), suitable for a demo, not production.
