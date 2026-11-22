"""Train, compare, evaluate and save all four models.

Run:  python -m src.train            (all modules)
      python -m src.train forecaster (one module)
Imbalance handling: for the two classifiers every linear/tree model is tried with no reweighting,
class_weight="balanced" and SMOTE. SMOTE lives *inside* an imblearn Pipeline, so it is applied only to the
training folds during CV and never to validation/test data (no leakage).
"""
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, adjusted_rand_score, average_precision_score, classification_report,
                             confusion_matrix, davies_bouldin_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier, XGBRegressor

from src import features as F
from src.data_prep import PROCESSED, RAW, ROOT, SEED, map_external_categories, clean_transactions

warnings.filterwarnings("ignore")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
INFO_PATH = MODELS / "model_info.json"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"
N_JOBS = 2
CAT_SAMPLE = 24000  # rows used for the categorizer (keeps CV fast on a laptop; stratified so imbalance is kept)
sns.set_theme(style="whitegrid")


# ====================================================================================== helpers
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def setup_mlflow(module):
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    name = f"finance-{module}"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns" / "artifacts" / module).as_uri())
    mlflow.set_experiment(name)


def log_candidate(name, params, cv_metrics, seconds):
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("git_commit", git_commit())
        mlflow.set_tag("role", "candidate")
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({f"cv_{k}": float(v) for k, v in cv_metrics.items()})
        mlflow.log_metric("train_seconds", seconds)


def log_winner(name, params, test_metrics, model_path, figures):
    with mlflow.start_run(run_name=f"WINNER-{name}"):
        mlflow.set_tag("git_commit", git_commit())
        mlflow.set_tag("role", "winner")
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items() if np.isscalar(v)})
        if model_path:
            mlflow.log_artifact(str(model_path), artifact_path="model")
        for f in figures:
            if Path(f).exists():
                mlflow.log_artifact(str(f), artifact_path="plots")


def savefig(name):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def load_info():
    return json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else {}


def save_model(module, pipeline, entry, beats_baseline):
    """Versioned joblib + registry update. Quality gate: registry only changes if the winner beats the baseline."""
    MODELS.mkdir(exist_ok=True)
    versions = [int(p.stem.split("_v")[-1]) for p in MODELS.glob(f"{module}_v*.joblib")]
    v = max(versions, default=0) + 1
    path = MODELS / f"{module}_v{v}.joblib"
    joblib.dump(pipeline, path, compress=3)
    if not beats_baseline:
        print(f"[{module}] QUALITY GATE: winner did not beat baseline; saved {path.name} but registry NOT updated")
        return path
    info = load_info()
    info[module] = dict(version=v, file=path.name, trained_at=datetime.now().isoformat(timespec="seconds"),
                        git_commit=git_commit(), **entry)
    INFO_PATH.write_text(json.dumps(info, indent=2, default=float))
    print(f"[{module}] saved {path.name} and updated model_info.json")
    return path


def make_pipe(model, smote=False, prep=None, k_neighbors=5):
    steps = [("prep", prep)]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=k_neighbors)))
    steps.append(("model", model))
    return Pipeline(steps)


def load_clean():
    df = pd.read_csv(PROCESSED / "transactions_clean.csv", parse_dates=["date"])
    df["clean_desc"] = df["clean_desc"].fillna("unknown")
    return df


SUMMARY = {}


# ====================================================================================== 4.1 categorizer
def train_categorizer(df):
    module = "categorizer"
    setup_mlflow(module)
    data = df[df["type"] == "debit"]
    data, _ = train_test_split(data, train_size=CAT_SAMPLE, stratify=data["category"], random_state=SEED)
    X, y = F.categorizer_frame(data), data["category"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    prep = F.categorizer_preprocessor

    candidates = {
        "Baseline: majority class": make_pipe(DummyClassifier(strategy="most_frequent"), prep=prep()),
        "NaiveBayes": make_pipe(_mnb(), prep=prep()),
        "NaiveBayes + SMOTE": make_pipe(_mnb(), smote=True, prep=prep()),
        "LogReg": make_pipe(LogisticRegression(max_iter=2000, C=5), prep=prep()),
        "LogReg + balanced": make_pipe(LogisticRegression(max_iter=2000, C=5, class_weight="balanced"), prep=prep()),
        "LogReg + SMOTE": make_pipe(LogisticRegression(max_iter=2000, C=5), smote=True, prep=prep()),
        "LinearSVM(cal)": make_pipe(CalibratedClassifierCV(LinearSVC(C=0.5), cv=3), prep=prep()),
        "LinearSVM(cal) + balanced": make_pipe(CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3), prep=prep()),
        "LinearSVM(cal) + SMOTE": make_pipe(CalibratedClassifierCV(LinearSVC(C=0.5), cv=3), smote=True, prep=prep()),
        "RandomForest + balanced": make_pipe(RandomForestClassifier(200, class_weight="balanced", n_jobs=N_JOBS,
                                                                    random_state=SEED, min_samples_leaf=1), prep=prep()),
        "LightGBM + balanced": make_pipe(LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                                        colsample_bytree=0.2, class_weight="balanced",
                                                        random_state=SEED, verbose=-1, n_jobs=N_JOBS), prep=prep()),
    }
    rows = _compare_classifiers(module, candidates, X_tr, y_tr, cv, scoring={
        "macro_f1": "f1_macro", "accuracy": "accuracy", "weighted_f1": "f1_weighted"}, primary="macro_f1")
    best_name = _pick(rows, "macro_f1", exclude="Baseline")
    best = candidates[best_name]

    # light tuning of the winner (<= 10 iterations)
    grid = _cat_grid(best_name)
    if grid:
        search = RandomizedSearchCV(best, grid, n_iter=min(10, _grid_size(grid)), cv=cv, scoring="f1_macro",
                                    random_state=SEED, n_jobs=N_JOBS)
        search.fit(X_tr, y_tr)
        print(f"[{module}] tuned {best_name}: {search.best_params_} cv macro F1 {search.best_score_:.4f}")
        best = search.best_estimator_
        rows.append(dict(model=f"{best_name} (tuned)", cv_macro_f1=search.best_score_, params=str(search.best_params_)))

    # confidence threshold on out-of-fold (validation) predictions
    proba = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=N_JOBS)
    classes = np.unique(y_tr)
    conf, pred = proba.max(1), classes[proba.argmax(1)]
    ths = np.round(np.arange(0.30, 0.95, 0.05), 2)
    curve = pd.DataFrame([dict(threshold=t, coverage=(conf >= t).mean(),
                               accuracy=(pred[conf >= t] == y_tr[conf >= t]).mean() if (conf >= t).any() else np.nan)
                          for t in ths])
    ok = curve[curve["accuracy"] >= 0.98]  # auto-accept only where >=98% of accepted predictions are right
    threshold = float(ok["threshold"].min()) if len(ok) else 0.6
    print(f"[{module}] confidence threshold chosen on validation: {threshold}")

    # final fit on all training data, single test evaluation
    best.fit(X_tr, y_tr)
    y_hat = best.predict(X_te)
    test = dict(macro_f1=f1_score(y_te, y_hat, average="macro"), accuracy=accuracy_score(y_te, y_hat),
                weighted_f1=f1_score(y_te, y_hat, average="weighted"))
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te)
    base_m = dict(macro_f1=f1_score(y_te, base, average="macro"), accuracy=accuracy_score(y_te, base))
    print(f"[{module}] TEST {best_name}: {test} | baseline {base_m}")
    report = classification_report(y_te, y_hat, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(RESULTS / "categorizer_classification_report.csv")

    test["unseen_merchant_macro_f1"] = _unseen_merchant_check(best, X, y, data["merchant"].to_numpy())
    figs = _categorizer_plots(best, X_tr, y_tr, y_te, y_hat, report, curve, threshold)
    ext = _external_check(best)
    if ext:
        test["external_macro_f1"] = ext
    pd.DataFrame(rows).to_csv(RESULTS / f"model_comparison_{module}.csv", index=False)
    entry = dict(algorithm=best_name, primary_metric="macro_f1", test_metrics=test, baseline_metrics=base_m,
                 confidence_threshold=threshold, n_train=len(X_tr), n_test=len(X_te))
    path = save_model(module, best, entry, test["macro_f1"] > base_m["macro_f1"])
    log_winner(best_name, {"threshold": threshold}, test, path, figs)
    SUMMARY[module] = dict(rows=rows, winner=best_name, test=test, baseline=base_m, primary="macro_f1")


def _unseen_merchant_check(model, X, y, merchants):
    """Diagnostic (not used for selection): train on some merchants, test on merchants never seen in training."""
    from sklearn.base import clone
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(X, y, merchants))
    m = clone(model).fit(X.iloc[tr], y[tr])
    score = f1_score(y[te], m.predict(X.iloc[te]), average="macro")
    print(f"[categorizer] generalisation to UNSEEN merchants: macro F1 = {score:.3f} "
          f"({len(set(merchants[te]))} held-out merchants)")
    return score


def _mnb():
    from sklearn.naive_bayes import MultinomialNB
    return MultinomialNB(alpha=0.1)


def _cat_grid(name):
    if name.startswith("LogReg"):
        return {"model__C": [0.5, 1, 2, 5, 10, 20]}
    if name.startswith("LinearSVM"):
        return {"model__estimator__C": [0.1, 0.25, 0.5, 1, 2]}
    if name.startswith("NaiveBayes"):
        return {"model__alpha": [0.01, 0.05, 0.1, 0.3, 1.0]}
    return None


def _grid_size(grid):
    return int(np.prod([len(v) for v in grid.values()]))


def _compare_classifiers(module, candidates, X, y, cv, scoring, primary, groups=None):
    rows = []
    for name, pipe in candidates.items():
        t = time.time()
        res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=N_JOBS)
        secs = time.time() - t
        m = {k: res[f"test_{k}"].mean() for k in scoring}
        sd = res[f"test_{primary}"].std()
        rows.append(dict(model=name, **{f"cv_{k}": v for k, v in m.items()}, **{f"cv_{primary}_std": sd},
                         train_seconds=round(secs, 1)))
        log_candidate(name, {"smote": "SMOTE" in name, "balanced": "balanced" in name}, m, secs)
        print(f"[{module}] {name:32s} cv {primary}={m[primary]:.4f} ± {sd:.4f} ({secs:.0f}s)")
    return rows


def _pick(rows, primary, exclude="Baseline", higher=True):
    """Best on CV; if another model is within 1 std (noise), prefer the faster one."""
    df = pd.DataFrame([r for r in rows if not r["model"].startswith(exclude) and f"cv_{primary}_std" in r])
    df = df.sort_values(f"cv_{primary}", ascending=not higher)
    top = df.iloc[0]
    tol = top[f"cv_{primary}_std"]
    close = df[(df[f"cv_{primary}"] - top[f"cv_{primary}"]).abs() <= tol]
    choice = close.sort_values("train_seconds").iloc[0]
    if choice["model"] != top["model"]:
        print(f"  {choice['model']} is within noise of {top['model']} and faster -> choosing it")
    return choice["model"]


def _categorizer_plots(best, X_tr, y_tr, y_te, y_hat, report, curve, threshold):
    figs = []
    labels = sorted(np.unique(np.concatenate([y_te, y_hat])))
    cm = confusion_matrix(y_te, y_hat, labels=labels, normalize="true")
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Categorizer - normalized confusion matrix (test)")
    figs.append(savefig("categorizer_confusion_matrix.png"))
    f1s = pd.Series({k: v["f1-score"] for k, v in report.items() if k in labels}).sort_values()
    plt.figure(figsize=(8, 6)); f1s.plot.barh(color="#4F46E5"); plt.xlabel("F1 (test)"); plt.title("Per-class F1")
    figs.append(savefig("categorizer_per_class_f1.png"))
    cmc = pd.DataFrame(confusion_matrix(y_te, y_hat, labels=labels), index=labels, columns=labels)
    pairs = [(t, p, cmc.loc[t, p]) for t in labels for p in labels if t != p and cmc.loc[t, p] > 0]
    pd.DataFrame(sorted(pairs, key=lambda x: -x[2])[:10], columns=["true", "predicted", "count"]).to_csv(
        RESULTS / "categorizer_confused_pairs.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(curve["threshold"], curve["accuracy"], marker="o", label="accuracy of accepted")
    plt.plot(curve["threshold"], curve["coverage"], marker="s", label="coverage (share auto-accepted)")
    plt.axvline(threshold, ls="--", c="grey"); plt.xlabel("confidence threshold"); plt.legend()
    plt.title("Confidence vs accuracy (validation, out-of-fold)")
    figs.append(savefig("categorizer_confidence_curve.png"))
    # top words per class from a linear model (interpretation only)
    lin = make_pipe(LogisticRegression(max_iter=2000, C=5), prep=F.categorizer_preprocessor()).fit(X_tr, y_tr)
    names = lin.named_steps["prep"].get_feature_names_out()
    coef = lin.named_steps["model"].coef_
    top = {c: [names[i].split("__", 1)[1] for i in np.argsort(coef[k])[-8:][::-1] if names[i].startswith("word")][:6]
           for k, c in enumerate(lin.named_steps["model"].classes_)}
    pd.DataFrame({k: pd.Series(v) for k, v in top.items()}).to_csv(RESULTS / "categorizer_top_words.csv", index=False)
    return figs


def _external_check(model):
    """Optional: score on public dataset #1 (if downloaded) after mapping its categories."""
    path = RAW / "hf_transactions.csv"
    if not path.exists():
        return None
    ext = pd.read_csv(path)
    cols = {c.lower(): c for c in ext.columns}
    desc_c, cat_c = cols.get("description"), cols.get("category")
    if not desc_c or not cat_c:
        print("[categorizer] external dataset columns not recognised - skipped")
        return None
    amt_c, date_c = cols.get("amount"), cols.get("date")
    e = pd.DataFrame({"user_id": "ext", "description": ext[desc_c],
                      "amount": pd.to_numeric(ext[amt_c], errors="coerce") if amt_c else 100.0,
                      "date": ext[date_c] if date_c else "2024-01-01", "type": "debit"})
    e = clean_transactions(e, verbose=False)
    y = map_external_categories(ext.loc[e.index], cat_c)
    score = f1_score(y, model.predict(F.categorizer_frame(e)), average="macro")
    print(f"[categorizer] external check on public dataset #1: macro F1 = {score:.3f}")
    return score


# ====================================================================================== 4.2 subscriptions
def subscription_dataset(df):
    feats = F.subscription_features(df)
    labels = pd.read_csv(RAW / "recurring_labels.csv")
    feats = feats.merge(labels, on=["user_id", "merchant"], how="left")
    feats["is_recurring"] = feats["is_recurring"].fillna(0).astype(int)
    return feats


def baseline_rule(X):
    return ((X["n_payments"] >= 3) & (X["gap_std"] < 3)).astype(int).to_numpy()


def train_subscription(df):
    module = "subscription"
    setup_mlflow(module)
    data = subscription_dataset(df)
    X, y, groups = data[F.SUB_FEATURES], data["is_recurring"].to_numpy(), data["user_id"].to_numpy()
    print(f"[{module}] rows={len(data)} positive share={y.mean():.3f} (imbalanced -> SMOTE / class weights compared)")
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, groups))
    X_tr, X_te, y_tr, y_te, g_tr = X.iloc[tr], X.iloc[te], y[tr], y[te], groups[tr]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    prep = F.subscription_preprocessor
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    candidates = {
        "LogReg": make_pipe(LogisticRegression(max_iter=2000), prep=prep()),
        "LogReg + balanced": make_pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), prep=prep()),
        "LogReg + SMOTE": make_pipe(LogisticRegression(max_iter=2000), smote=True, prep=prep()),
        "RandomForest": make_pipe(RandomForestClassifier(300, n_jobs=N_JOBS, random_state=SEED), prep=prep()),
        "RandomForest + balanced": make_pipe(RandomForestClassifier(300, class_weight="balanced", n_jobs=N_JOBS,
                                                                    random_state=SEED), prep=prep()),
        "RandomForest + SMOTE": make_pipe(RandomForestClassifier(300, n_jobs=N_JOBS, random_state=SEED), smote=True, prep=prep()),
        "GradientBoosting": make_pipe(GradientBoostingClassifier(random_state=SEED), prep=prep()),
        "GradientBoosting + SMOTE": make_pipe(GradientBoostingClassifier(random_state=SEED), smote=True, prep=prep()),
        "XGBoost": make_pipe(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, random_state=SEED,
                                           n_jobs=N_JOBS, eval_metric="logloss"), prep=prep()),
        "XGBoost + scale_pos_weight": make_pipe(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                                              scale_pos_weight=spw, random_state=SEED, n_jobs=N_JOBS,
                                                              eval_metric="logloss"), prep=prep()),
        "XGBoost + SMOTE": make_pipe(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, random_state=SEED,
                                                   n_jobs=N_JOBS, eval_metric="logloss"), smote=True, prep=prep()),
    }
    scoring = {"f1": "f1", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc", "pr_auc": "average_precision"}
    # baseline rule CV
    bf = [f1_score(y_tr[v], baseline_rule(X_tr.iloc[v])) for _, v in cv.split(X_tr, y_tr, g_tr)]
    rows = [dict(model="Baseline: rule (>=3 payments & gap std < 3d)", cv_f1=np.mean(bf), cv_f1_std=np.std(bf))]
    print(f"[{module}] baseline rule cv f1={np.mean(bf):.4f}")
    rows += _compare_classifiers(module, candidates, X_tr, y_tr, cv, scoring, "f1", groups=g_tr)
    best_name = _pick(rows, "f1")
    best = candidates[best_name]
    grid = _sub_grid(best_name)
    if grid:
        search = RandomizedSearchCV(best, grid, n_iter=min(10, _grid_size(grid)), cv=cv, scoring="f1",
                                    random_state=SEED, n_jobs=N_JOBS)
        search.fit(X_tr, y_tr, groups=g_tr)
        print(f"[{module}] tuned {best_name}: {search.best_params_} cv F1 {search.best_score_:.4f}")
        best = search.best_estimator_
        rows.append(dict(model=f"{best_name} (tuned)", cv_f1=search.best_score_, params=str(search.best_params_)))
    best.fit(X_tr, y_tr)
    proba = best.predict_proba(X_te)[:, 1]
    y_hat = (proba >= 0.5).astype(int)
    test = dict(f1=f1_score(y_te, y_hat), precision=precision_score(y_te, y_hat), recall=recall_score(y_te, y_hat),
                roc_auc=roc_auc_score(y_te, proba), pr_auc=average_precision_score(y_te, proba))
    b = baseline_rule(X_te)
    base_m = dict(f1=f1_score(y_te, b), precision=precision_score(y_te, b), recall=recall_score(y_te, b))
    print(f"[{module}] TEST {best_name}: {test} | baseline {base_m}")
    figs = _subscription_plots(best, X_tr, X_te, y_te, y_hat, proba)
    pd.DataFrame(rows).to_csv(RESULTS / f"model_comparison_{module}.csv", index=False)
    entry = dict(algorithm=best_name, primary_metric="f1", test_metrics=test, baseline_metrics=base_m,
                 decision_threshold=0.5, features=F.SUB_FEATURES)
    path = save_model(module, best, entry, test["f1"] > base_m["f1"])
    log_winner(best_name, {}, test, path, figs)
    SUMMARY[module] = dict(rows=rows, winner=best_name, test=test, baseline=base_m, primary="f1")


def _sub_grid(name):
    if name.startswith("XGBoost"):
        return {"model__max_depth": [3, 4, 5, 6], "model__n_estimators": [200, 300, 500],
                "model__learning_rate": [0.05, 0.1, 0.2]}
    if name.startswith("RandomForest"):
        return {"model__max_depth": [None, 8, 12, 16], "model__min_samples_leaf": [1, 3, 5]}
    if name.startswith("GradientBoosting"):
        return {"model__max_depth": [2, 3, 4], "model__n_estimators": [100, 200, 300]}
    if name.startswith("LogReg"):
        return {"model__C": [0.1, 0.5, 1, 5, 10]}
    return None


def _shap_summary(pipe, X, title, fname, max_rows=800):
    import shap
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    Xt = prep.transform(X.iloc[:max_rows])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    try:
        if hasattr(model, "feature_importances_"):
            vals = shap.TreeExplainer(model).shap_values(Xt)
        else:
            vals = shap.LinearExplainer(model, Xt).shap_values(Xt)
        if isinstance(vals, list):
            vals = vals[-1]
        if getattr(vals, "ndim", 2) == 3:
            vals = vals[:, :, -1]
        shap.summary_plot(vals, Xt, feature_names=names, show=False, max_display=15)
        plt.title(title)
        return savefig(fname)
    except Exception as e:  # SHAP is interpretation only; never block training
        print(f"  SHAP skipped: {e}")
        plt.close()
        return None


def _subscription_plots(best, X_tr, X_te, y_te, y_hat, proba):
    figs = []
    plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_matrix(y_te, y_hat), annot=True, fmt="d", cmap="Blues",
                xticklabels=["not recurring", "recurring"], yticklabels=["not recurring", "recurring"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Subscription detector (test)")
    figs.append(savefig("subscription_confusion_matrix.png"))
    fpr, tpr, _ = roc_curve(y_te, proba)
    p, r, _ = precision_recall_curve(y_te, proba)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="grey"); ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
    ax[1].plot(r, p); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    figs.append(savefig("subscription_roc_pr.png"))
    model = best.named_steps["model"]
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    if imp is not None:
        plt.figure(figsize=(6, 4))
        pd.Series(imp, index=F.SUB_FEATURES).sort_values().plot.barh(color="#4F46E5")
        plt.title("Feature importance"); figs.append(savefig("subscription_feature_importance.png"))
    figs.append(_shap_summary(best, X_te, "SHAP - subscription detector", "subscription_shap.png"))
    return [f for f in figs if f]


# ====================================================================================== 4.3 forecaster
def check_forecast_leakage(df):
    """Features for month t must not change when data after month t is removed."""
    users = df["user_id"].unique()[:5]
    d = df[df["user_id"].isin(users)]
    full = F.forecast_features(F.monthly_table(d))
    t = full["month"].sort_values().unique()[len(full["month"].unique()) // 2]
    cut = d[d["date"] <= t.to_timestamp(how="end")]
    part = F.forecast_features(F.monthly_table(cut))
    a = full[full["month"] == t].set_index(["user_id", "category"])[F.FC_NUM].sort_index()
    b = part[part["month"] == t].set_index(["user_id", "category"])[F.FC_NUM].sort_index()
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), "LEAKAGE: features use future data"
    print("[forecaster] leakage check passed: features for month t only use data up to t")


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def smape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = (np.abs(a) + np.abs(b))
    return float(np.mean(np.where(den == 0, 0, 2 * np.abs(a - b) / np.where(den == 0, 1, den))) * 100)


def reg_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    nz = y > 0  # sMAPE is 200% on every month with zero actual spend, so also report it on non-zero months
    return dict(rmse=rmse(y, p), mae=mean_absolute_error(y, p), smape=smape(y, p), smape_nonzero=smape(y[nz], p[nz]),
                r2=r2_score(y, p))


def train_forecaster(df):
    module = "forecaster"
    setup_mlflow(module)
    check_forecast_leakage(df)
    feats = F.forecast_features(F.monthly_table(df))
    data = feats[feats["target"].notna()].sort_values("target_month").reset_index(drop=True)
    months = sorted(data["target_month"].unique())
    test_m, val_m = months[-3:], months[-7:-3]
    tr = data[~data["target_month"].isin(test_m + val_m)]
    va = data[data["target_month"].isin(val_m)]
    te = data[data["target_month"].isin(test_m)]
    print(f"[{module}] chronological split: train {months[0]}..{tr['target_month'].max()} | val {val_m[0]}..{val_m[-1]}"
          f" | test {test_m[0]}..{test_m[-1]}")
    cols = F.FC_NUM + ["category"]
    prep = F.forecast_preprocessor
    candidates = {
        "Ridge": make_pipe(Ridge(alpha=1.0), prep=prep()),
        "RandomForest": make_pipe(RandomForestRegressor(200, max_depth=14, min_samples_leaf=5, n_jobs=N_JOBS,
                                                        random_state=SEED), prep=prep()),
        "GradientBoosting": make_pipe(GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=SEED), prep=prep()),
        "XGBoost": make_pipe(XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, random_state=SEED, n_jobs=N_JOBS), prep=prep()),
        "LightGBM": make_pipe(LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                            subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1,
                                            n_jobs=N_JOBS), prep=prep()),
    }
    rows = []
    for name, col in [("Baseline: last month", "lag1"), ("Baseline: 3-month mean", "roll3_mean")]:
        m = reg_metrics(va["target"], va[col])
        rows.append(dict(model=name, **{f"val_{k}": v for k, v in m.items()}))
        print(f"[{module}] {name:28s} val rmse={m['rmse']:.1f}")
    fitted = {}
    for name, pipe in candidates.items():
        t = time.time()
        pipe.fit(tr[cols], tr["target"])
        secs = time.time() - t
        m = reg_metrics(va["target"], F.predict_spend(pipe, va))
        fitted[name] = pipe
        rows.append(dict(model=name, **{f"val_{k}": v for k, v in m.items()}, train_seconds=round(secs, 1)))
        log_candidate(name, {}, {f"val_{k}": v for k, v in m.items()}, secs)
        print(f"[{module}] {name:28s} val rmse={m['rmse']:.1f} mae={m['mae']:.1f} r2={m['r2']:.3f} ({secs:.0f}s)")
    cand = pd.DataFrame([r for r in rows if not r["model"].startswith("Baseline")]).sort_values("val_rmse")
    best_name = cand.iloc[0]["model"]
    # within 1% RMSE counts as noise -> take the faster one
    close = cand[cand["val_rmse"] <= cand.iloc[0]["val_rmse"] * 1.01].sort_values("train_seconds")
    if close.iloc[0]["model"] != best_name:
        print(f"  {close.iloc[0]['model']} within 1% of {best_name} and faster -> choosing it")
        best_name = close.iloc[0]["model"]
    best = fitted[best_name]

    # light tuning with month-aligned time-series folds inside the training block
    grid = _fc_grid(best_name)
    if grid:
        tr_months = sorted(tr["target_month"].unique())
        folds = []
        for k in (3, 2, 1):
            cut = tr_months[-k * 2]
            tri = np.where(tr["target_month"] < cut)[0]
            vai = np.where((tr["target_month"] >= cut) & (tr["target_month"] < tr_months[min(len(tr_months) - 1, tr_months.index(cut) + 2)]))[0]
            folds.append((tri, vai))
        search = RandomizedSearchCV(best, grid, n_iter=min(8, _grid_size(grid)), cv=folds,
                                    scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1)
        search.fit(tr[cols].reset_index(drop=True), tr["target"].reset_index(drop=True))
        tuned = search.best_estimator_.fit(tr[cols], tr["target"])
        tv = rmse(va["target"], F.predict_spend(tuned, va))
        print(f"[{module}] tuned {best_name}: {search.best_params_} val rmse {tv:.1f}")
        rows.append(dict(model=f"{best_name} (tuned)", val_rmse=tv, params=str(search.best_params_)))
        if tv < rmse(va["target"], F.predict_spend(best, va)):
            best = tuned
    residuals = va["target"] - F.predict_spend(best, va)
    q10, q90 = float(np.quantile(residuals, 0.10)), float(np.quantile(residuals, 0.90))

    # refit on train+val, evaluate once on test
    trva = pd.concat([tr, va])
    best.fit(trva[cols], trva["target"])
    pred = F.predict_spend(best, te)
    test = reg_metrics(te["target"], pred)
    base_m = reg_metrics(te["target"], te["roll3_mean"])
    base_last = reg_metrics(te["target"], te["lag1"])
    print(f"[{module}] TEST {best_name}: {test} | baseline 3-month mean {base_m} | last month {base_last}")
    per_cat = te.assign(pred=pred).groupby("category").apply(
        lambda g: pd.Series(dict(rmse=rmse(g["target"], g["pred"]), mae=mean_absolute_error(g["target"], g["pred"]),
                                 baseline_rmse=rmse(g["target"], g["roll3_mean"]))), include_groups=False)
    per_cat.to_csv(RESULTS / "forecaster_error_by_category.csv")
    figs = _forecast_plots(best, te, pred, per_cat, test, base_m, base_last, residuals)
    pd.DataFrame(rows).to_csv(RESULTS / f"model_comparison_{module}.csv", index=False)
    entry = dict(algorithm=best_name, primary_metric="rmse", test_metrics=test, baseline_metrics=base_m,
                 baseline_last_month=base_last, residual_q10=q10, residual_q90=q90)
    best_base = min(base_m["rmse"], base_last["rmse"])
    path = save_model(module, best, entry, test["rmse"] < best_base)
    log_winner(best_name, {"residual_q10": q10, "residual_q90": q90}, test, path, figs)
    SUMMARY[module] = dict(rows=rows, winner=best_name, test=test, baseline=base_m, primary="rmse")


def _fc_grid(name):
    if name in ("XGBoost", "LightGBM"):
        return {"model__n_estimators": [200, 400, 700], "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth" if name == "XGBoost" else "model__num_leaves": [4, 6, 8] if name == "XGBoost" else [15, 31, 63]}
    if name == "RandomForest":
        return {"model__max_depth": [10, 14, 20], "model__min_samples_leaf": [2, 5, 10]}
    if name == "GradientBoosting":
        return {"model__max_depth": [3, 4, 5], "model__n_estimators": [150, 300]}
    if name == "Ridge":
        return {"model__alpha": [0.1, 1, 10, 100]}
    return None


def _forecast_plots(best, te, pred, per_cat, test, base_m, base_last, residuals):
    figs = []
    plt.figure(figsize=(5, 5))
    plt.scatter(te["target"], pred, s=4, alpha=0.3, color="#4F46E5")
    lim = np.percentile(te["target"], 99.5)
    plt.plot([0, lim], [0, lim], "--", c="grey"); plt.xlim(0, lim); plt.ylim(0, lim)
    plt.xlabel("Actual next-month spend (₹)"); plt.ylabel("Predicted (₹)"); plt.title("Predicted vs actual (test)")
    figs.append(savefig("forecaster_pred_vs_actual.png"))
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals.clip(*np.percentile(residuals, [1, 99])), bins=60, color="#4F46E5")
    plt.xlabel("Residual (₹, validation)"); plt.title("Residuals")
    figs.append(savefig("forecaster_residuals.png"))
    plt.figure(figsize=(8, 5))
    per_cat[["rmse", "baseline_rmse"]].sort_values("rmse").plot.barh(ax=plt.gca(), color=["#4F46E5", "#CBD5E1"])
    plt.xlabel("RMSE (₹)"); plt.title("Error by category (test)")
    figs.append(savefig("forecaster_error_by_category.png"))
    plt.figure(figsize=(6, 4))
    pd.Series({"Last month": base_last["rmse"], "3-month mean": base_m["rmse"], "Model": test["rmse"]}).plot.bar(
        color=["#CBD5E1", "#CBD5E1", "#4F46E5"], rot=0)
    plt.ylabel("RMSE (₹, test)"); plt.title("Baselines vs model")
    figs.append(savefig("forecaster_baseline_vs_model.png"))
    figs.append(_shap_summary(best, te[F.FC_NUM + ["category"]], "SHAP - forecaster", "forecaster_shap.png"))
    return [f for f in figs if f]


# ====================================================================================== 4.4 persona
def name_clusters(profile):
    """Assign human-readable names from each cluster's average profile (original units)."""
    names, used = {}, set()
    rules = [
        ("Debt-Heavy", "share_EMI & Loans", "A large share of money goes to EMIs and loans.",
         "Prioritise the highest-interest loan; avoid new EMIs until it falls below 20% of income."),
        ("Saver", "savings_rate", "Saves and invests a big share of income.",
         "Great discipline - check your emergency fund covers 6 months, then keep the SIPs going."),
        ("Lifestyle Spender", "lifestyle", "Spends heavily on dining, shopping, entertainment and travel.",
         "Set a monthly cap on dining and shopping; even a 15% cut adds up fast."),
        ("Student / Early Career", "small_ticket", "Lower income, many small transactions, education costs.",
         "Track small UPI spends - they add up. Start a small SIP, even ₹500/month."),
        ("Family Budgeter", "share_Groceries", "Essentials-heavy: groceries, education and health dominate.",
         "Plan groceries weekly and compare quick-commerce prices with supermarkets."),
    ]
    p = profile.copy()
    p["lifestyle"] = p[[c for c in ["share_Food & Dining", "share_Shopping", "share_Entertainment", "share_Travel"]]].sum(1)
    p["small_ticket"] = -p["avg_txn_size"]
    remaining = list(p.index)
    for name, col, desc, tip in rules:
        if not remaining:
            break
        k = p.loc[remaining, col].idxmax()
        names[str(k)] = dict(name=name, description=desc, tip=tip)
        remaining.remove(k)
    for k in remaining:
        names[str(k)] = dict(name=f"Balanced Spender {k}", description="No single category dominates.",
                             tip="Review your top 3 categories each month.")
    return names


def train_persona(df):
    module = "persona"
    setup_mlflow(module)
    feats = F.persona_features(df)
    feats = feats[feats["n_months"] >= 3].reset_index(drop=True)
    users = pd.read_csv(RAW / "users.csv")[["user_id", "archetype"]]
    feats = feats.merge(users, on="user_id", how="left")  # archetype kept ONLY for post-hoc ARI
    tr_u, te_u = train_test_split(feats["user_id"], test_size=0.2, random_state=SEED)
    tr, te = feats[feats["user_id"].isin(tr_u)], feats[feats["user_id"].isin(te_u)]
    prep = F.persona_preprocessor().fit(tr[F.PERSONA_FEATURES])
    Xtr = prep.transform(tr[F.PERSONA_FEATURES])
    rows = []
    algos = {
        "KMeans": lambda k: KMeans(k, n_init=10, random_state=SEED),
        "GaussianMixture": lambda k: GaussianMixture(k, covariance_type="diag", random_state=SEED),
        "Agglomerative(Ward)": lambda k: AgglomerativeClustering(k, linkage="ward"),
    }
    labels_store = {}
    for name, make in algos.items():
        for k in range(2, 9):
            t = time.time()
            m = make(k)
            lab = m.fit_predict(Xtr)
            secs = time.time() - t
            sizes = np.bincount(lab, minlength=k) / len(lab)
            r = dict(model=name, k=k, silhouette=silhouette_score(Xtr, lab), davies_bouldin=davies_bouldin_score(Xtr, lab),
                     min_cluster_share=sizes.min(), inertia=getattr(m, "inertia_", np.nan),
                     ari_vs_archetype=adjusted_rand_score(tr["archetype"], lab), train_seconds=round(secs, 2))
            rows.append(r)
            labels_store[(name, k)] = lab
            log_candidate(f"{name} k={k}", {"k": k}, {kk: vv for kk, vv in r.items() if kk in
                                                       ("silhouette", "davies_bouldin", "min_cluster_share")}, secs)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"model_comparison_{module}.csv", index=False)
    _k_plots(res)
    # Selection: highest silhouette among configs with no junk cluster (<5% of users) and k >= 3
    # (k=2 only splits "savers vs everyone" which is not a useful persona set); Davies-Bouldin breaks ties (<0.01).
    ok = res[(res["min_cluster_share"] >= 0.05) & (res["k"] >= 3)].sort_values("silhouette", ascending=False)
    top = ok.iloc[0]
    ties = ok[ok["silhouette"] >= top["silhouette"] - 0.01].sort_values("davies_bouldin")
    choice = ties.iloc[0]
    best_name, k = choice["model"], int(choice["k"])
    print(f"[{module}] best overall silhouette: {res.sort_values('silhouette').iloc[-1][['model', 'k', 'silhouette']].to_dict()}")
    print(f"[{module}] chosen {best_name} k={k}: silhouette={choice['silhouette']:.3f} DB={choice['davies_bouldin']:.3f} "
          f"ARI vs hidden archetype={choice['ari_vs_archetype']:.3f}")
    lab = labels_store[(best_name, k)]
    # one serving pipeline: preprocessing + nearest centroid of the chosen clustering (works for every algorithm)
    pipe = Pipeline([("prep", F.persona_preprocessor()), ("model", NearestCentroid())])
    pipe.fit(tr[F.PERSONA_FEATURES], lab)
    te_lab = pipe.predict(te[F.PERSONA_FEATURES])
    Xte = pipe.named_steps["prep"].transform(te[F.PERSONA_FEATURES])
    test = dict(silhouette=silhouette_score(Xte, te_lab) if len(set(te_lab)) > 1 else np.nan,
                davies_bouldin=davies_bouldin_score(Xte, te_lab) if len(set(te_lab)) > 1 else np.nan,
                ari_vs_archetype=adjusted_rand_score(te["archetype"], te_lab))
    print(f"[{module}] TEST (held-out users): {test}")
    profile = tr.assign(cluster=lab).groupby("cluster")[F.PERSONA_FEATURES].mean()
    names = name_clusters(profile)
    profile.to_csv(RESULTS / "persona_cluster_profiles.csv")
    pd.crosstab(tr.assign(cluster=lab)["cluster"].map(lambda c: names[str(c)]["name"]), tr["archetype"]).to_csv(
        RESULTS / "persona_vs_archetype.csv")
    figs = _persona_plots(Xtr, lab, names, profile)
    # baseline: random assignment silhouette is ~0; we require a positive silhouette
    entry = dict(algorithm=f"{best_name} (k={k})", primary_metric="silhouette", test_metrics=test,
                 baseline_metrics=dict(silhouette=0.0, note="random labels ~ 0"), k=k, persona_names=names,
                 persona_profiles=profile.round(4).to_dict(orient="index"), min_months=3)
    path = save_model(module, pipe, entry, choice["silhouette"] > 0)
    log_winner(f"{best_name} k={k}", {"k": k}, test, path, figs)
    SUMMARY[module] = dict(rows=res.sort_values("silhouette", ascending=False).head(8).to_dict("records"),
                           winner=f"{best_name} k={k}", test=test, baseline=entry["baseline_metrics"], primary="silhouette")


def _k_plots(res):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    km = res[res["model"] == "KMeans"]
    ax[0].plot(km["k"], km["inertia"], marker="o"); ax[0].set(title="Elbow (KMeans inertia)", xlabel="k")
    for name, g in res.groupby("model"):
        ax[1].plot(g["k"], g["silhouette"], marker="o", label=name)
        ax[2].plot(g["k"], g["davies_bouldin"], marker="o", label=name)
    ax[1].set(title="Silhouette (higher better)", xlabel="k"); ax[2].set(title="Davies-Bouldin (lower better)", xlabel="k")
    ax[1].legend()
    savefig("persona_k_selection.png")


def _persona_plots(X, lab, names, profile):
    figs = []
    pc = PCA(2, random_state=SEED).fit_transform(X)
    plt.figure(figsize=(7, 5))
    for c in np.unique(lab):
        plt.scatter(pc[lab == c, 0], pc[lab == c, 1], s=14, label=names[str(c)]["name"])
    plt.legend(fontsize=8); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Personas (PCA 2D, train users)")
    figs.append(savefig("persona_pca.png"))
    plt.figure(figsize=(6, 4))
    pd.Series(lab).map(lambda c: names[str(c)]["name"]).value_counts().plot.barh(color="#4F46E5")
    plt.xlabel("users"); plt.title("Cluster sizes")
    figs.append(savefig("persona_cluster_sizes.png"))
    z = (profile - profile.mean()) / profile.std().replace(0, 1)
    z.index = [names[str(c)]["name"] for c in z.index]
    plt.figure(figsize=(14, 4))
    sns.heatmap(z, cmap="RdBu_r", center=0, xticklabels=[c.replace("share_", "") for c in z.columns])
    plt.title("Cluster centroids (z-score across clusters)")
    figs.append(savefig("persona_centroid_heatmap.png"))
    return figs


# ====================================================================================== summary
def write_summary():
    lines = ["# Results summary", "",
             "> All numbers come from `python -m src.train` on **synthetic data** (see README). They are not real-world performance.", ""]
    info = load_info()
    for module, s in SUMMARY.items():
        lines += [f"## {module}", "", f"Primary metric: **{s['primary']}**", "",
                  pd.DataFrame(s["rows"]).round(4).to_markdown(index=False), "",
                  f"**Winner:** {s['winner']}  ", f"**Test metrics:** " + ", ".join(
                      f"{k}={v:.4f}" for k, v in s["test"].items() if isinstance(v, (int, float))), "  ",
                  f"**Baseline (test):** " + ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                                                        for k, v in s["baseline"].items()), ""]
        if module in info:
            lines += [f"Registered version: v{info[module]['version']} ({info[module]['file']})", ""]
    path = RESULTS / "summary.md"
    old = path.read_text() if path.exists() else ""
    # keep sections of modules not retrained in this run
    path.write_text("\n".join(lines) + ("\n\n<!-- previous -->\n" + old.split("<!-- previous -->")[0] if old and len(SUMMARY) < 4 else ""))
    print(f"[summary] wrote {path}")


def main(modules=None):
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    np.random.seed(SEED)
    df = load_clean()
    steps = {"categorizer": train_categorizer, "subscription": train_subscription,
             "forecaster": train_forecaster, "persona": train_persona}
    for name in modules or steps:
        t = time.time()
        print(f"\n===== {name} =====")
        steps[name](df)
        print(f"===== {name} done in {time.time() - t:.0f}s")
    write_summary()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
