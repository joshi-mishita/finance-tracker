"""Feature building for all four models. Used by BOTH training (train.py) and serving (services.py)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

from src.data_prep import EXPENSE_CATEGORIES, NON_EXPENSE

# ---------------------------------------------------------------- 4.1 categorizer
CAT_COLS = ["clean_desc", "payment_mode", "day_of_week", "log_amount"]


def categorizer_frame(df):
    """Input columns for the categorizer (df must already be cleaned)."""
    X = df[CAT_COLS].copy()
    X["day_of_week"] = X["day_of_week"].astype(str)
    return X


def categorizer_preprocessor(max_word=8000, max_char=12000):
    return ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True), "clean_desc"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=max_char,
                                 sublinear_tf=True), "clean_desc"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["payment_mode", "day_of_week"]),
        ("num", MinMaxScaler(clip=True), ["log_amount"]),  # non-negative so Naive Bayes works too
    ])


# ---------------------------------------------------------------- 4.2 subscriptions
SUB_FEATURES = ["n_payments", "gap_mean", "gap_median", "gap_std", "gap_cv", "amount_mean", "amount_cv",
                "same_amount_share", "dom_std", "months_active", "longest_missing"]


def subscription_features(df):
    """One row per (user_id, merchant) with >=2 debit payments."""
    d = df[df["type"] == "debit"].sort_values("date")
    rows = []
    for (uid, merch), g in d.groupby(["user_id", "merchant"], sort=False):
        if len(g) < 2:
            continue
        gaps = g["date"].diff().dt.days.dropna().to_numpy()
        amt = g["amount"].to_numpy()
        med_amt = np.median(amt)
        gm = gaps.mean()
        rows.append(dict(
            user_id=uid, merchant=merch, n_payments=len(g), gap_mean=gm, gap_median=np.median(gaps),
            gap_std=gaps.std(), gap_cv=gaps.std() / gm if gm else 0, amount_mean=amt.mean(),
            amount_cv=amt.std() / amt.mean(), same_amount_share=np.mean(np.abs(amt - med_amt) <= 0.02 * med_amt),
            dom_std=g["date"].dt.day.std(ddof=0), months_active=g["date"].dt.to_period("M").nunique(),
            longest_missing=gaps.max() - np.median(gaps), last_date=g["date"].max(),
            category=g["category"].mode().iat[0] if "category" in g else None,
        ))
    return pd.DataFrame(rows)


def subscription_preprocessor():
    return ColumnTransformer([("num", StandardScaler(), SUB_FEATURES)])


# ---------------------------------------------------------------- 4.3 forecasting
FC_NUM = ["lag1", "lag2", "lag3", "roll3_mean", "roll3_std", "lag12", "has_lag12", "txn_count", "avg_ticket",
          "income", "spend_to_income", "month_sin", "month_cos"]


def monthly_table(df):
    """Full grid of (user, category, month) with spend, txn count and the user's income that month."""
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    exp = d[(d["type"] == "debit") & d["category"].isin(EXPENSE_CATEGORIES)]
    agg = exp.groupby(["user_id", "category", "month"]).agg(spend=("amount", "sum"), txn_count=("amount", "size"))
    income = d[d["type"] == "credit"].groupby(["user_id", "month"])["amount"].sum().rename("income")
    grids = []
    for uid, g in d.groupby("user_id"):
        months = pd.period_range(g["month"].min(), g["month"].max(), freq="M")
        grids.append(pd.MultiIndex.from_product([[uid], EXPENSE_CATEGORIES, months],
                                                names=["user_id", "category", "month"]).to_frame(index=False))
    grid = pd.concat(grids, ignore_index=True)
    m = grid.merge(agg.reset_index(), how="left", on=["user_id", "category", "month"])
    m[["spend", "txn_count"]] = m[["spend", "txn_count"]].fillna(0)
    m = m.merge(income.reset_index(), how="left", on=["user_id", "month"])
    m["income"] = m["income"].fillna(0)
    return m.sort_values(["user_id", "category", "month"]).reset_index(drop=True)


def forecast_features(monthly):
    """Row at month t uses only data up to month t; target = spend in month t+1 (NaN for the last month)."""
    m = monthly.sort_values(["user_id", "category", "month"]).copy()
    g = m.groupby(["user_id", "category"], sort=False)["spend"]
    m["lag1"] = m["spend"]
    m["lag2"] = g.shift(1)
    m["lag3"] = g.shift(2)
    m["roll3_mean"] = m[["lag1", "lag2", "lag3"]].mean(axis=1)
    m["roll3_std"] = m[["lag1", "lag2", "lag3"]].std(axis=1)
    m["lag12"] = g.shift(11)  # same month last year, relative to the target month t+1
    m["has_lag12"] = m["lag12"].notna().astype(int)
    m["lag12"] = m["lag12"].fillna(m["roll3_mean"])
    m["avg_ticket"] = np.where(m["txn_count"] > 0, m["spend"] / m["txn_count"].clip(lower=1), 0)
    m["spend_to_income"] = m["lag1"] / m["income"].replace(0, np.nan)
    m["spend_to_income"] = m["spend_to_income"].fillna(0)
    target_month = (m["month"] + 1)
    mn = target_month.dt.month
    m["month_sin"], m["month_cos"] = np.sin(2 * np.pi * mn / 12), np.cos(2 * np.pi * mn / 12)
    m["target_month"] = target_month
    m["target"] = g.shift(-1)
    return m[m["lag3"].notna()].reset_index(drop=True)


def predict_spend(model, X):
    """Forecast spend; predictions are floored at 0 (spend can't be negative). Used in training and the app."""
    return np.clip(model.predict(X[FC_NUM + ["category"]]), 0, None)


def forecast_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), FC_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
    ])


# ---------------------------------------------------------------- 4.4 persona
PERSONA_SHARE_COLS = [f"share_{c}" for c in EXPENSE_CATEGORIES]
PERSONA_NUM = ["savings_rate", "weekend_ratio", "monthly_cv", "avg_txn_size", "txns_per_month"]
PERSONA_FEATURES = PERSONA_NUM + PERSONA_SHARE_COLS
PERSONA_LOG = ["avg_txn_size", "txns_per_month", "weekend_ratio"]


def persona_features(df):
    """One row per user, aggregated over all their months. Includes n_months for the cold-start rule."""
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    out = []
    for uid, g in d.groupby("user_id"):
        debit = g[g["type"] == "debit"]
        n_months = g["month"].nunique()
        income = g.loc[g["type"] == "credit", "amount"].sum()
        expense = debit.loc[~debit["category"].isin(NON_EXPENSE), "amount"].sum()
        row = dict(user_id=uid, n_months=n_months,
                   savings_rate=(income - expense) / income if income else 0)
        tot = debit["amount"].sum() or 1
        shares = debit.groupby("category")["amount"].sum() / tot
        for c in EXPENSE_CATEGORIES:
            row[f"share_{c}"] = float(shares.get(c, 0))
        wk = debit["date"].dt.dayofweek >= 5
        n_days = max((g["date"].max() - g["date"].min()).days + 1, 1)
        we_days = max(n_days * 2 / 7, 1)
        wd_days = max(n_days * 5 / 7, 1)
        row["weekend_ratio"] = (debit.loc[wk, "amount"].sum() / we_days) / max(debit.loc[~wk, "amount"].sum() / wd_days, 1)
        monthly = debit.groupby("month")["amount"].sum()
        row["monthly_cv"] = monthly.std() / monthly.mean() if len(monthly) > 1 else 0
        row["avg_txn_size"] = debit["amount"].mean() if len(debit) else 0
        row["txns_per_month"] = len(debit) / max(n_months, 1)
        out.append(row)
    return pd.DataFrame(out)


def persona_preprocessor():
    log_cols = PERSONA_LOG
    other = [c for c in PERSONA_FEATURES if c not in log_cols]
    return ColumnTransformer([
        ("log", _log_then_scale(), log_cols),
        ("num", StandardScaler(), other),
    ])


def _log_then_scale():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
