"""App logic used by app.py: database, predictions, forecast, budget, savings, autopay, patterns.

All feature code comes from src/features.py, so serving uses exactly the same transformations as training.
"""
import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import features as F
from src.data_prep import (DISCRETIONARY, ESSENTIALS, EXPENSE_CATEGORIES, NON_EXPENSE, RAW, ROOT,
                           clean_transactions, import_bank_csv)

DB_PATH = Path(os.environ.get("FINANCE_DB", ROOT / "finance.db"))
MODELS = ROOT / "models"
DEMO_USER_ID = "U0001"  # a generated user (Lifestyle Spender archetype) used for the demo button


# ====================================================================================== formatting
def inr(x, decimals=0):
    """₹ with Indian digit grouping: 12,34,567."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    neg = x < 0
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = ",".join([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1])
        whole = f"{head},{tail}"
    return f"{'-' if neg else ''}₹{whole}{'.' + frac if frac else ''}"


# ====================================================================================== database
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, monthly_income REAL, created_at TEXT, budget_json TEXT);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, date TEXT, description TEXT, amount REAL, type TEXT,
    category TEXT, predicted_category TEXT, confidence REAL, needs_review INTEGER, user_corrected INTEGER DEFAULT 0,
    source TEXT);
CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, merchant TEXT, category TEXT, avg_amount REAL,
    frequency_days REAL, next_due_date TEXT, yearly_cost REAL, confidence REAL, autopay INTEGER DEFAULT 0,
    status TEXT DEFAULT 'detected', UNIQUE(username, merchant));
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, name TEXT, target_amount REAL, saved_amount REAL,
    target_date TEXT);
"""


def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with conn() as c:
        c.executescript(SCHEMA)


def get_user(username):
    with conn() as c:
        r = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return dict(r) if r else None


def create_user(username, income):
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?)",
                  (username, float(income), datetime.now().isoformat(timespec="seconds"), None))


def update_income(username, income):
    with conn() as c:
        c.execute("UPDATE users SET monthly_income=? WHERE username=?", (float(income), username))


def save_budget(username, budget: dict):
    with conn() as c:
        c.execute("UPDATE users SET budget_json=? WHERE username=?", (json.dumps(budget), username))


def get_budget(username):
    u = get_user(username)
    return json.loads(u["budget_json"]) if u and u.get("budget_json") else {}


def load_transactions(username):
    with conn() as c:
        df = pd.read_sql("SELECT * FROM transactions WHERE username=? ORDER BY date DESC", c, params=(username,))
    df["date"] = pd.to_datetime(df["date"])
    return df


def insert_transactions(username, df, source):
    cols = ["date", "description", "amount", "type", "category", "predicted_category", "confidence", "needs_review"]
    rows = [(username, pd.Timestamp(r.date).strftime("%Y-%m-%d"), r.description, float(r.amount), r.type, r.category,
             r.predicted_category, float(r.confidence), int(r.needs_review), source)
            for r in df[cols].itertuples(index=False)]
    with conn() as c:
        c.executemany("INSERT INTO transactions (username,date,description,amount,type,category,predicted_category,"
                      "confidence,needs_review,source) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def update_category(txn_id, new_category):
    """User correction: category changes, the model's original guess stays in predicted_category."""
    with conn() as c:
        c.execute("UPDATE transactions SET category=?, user_corrected=1, needs_review=0 WHERE id=?",
                  (new_category, int(txn_id)))


def delete_user_data(username):
    with conn() as c:
        for t in ("transactions", "subscriptions", "goals"):
            c.execute(f"DELETE FROM {t} WHERE username=?", (username,))


# ====================================================================================== models
def load_models():
    """Read models/model_info.json and load the active version of every model."""
    info_path = MODELS / "model_info.json"
    if not info_path.exists():
        return {}, {}
    info = json.loads(info_path.read_text())
    models = {}
    for name, entry in info.items():
        p = MODELS / entry["file"]
        if p.exists():
            models[name] = joblib.load(p)
    return models, info


def _to_clean(df, username="me"):
    d = df.copy()
    d["user_id"] = username
    d["date"] = pd.to_datetime(d["date"])
    if "type" not in d:
        d["type"] = "debit"
    return clean_transactions(d, verbose=False)


def categorize(df, models, info):
    """Adds predicted_category, confidence, needs_review (+ category) to a raw transactions frame."""
    clean = _to_clean(df.reset_index(drop=True))
    model, thr = models["categorizer"], info["categorizer"]["confidence_threshold"]
    proba = model.predict_proba(F.categorizer_frame(clean))
    classes = model.classes_
    out = clean.copy()
    out["predicted_category"] = classes[proba.argmax(1)]
    out["confidence"] = proba.max(1)
    credit = out["type"] == "credit"
    # the categorizer is trained on debits; credits are labelled with a simple rule
    out.loc[credit, "predicted_category"] = np.where(out.loc[credit, "clean_desc"].str.contains("sal"), "Salary",
                                                     "Other Income")
    out.loc[credit, "confidence"] = 1.0
    out["needs_review"] = (out["confidence"] < thr).astype(int)
    out["category"] = out["predicted_category"]
    return out


def suggest_category(description, amount, when, ttype, models, info):
    row = pd.DataFrame([dict(date=when, description=description, amount=amount, type=ttype)])
    r = categorize(row, models, info).iloc[0]
    return r["predicted_category"], float(r["confidence"]), bool(r["needs_review"])


def import_csv(file, username, models, info):
    """Accepts either date/description/amount[/type] or a bank export with Narration/Withdrawal/Deposit columns."""
    raw = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in raw.columns}
    if "narration" in cols:
        tmp = Path("/tmp/_upload.csv")
        raw.to_csv(tmp, index=False)
        df = import_bank_csv(tmp)
    else:
        need = {"date", "description", "amount"}
        if not need <= set(cols):
            raise ValueError(f"CSV needs columns {sorted(need)} (optional: type). Found: {list(raw.columns)}")
        df = pd.DataFrame({"date": pd.to_datetime(raw[cols["date"]], errors="coerce", dayfirst=True, format="mixed"),
                           "description": raw[cols["description"]].astype(str),
                           "amount": pd.to_numeric(raw[cols["amount"]], errors="coerce")})
        if "type" in cols:
            df["type"] = raw[cols["type"]].str.lower().str.strip()
        else:
            # signed amounts: negative = debit, positive = credit; all-positive files are treated as spending
            df["type"] = np.where(df["amount"] < 0, "debit", "credit") if (df["amount"] < 0).any() else "debit"
        df["amount"] = df["amount"].abs()
    n_in = len(df)
    df = df.dropna(subset=["date", "amount"])
    out = categorize(df, models, info)
    n = insert_transactions(username, out, "csv")
    return dict(rows_in=n_in, imported=n, needs_review=int(out["needs_review"].sum()),
                by_category=out["category"].value_counts().to_dict())


def load_demo(username, models, info):
    """Copy one generated user's history into the DB, shifted so the latest month is last month."""
    path = RAW / "transactions.csv"
    if not path.exists():
        raise FileNotFoundError("Generated data not found. Run `python -m src.data_prep` first.")
    tx = pd.read_csv(path, parse_dates=["date"])
    tx = tx[tx["user_id"] == DEMO_USER_ID].copy()
    last = tx["date"].max().to_period("M")
    target = pd.Timestamp(date.today()).to_period("M") - 1
    shift = (target - last).n
    tx["date"] = tx["date"].apply(lambda d: d + pd.DateOffset(months=shift))
    delete_user_data(username)
    out = categorize(tx[["date", "description", "amount", "type"]], models, info)
    insert_transactions(username, out, "csv")
    users = pd.read_csv(RAW / "users.csv")
    income = float(users.loc[users["user_id"] == DEMO_USER_ID, "monthly_income"].iat[0])
    create_user(username, income)
    update_income(username, income)
    return len(out)


# ====================================================================================== subscriptions
def detect_subscriptions(username, models, info):
    tx = load_transactions(username)
    if tx.empty or "subscription" not in models:
        return pd.DataFrame()
    clean = _to_clean(tx[["date", "description", "amount", "type", "category"]], username)
    feats = F.subscription_features(clean)
    if feats.empty:
        return feats
    proba = models["subscription"].predict_proba(feats[F.SUB_FEATURES])[:, 1]
    feats["confidence"] = proba
    subs = feats[proba >= info["subscription"].get("decision_threshold", 0.5)].copy()
    # stopped subscriptions: no payment for more than 2 cycles
    today = pd.Timestamp(date.today())
    subs = subs[(today - subs["last_date"]).dt.days <= subs["gap_median"] * 2 + 7]
    subs["next_due_date"] = subs["last_date"] + pd.to_timedelta(subs["gap_median"].round(), unit="D")
    subs["yearly_cost"] = subs["amount_mean"] * 365 / subs["gap_median"].clip(lower=1)
    with conn() as c:
        for r in subs.itertuples():
            c.execute("""INSERT INTO subscriptions (username, merchant, category, avg_amount, frequency_days,
                         next_due_date, yearly_cost, confidence) VALUES (?,?,?,?,?,?,?,?)
                         ON CONFLICT(username, merchant) DO UPDATE SET avg_amount=excluded.avg_amount,
                         frequency_days=excluded.frequency_days, yearly_cost=excluded.yearly_cost,
                         confidence=excluded.confidence,
                         next_due_date=CASE WHEN subscriptions.autopay=1 AND subscriptions.next_due_date > excluded.next_due_date
                                       THEN subscriptions.next_due_date ELSE excluded.next_due_date END,
                         category=excluded.category""",
                      (username, r.merchant, r.category, float(r.amount_mean), float(r.gap_median),
                       r.next_due_date.strftime("%Y-%m-%d"), float(r.yearly_cost), float(r.confidence)))
    return get_subscriptions(username)


def get_subscriptions(username, include_dismissed=False):
    with conn() as c:
        df = pd.read_sql("SELECT * FROM subscriptions WHERE username=?" +
                         ("" if include_dismissed else " AND status!='dismissed'") + " ORDER BY next_due_date",
                         c, params=(username,))
    return df


def set_subscription(sub_id, **fields):
    keys = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE subscriptions SET {keys} WHERE id=?", (*fields.values(), int(sub_id)))


def run_autopay(username, today=None):
    """SIMULATED autopay: for each autopay subscription due on/before today, record a payment and roll the date."""
    today = pd.Timestamp(today or date.today())
    paid = []
    subs = get_subscriptions(username)
    for s in subs[subs["autopay"] == 1].itertuples():
        due = pd.Timestamp(s.next_due_date)
        while due <= today:
            row = pd.DataFrame([dict(date=due, description=f"AUTOPAY (SIMULATED) {s.merchant}", amount=s.avg_amount,
                                     type="debit", category=s.category, predicted_category=s.category,
                                     confidence=1.0, needs_review=0)])
            insert_transactions(username, row, "autopay_simulated")
            paid.append((s.merchant, due.date(), s.avg_amount))
            due = due + pd.Timedelta(days=max(int(round(s.frequency_days)), 1))
        set_subscription(s.id, next_due_date=due.strftime("%Y-%m-%d"))
    return paid


# ====================================================================================== forecast & budget
def _complete_months(tx):
    """Drop the current (incomplete) calendar month."""
    cur = pd.Timestamp(date.today()).to_period("M")
    return tx[tx["date"].dt.to_period("M") < cur]


def forecast_next_month(username, models, info):
    tx = _complete_months(load_transactions(username))
    if tx.empty or tx["date"].dt.to_period("M").nunique() < 3 or "forecaster" not in models:
        return None
    d = tx.assign(user_id=username)
    feats = F.forecast_features(F.monthly_table(d))
    last = feats[feats["month"] == feats["month"].max()].copy()
    pred = F.predict_spend(models["forecaster"], last)
    q10, q90 = info["forecaster"]["residual_q10"], info["forecaster"]["residual_q90"]
    out = pd.DataFrame({"category": last["category"].values, "forecast": pred,
                        "low": np.clip(pred + q10, 0, None), "high": np.clip(pred + q90, 0, None),
                        "last_month": last["lag1"].values, "avg_3m": last["roll3_mean"].values})
    out.attrs["target_month"] = str(last["target_month"].iat[0])
    return out.sort_values("forecast", ascending=False).reset_index(drop=True)


def expected_income(username):
    tx = _complete_months(load_transactions(username))
    inc = tx[tx["type"] == "credit"]
    if inc.empty:
        u = get_user(username)
        return float(u["monthly_income"]) if u else 0.0
    monthly = inc.groupby(inc["date"].dt.to_period("M"))["amount"].sum()
    return float(monthly.tail(3).mean())


def savings_forecast(username, fc):
    """Expected income (recent average) - predicted spending (investments count as savings, not spend)."""
    spend = fc.loc[~fc["category"].isin(NON_EXPENSE), "forecast"].sum()
    inc = expected_income(username)
    return dict(income=inc, spend=float(spend), investments=float(fc.loc[fc["category"].isin(NON_EXPENSE), "forecast"].sum()),
                savings=inc - spend)


def goals_monthly_need(username):
    g = get_goals(username)
    if g.empty:
        return 0.0
    today = pd.Timestamp(date.today())
    months = ((pd.to_datetime(g["target_date"]) - today).dt.days / 30.4).clip(lower=1)
    return float(((g["target_amount"] - g["saved_amount"]).clip(lower=0) / months).sum())


def recommend_budget(username, fc):
    sv = savings_forecast(username, fc)
    need = goals_monthly_need(username)
    shortfall = max(need - (sv["savings"]), 0)
    disc_total = fc.loc[fc["category"].isin(DISCRETIONARY), "forecast"].sum()
    cut = min(0.30, shortfall / disc_total) if disc_total > 0 and shortfall > 0 else 0.0
    rows = []
    for r in fc.itertuples():
        if r.category in ESSENTIALS:
            amt, why = r.forecast * 1.05, "Essential: forecast + 5% buffer"
        elif r.category in DISCRETIONARY and cut > 0:
            amt, why = r.forecast * (1 - cut), f"Discretionary: cut {cut:.0%} to fund your goals"
        elif r.category in DISCRETIONARY:
            amt, why = r.forecast, "Discretionary: keep at forecast (goals on track)"
        else:
            amt, why = r.forecast, "Keep at forecast"
        rows.append(dict(category=r.category, forecast=round(r.forecast), recommended=round(amt), reason=why))
    out = pd.DataFrame(rows)
    out.attrs.update(cut=cut, goal_need=need, shortfall=shortfall)
    return out


# ====================================================================================== goals
def add_goal(username, name, target, saved, target_date):
    with conn() as c:
        c.execute("INSERT INTO goals (username,name,target_amount,saved_amount,target_date) VALUES (?,?,?,?,?)",
                  (username, name, float(target), float(saved), str(target_date)))


def update_goal_saved(goal_id, saved):
    with conn() as c:
        c.execute("UPDATE goals SET saved_amount=? WHERE id=?", (float(saved), int(goal_id)))


def delete_goal(goal_id):
    with conn() as c:
        c.execute("DELETE FROM goals WHERE id=?", (int(goal_id),))


def get_goals(username):
    with conn() as c:
        return pd.read_sql("SELECT * FROM goals WHERE username=?", c, params=(username,))


def goal_feasibility(target, saved, monthly_savings, target_date):
    left = max(target - saved, 0)
    if left == 0:
        return "Goal reached 🎉", True
    if monthly_savings <= 0:
        return "Your predicted savings are ≤ ₹0 next month, so this goal is not reachable yet. See the budget page.", False
    months = int(np.ceil(left / monthly_savings))
    deadline = max((pd.Timestamp(target_date) - pd.Timestamp(date.today())).days / 30.4, 0)
    ok = months <= deadline
    msg = (f"At your predicted savings of {inr(monthly_savings)}/month you will reach this in {months} month"
           f"{'s' if months != 1 else ''} - {'on track ✅' if ok else 'later than your target date ⚠️'}")
    return msg, ok


# ====================================================================================== insights
def kpis(tx, month):
    """Income, expenses, net savings and savings rate for one calendar month (Period)."""
    m = tx[tx["date"].dt.to_period("M") == month]
    income = m.loc[m["type"] == "credit", "amount"].sum()
    exp = m.loc[(m["type"] == "debit") & ~m["category"].isin(NON_EXPENSE), "amount"].sum()
    return dict(income=income, expenses=exp, savings=income - exp, rate=(income - exp) / income if income else 0)


def persona(username, models, info):
    tx = load_transactions(username)
    if tx.empty or "persona" not in models:
        return None
    feats = F.persona_features(tx.assign(user_id=username))
    if feats["n_months"].iat[0] < info["persona"].get("min_months", 3):
        return {"cold_start": True, "n_months": int(feats["n_months"].iat[0])}
    cluster = str(models["persona"].predict(feats[F.PERSONA_FEATURES])[0])
    p = info["persona"]["persona_names"][cluster]
    return dict(cold_start=False, cluster=cluster, **p, user=feats.iloc[0].to_dict(),
                avg=info["persona"]["persona_profiles"][cluster])


def patterns(tx):
    d = tx[(tx["type"] == "debit") & ~tx["category"].isin(NON_EXPENSE)].copy()
    if d.empty:
        return {}
    d["weekend"] = d["date"].dt.dayofweek >= 5
    n_days = max((d["date"].max() - d["date"].min()).days + 1, 1)
    we = d.loc[d["weekend"], "amount"].sum() / (n_days * 2 / 7)
    wd = d.loc[~d["weekend"], "amount"].sum() / (n_days * 5 / 7)
    d["month"] = d["date"].dt.to_period("M")
    months = sorted(d["month"].unique())
    growth = pd.Series(dtype=float)
    if len(months) >= 6:
        recent = d[d["month"].isin(months[-3:])].groupby("category")["amount"].sum()
        prev = d[d["month"].isin(months[-6:-3])].groupby("category")["amount"].sum()
        growth = ((recent - prev) / prev.replace(0, np.nan)).dropna().sort_values(ascending=False)
    clean = _to_clean(d[["date", "description", "amount", "type"]])
    merchants = clean.groupby("merchant")["amount"].agg(["sum", "count"]).sort_values("sum", ascending=False).head(8)
    return dict(weekend_daily=we, weekday_daily=wd, growth=growth, merchants=merchants,
                by_weekday=d.groupby(d["date"].dt.day_name())["amount"].sum() / max(n_days / 7, 1))


def weekend_test(tx):
    """Statistical decision: is daily spend on weekends different from weekdays? Mann-Whitney U (non-parametric)."""
    from scipy.stats import mannwhitneyu
    d = tx[(tx["type"] == "debit") & ~tx["category"].isin(NON_EXPENSE)]
    if d.empty:
        return None
    days = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    daily = d.groupby(d["date"].dt.normalize())["amount"].sum().reindex(days, fill_value=0)
    we, wd = daily[daily.index.dayofweek >= 5], daily[daily.index.dayofweek < 5]
    if len(we) < 8 or len(wd) < 8:
        return None
    stat, p = mannwhitneyu(we, wd, alternative="two-sided")
    return dict(p_value=float(p), weekend_median=float(we.median()), weekday_median=float(wd.median()),
                significant=bool(p < 0.05), n_weekend=len(we), n_weekday=len(wd))


def unusual_transactions(tx, k=3.0, months=3):
    """Flag debits above Q3 + k*IQR of the user's own history in that category (shown for recent months)."""
    d = tx[tx["type"] == "debit"].copy()
    if len(d) < 30:
        return d.iloc[0:0]
    g = d.groupby("category")["amount"]
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    d["limit"] = q3 + k * (q3 - q1)
    d["times_typical"] = d["amount"] / g.transform("median")
    recent = d["date"] >= d["date"].max() - pd.DateOffset(months=months)
    out = d[(d["amount"] > d["limit"]) & recent & (g.transform("size") >= 10)]
    return out.sort_values("date", ascending=False)[["date", "description", "category", "amount", "times_typical"]]


def feedback_stats():
    """Simple model-health signal from the app's own data: how often users correct the categorizer."""
    with conn() as c:
        r = c.execute("""SELECT COUNT(*) n, SUM(user_corrected) corrected, SUM(needs_review) review,
                         AVG(confidence) conf FROM transactions WHERE source != 'autopay_simulated'""").fetchone()
        top = pd.read_sql("""SELECT predicted_category AS predicted, category AS corrected_to, COUNT(*) AS n
                             FROM transactions WHERE user_corrected=1 GROUP BY 1,2 ORDER BY n DESC LIMIT 10""", c)
    n = r["n"] or 0
    return dict(n=n, correction_rate=(r["corrected"] or 0) / n if n else 0, review_rate=(r["review"] or 0) / n if n else 0,
                avg_confidence=r["conf"] or 0, top_corrections=top)


def recurring_share(tx, subs):
    d = tx[tx["type"] == "debit"]
    if d.empty or subs.empty:
        return 0.0
    clean = _to_clean(d[["date", "description", "amount", "type"]])
    return float(clean.loc[clean["merchant"].isin(subs["merchant"]), "amount"].sum() / clean["amount"].sum())
