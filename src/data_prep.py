"""Data acquisition, synthetic generation and cleaning.

Run:  python -m src.data_prep
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# --------------------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------------------
EXPENSE_CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Shopping", "Bills & Utilities", "Rent",
    "Entertainment", "Health", "Education", "Travel", "Subscriptions", "EMI & Loans",
    "Investments", "Personal Care", "Others",
]
INCOME_CATEGORIES = ["Salary", "Other Income"]
ALL_CATEGORIES = EXPENSE_CATEGORIES + INCOME_CATEGORIES
ESSENTIALS = ["Rent", "Groceries", "Bills & Utilities", "EMI & Loans", "Health"]
DISCRETIONARY = ["Food & Dining", "Shopping", "Entertainment", "Travel"]
NON_EXPENSE = ["Investments"]  # money moved to savings, not "spent"

# HCES 2023-24 urban share of MPCE (%) mapped to our taxonomy.
# cited = number from MoSPI factsheet / press note; approx = approximate, verify (see data/SOURCES.md)
HCES_URBAN_SHARES = {
    "Groceries": 39.68 - 11.09,  # cited: food total minus processed food/beverages
    "Food & Dining": 11.09,      # cited: beverages, refreshments & processed food
    "Transport": 8.46,           # cited: conveyance
    "Shopping": 6.87 + 5.4,      # cited durables 6.87 + approx clothing/footwear 5.4
    "Rent": 6.58,                # cited
    "Entertainment": 2.0,        # approx part of misc goods & entertainment (6.92 cited)
    "Personal Care": 4.0,        # approx toilet articles
    "Bills & Utilities": 8.0,    # approx fuel & light + consumer services (phone, internet)
    "Health": 6.0,               # approx medical
    "Education": 6.0,            # approx education
    "Travel": 2.0,               # approx (hotel/long-distance part of conveyance & rent)
    "Others": 5.0,               # remainder
}

ARCHETYPES = ["Saver", "Lifestyle Spender", "Family Budgeter", "Student/Early Career", "Debt-Heavy"]
ARCHETYPE_SHIFT = {  # multiplicative shift on HCES consumption shares
    "Saver": {"Shopping": 0.6, "Food & Dining": 0.6, "Entertainment": 0.6, "Travel": 0.7},
    "Lifestyle Spender": {"Food & Dining": 1.9, "Shopping": 1.6, "Entertainment": 2.0, "Travel": 2.2,
                          "Personal Care": 1.4, "Groceries": 0.8},
    "Family Budgeter": {"Groceries": 1.35, "Education": 1.9, "Health": 1.3, "Food & Dining": 0.7,
                        "Entertainment": 0.7},
    "Student/Early Career": {"Education": 1.5, "Food & Dining": 1.4, "Transport": 1.2, "Groceries": 0.7,
                             "Health": 0.6},
    "Debt-Heavy": {"Shopping": 0.8, "Travel": 0.6, "Entertainment": 0.8},
}
SAVINGS_RANGE = {"Saver": (0.30, 0.45), "Lifestyle Spender": (0.0, 0.10), "Family Budgeter": (0.10, 0.20),
                 "Student/Early Career": (0.02, 0.12), "Debt-Heavy": (0.0, 0.08)}

MERCHANTS = {
    "Food & Dining": ["ZOMATO", "SWIGGY", "DOMINOS", "MCDONALDS", "STARBUCKS", "HALDIRAMS", "CAFE COFFEE DAY",
                      "KFC", "BARBEQUE NATION", "CHAAYOS", "BURGER KING", "THEOBROMA"],
    "Groceries": ["BIGBASKET", "BLINKIT", "ZEPTO", "DMART", "RELIANCE FRESH", "MORE SUPERMARKET", "JIOMART",
                  "NATURES BASKET", "SWIGGY INSTAMART", "SPENCERS"],
    "Transport": ["UBER", "OLA", "RAPIDO", "INDIAN OIL", "HP PETROL", "BHARAT PETROLEUM", "DELHI METRO",
                  "BMTC", "FASTAG"],
    "Shopping": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "CROMA", "RELIANCE DIGITAL", "DECATHLON",
                 "LIFESTYLE", "WESTSIDE", "IKEA"],
    "Bills & Utilities": ["BESCOM", "TATA POWER", "ADANI ELECTRICITY", "AIRTEL", "JIO", "VODAFONE IDEA",
                          "ACT FIBERNET", "MAHANAGAR GAS", "BWSSB WATER", "AIRTEL XSTREAM", "AIRTEL POSTPAID",
                          "HP GAS", "INDANE GAS"],
    "Rent": ["HOUSE RENT", "NOBROKER RENT", "RENT TRANSFER"],
    "Entertainment": ["BOOKMYSHOW", "PVR", "INOX", "STEAM", "PLAYSTATION", "WONDERLA", "SMAAASH"],
    "Health": ["APOLLO PHARMACY", "MEDPLUS", "PHARMEASY", "TATA 1MG", "PRACTO", "MAX HOSPITAL",
               "THYROCARE"],
    "Education": ["BYJUS", "UNACADEMY", "COURSERA", "UDEMY", "SCHOOL FEES", "UNIVERSITY FEES", "VEDANTU"],
    "Travel": ["MAKEMYTRIP", "IRCTC", "INDIGO", "GOIBIBO", "OYO", "CLEARTRIP", "AIR INDIA", "REDBUS"],
    "Subscriptions": ["NETFLIX", "SPOTIFY", "HOTSTAR", "AMAZON PRIME", "YOUTUBE PREMIUM", "ZEE5", "SONYLIV",
                      "APPLE SERVICES"],
    "EMI & Loans": ["HDFC BANK EMI", "BAJAJ FINSERV", "ICICI LOAN", "SBI HOME LOAN", "KREDITBEE",
                    "TATA CAPITAL"],
    "Investments": ["ZERODHA", "GROWW", "HDFC MF SIP", "AXIS MF SIP", "PPF DEPOSIT", "NPS CONTRIBUTION",
                    "LIC OF INDIA"],
    "Personal Care": ["NYKAA", "URBAN COMPANY", "LAKME SALON", "CULT FIT", "NATURALS SALON", "PURPLLE"],
    "Others": ["ATM WITHDRAWAL", "CASH WITHDRAWAL", "DONATION", "GIFT", "PAYTM WALLET", "AMAZON PAY"],
}
EMPLOYERS = ["INFOSYS LTD", "TCS LTD", "WIPRO LTD", "HCL TECH", "ACCENTURE", "DELOITTE", "STARTUP PVT LTD",
             "RELIANCE IND", "HDFC BANK LTD", "GOVT OF INDIA"]
PEOPLE = ["RAHUL SHARMA", "PRIYA SINGH", "AMIT KUMAR", "SNEHA PATEL", "VIKRAM RAO", "ANJALI GUPTA",
          "ROHIT VERMA", "KAVYA NAIR", "ARJUN MEHTA", "NEHA JOSHI"]
CITIES = ["MUMBAI", "BANGALORE", "DELHI", "PUNE", "HYDERABAD", "CHENNAI", "KOLKATA", "JAIPUR", "INDORE"]
PUBLIC_FILES = {"hf_transactions": "hf_transactions.csv", "upi_2024": "upi_transactions_2024.csv",
                "credit_card_india": "credit_card_india.csv"}

# Public dataset #1 uses 7 coarse categories that sometimes disagree with our taxonomy (e.g. Blinkit = "Food",
# Netflix = "Entertainment", Gym = "Health"), so it is mapped per merchant first, then per category.
HF_MERCHANT_MAP = {
    "Blinkit": "Groceries", "BigBasket": "Groceries", "Local Grocery": "Groceries",
    "Netflix": "Subscriptions", "Prime Video": "Subscriptions", "Disney+ Hotstar": "Subscriptions",
    "Spotify": "Subscriptions", "Gym Membership": "Personal Care",
    "Parking Fee": "Transport", "Ola Cab": "Transport", "Uber Ride": "Transport", "Auto Rickshaw": "Transport",
    "Shell Fuel": "Transport", "Petrol Pump": "Transport", "IndiGo Flight": "Travel", "IRCTC Ticket": "Travel",
    "Monthly Salary": "Salary",
}
HF_CATEGORY_MAP = {"Food": "Food & Dining", "Shopping": "Shopping", "Bills": "Bills & Utilities",
                   "Entertainment": "Entertainment", "Health": "Health", "Travel": "Travel", "Income": "Other Income"}
# Public dataset #2 (UPI 2024) merchant_category -> our taxonomy
UPI_CATEGORY_MAP = {"Grocery": "Groceries", "Food": "Food & Dining", "Shopping": "Shopping", "Fuel": "Transport",
                    "Transport": "Transport", "Utilities": "Bills & Utilities", "Entertainment": "Entertainment",
                    "Healthcare": "Health", "Education": "Education", "Other": "Others"}


def _norm_merchant(name):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", str(name).upper())).strip()


_ORIGINAL_MERCHANTS = {m for ms in MERCHANTS.values() for m in ms}


def _hf_raw():
    path = RAW / PUBLIC_FILES["hf_transactions"]
    if not path.exists():
        return None
    hf = pd.read_csv(path)
    hf["merchant_raw"] = hf["description"].str.replace(r"\s*txn_\d+$", "", regex=True).str.strip()
    return hf


def hf_holdout_merchants():
    """Half of dataset #1's merchants that are NEW to us (not in our own lists) are held out completely:
    never added to the generator, never used for training. They give an honest unseen-merchant external test."""
    hf = _hf_raw()
    if hf is None:
        return set()
    new = sorted(m for m in hf["merchant_raw"].unique() if _norm_merchant(m) not in _ORIGINAL_MERCHANTS)
    perm = np.random.default_rng(SEED).permutation(len(new))
    return {new[i] for i in perm[: len(new) // 2]}


def load_hf_labeled():
    """Public dataset #1 in our schema (date, description, amount, type, category, merchant_raw, holdout). None if absent."""
    hf = _hf_raw()
    if hf is None:
        return None
    cat = hf["merchant_raw"].map(HF_MERCHANT_MAP).fillna(hf["category"].map(HF_CATEGORY_MAP)).fillna("Others")
    hold = hf_holdout_merchants()
    return pd.DataFrame({"date": pd.to_datetime(hf["date"]), "description": hf["description"],
                         "amount": hf["amount"].abs(), "type": np.where(hf["category"] == "Income", "credit", "debit"),
                         "category": cat, "merchant_raw": hf["merchant_raw"],
                         "holdout": hf["merchant_raw"].isin(hold)})


def _extend_merchants_from_hf():
    """Add dataset #1's real merchant names (except the held-out half) to the generator's lists (deterministic)."""
    hf = load_hf_labeled()
    if hf is None:
        return 0
    added = 0
    for m, c in sorted(set(zip(hf.loc[~hf["holdout"], "merchant_raw"], hf.loc[~hf["holdout"], "category"]))):
        name = _norm_merchant(m)
        if c in MERCHANTS and name not in MERCHANTS[c]:
            MERCHANTS[c].append(name)
            added += 1
    return added


N_HF_MERCHANTS = _extend_merchants_from_hf()
KNOWN_MERCHANTS = sorted({m for ms in MERCHANTS.values() for m in ms} | {"SALARY"} | set(EMPLOYERS))
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# map external datasets' categories to our taxonomy (extend after inspecting the real columns)
CATEGORY_MAP = {
    "food": "Food & Dining", "dining": "Food & Dining", "restaurant": "Food & Dining",
    "grocery": "Groceries", "groceries": "Groceries", "travel": "Travel", "transport": "Transport",
    "fuel": "Transport", "shopping": "Shopping", "bills": "Bills & Utilities", "utilities": "Bills & Utilities",
    "entertainment": "Entertainment", "health": "Health", "healthcare": "Health", "education": "Education",
    "rent": "Rent", "emi": "EMI & Loans", "loan": "EMI & Loans", "investment": "Investments",
    "subscription": "Subscriptions", "personal care": "Personal Care", "salary": "Salary", "others": "Others",
    "other": "Others", "entertainment ": "Entertainment",
}


# --------------------------------------------------------------------------------------
# 2.1 Public data (manual download)
# --------------------------------------------------------------------------------------
def load_public():
    """Load whichever public datasets are present in data/raw/. Returns dict name -> DataFrame."""
    out = {}
    for name, fname in PUBLIC_FILES.items():
        path = RAW / fname
        if not path.exists():
            print(f"[load_public] {fname} not found in data/raw/ - skipping (see data/SOURCES.md)")
            continue
        df = pd.read_csv(path)
        print(f"[load_public] {fname}: {df.shape}, columns={list(df.columns)}")
        out[name] = df
    return out


def upi_calibration():
    """Ticket-size distribution and weekend ratio per category from public dataset #2 (successful UPI payments).
    Returns DataFrame indexed by our category with median, log_sd, mean_ticket, weekend_ratio; None if absent."""
    path = RAW / PUBLIC_FILES["upi_2024"]
    if not path.exists():
        return None
    u = pd.read_csv(path, usecols=["merchant_category", "amount (INR)", "transaction_status", "is_weekend"])
    u = u[(u["transaction_status"] == "SUCCESS") & (u["amount (INR)"] > 0)]
    u["category"] = u["merchant_category"].map(UPI_CATEGORY_MAP)
    rows = {}
    for c, g in u.dropna(subset=["category"]).groupby("category"):
        amt = g["amount (INR)"].astype(float)
        we = g["is_weekend"].astype(bool)
        log_sd = float(np.log(amt).std())
        rows[c] = dict(median=float(amt.median()), log_sd=log_sd, mean_ticket=float(amt.median() * np.exp(log_sd ** 2 / 2)),
                       weekend_ratio=float((we.sum() / 2) / max((~we).sum() / 5, 1)), n=len(g))
    return pd.DataFrame(rows).T


def map_external_categories(df, col):
    """Map an external category column to our taxonomy; unknown values become 'Others'."""
    return df[col].astype(str).str.strip().str.lower().map(CATEGORY_MAP).fillna("Others")


# --------------------------------------------------------------------------------------
# 2.3 Synthetic generator
# --------------------------------------------------------------------------------------
def _ref(rng, n=None):
    n = n or rng.integers(6, 13)
    return "".join(rng.choice(list("0123456789"), n))


def _typo(rng, s):
    if len(s) < 5:
        return s
    i = rng.integers(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def make_description(rng, merchant, category, date, mode=None):
    """Turn a merchant into an Indian bank-statement style narration, with noise."""
    if mode is None:
        if category in ("Rent",):
            mode = rng.choice(["NEFT", "UPI", "IMPS"])
        elif category in ("EMI & Loans", "Investments"):
            mode = rng.choice(["ACH", "ACH", "NACH"])
        elif category in ("Shopping", "Subscriptions", "Travel"):
            mode = rng.choice(["CARD", "UPI", "POS"], p=[0.5, 0.35, 0.15])
        elif merchant in ("ATM WITHDRAWAL", "CASH WITHDRAWAL"):
            mode = "ATM"
        else:
            mode = rng.choice(["UPI", "POS", "CARD"], p=[0.65, 0.2, 0.15])
    m = merchant
    if rng.random() < 0.03:
        m = _typo(rng, m)
    if rng.random() < 0.05 and len(m) > 8:
        m = m[: rng.integers(6, len(m))]  # truncation
    mon = MONTHS[date.month - 1]
    if mode == "UPI":
        tpl = rng.choice([f"UPI/{m}/{_ref(rng)}/Payment", f"UPI-{m.replace(' ', '')}-{m.split()[0].lower()}@ybl-{_ref(rng)}",
                          f"UPI/P2M/{_ref(rng)}/{m}"])
    elif mode == "POS":
        tpl = rng.choice([f"POS {m} {rng.choice(CITIES)}", f"POS {_ref(rng, 4)} {m}"])
    elif mode == "CARD":
        tpl = rng.choice([f"{m.replace(' ', '')}*ORDER {_ref(rng, 5)}", f"ECOM {m} {_ref(rng)}"])
    elif mode == "NEFT":
        tpl = rng.choice([f"NEFT-{m}-{mon}", f"NEFT/{_ref(rng)}/{m}"])
    elif mode == "IMPS":
        tpl = f"IMPS/{_ref(rng)}/{m}"
    elif mode in ("ACH", "NACH"):
        tpl = rng.choice([f"ACH D- {m} {_ref(rng)}", f"NACH DR {m} {_ref(rng, 8)}"])
    elif mode == "ATM":
        tpl = f"ATM WDL {_ref(rng, 6)} {rng.choice(CITIES)}"
    else:
        tpl = m
    r = rng.random()
    if r < 0.3:
        tpl = tpl.lower()
    elif r < 0.4:
        tpl = tpl.title()
    return tpl


def _recurring_plan(rng, user, months):
    """List of recurring streams for a user: dict(merchant, category, amount, period_days, start, end, vary, recurring)."""
    inc, arch = user["monthly_income"], user["archetype"]
    plan = []

    def add(merchant, cat, amount, period=30, vary=0.0, recurring=1, start=0, end=None, day=None):
        plan.append(dict(merchant=merchant, category=cat, amount=round(float(amount), 0), period=period,
                         vary=vary, recurring=recurring, start=start, end=end if end is not None else months,
                         day=int(day if day is not None else rng.integers(1, 28))))

    if rng.random() < {"Saver": 0.5, "Student/Early Career": 0.8}.get(arch, 0.6):
        add(rng.choice(MERCHANTS["Rent"]), "Rent", inc * rng.uniform(0.15, 0.28), day=rng.integers(1, 6))
    n_emi = {"Debt-Heavy": rng.integers(2, 4)}.get(arch, int(rng.random() < 0.3))
    for m in rng.choice(MERCHANTS["EMI & Loans"], n_emi, replace=False):
        share = rng.uniform(0.10, 0.18) if arch == "Debt-Heavy" else rng.uniform(0.05, 0.12)
        add(m, "EMI & Loans", inc * share, day=rng.choice([5, 10]))
    n_sip = {"Saver": rng.integers(1, 4)}.get(arch, int(rng.random() < 0.4))
    for m in rng.choice(["ZERODHA", "GROWW", "HDFC MF SIP", "AXIS MF SIP", "NPS CONTRIBUTION"], n_sip, replace=False):
        share = rng.uniform(0.06, 0.12) if arch == "Saver" else rng.uniform(0.02, 0.05)
        add(m, "Investments", inc * share, day=rng.choice([5, 15]))
    if rng.random() < 0.35:  # quarterly insurance premium
        add("LIC OF INDIA", "Investments", inc * rng.uniform(0.05, 0.1), period=91)
    n_sub = {"Lifestyle Spender": rng.integers(2, 5), "Saver": rng.integers(0, 2)}.get(arch, rng.integers(0, 3))
    prices = {"NETFLIX": 649, "SPOTIFY": 119, "HOTSTAR": 299, "AMAZON PRIME": 299, "YOUTUBE PREMIUM": 149,
              "ZEE5": 99, "SONYLIV": 299, "APPLE SERVICES": 179}
    for m in rng.choice(list(prices), n_sub, replace=False):
        r = rng.random()
        start = rng.integers(0, months // 2) if r < 0.3 else (months - rng.integers(2, 5) if r < 0.4 else 0)
        end = rng.integers(start + 4, months) if (rng.random() < 0.2 and start + 4 < months) else months  # some cancelled
        add(m, "Subscriptions", prices[m], start=start, end=end)
    if rng.random() < 0.7:
        add(rng.choice(["ACT FIBERNET", "AIRTEL XSTREAM"]), "Bills & Utilities", rng.choice([499, 699, 999]))
    if rng.random() < 0.9:  # prepaid mobile: 28-day cycle
        add(rng.choice(["JIO", "AIRTEL", "VODAFONE IDEA"]), "Bills & Utilities", rng.choice([239, 299, 349, 399]),
            period=28, vary=0.0)
    if rng.random() < 0.25:
        add("CULT FIT", "Personal Care", rng.choice([999, 1499, 1999]))
    if rng.random() < 0.35:  # hard positive: postpaid bill, recurring but amount varies a little
        add("AIRTEL POSTPAID", "Bills & Utilities", rng.choice([499, 599, 799]), vary=0.08)
    if rng.random() < 0.4:  # hard positive: monthly salary to house help, paid to a person's name via UPI
        add(rng.choice(PEOPLE), "Others", round(rng.uniform(3000, 9000), -2), day=rng.integers(1, 8))
    if rng.random() < 0.4:  # hard negative: LPG refill, ~fixed price but irregular timing
        add(rng.choice(["HP GAS", "INDANE GAS"]), "Bills & Utilities", rng.choice([853, 903, 950]), period=-1,
            recurring=0)
    # hard negative: fixed round amount but irregular timing (top-ups), not a subscription
    if rng.random() < 0.6:
        add(rng.choice(["FASTAG", "DELHI METRO", "BMTC"]), "Transport", rng.choice([500, 1000, 2000]),
            period=-1, recurring=0)
    if rng.random() < 0.4:
        add(rng.choice(["PAYTM WALLET", "AMAZON PAY"]), "Others", rng.choice([500, 1000]), period=-1, recurring=0)
    # hard negative: monthly but varying electricity bill (not a fixed subscription)
    add(rng.choice(["BESCOM", "TATA POWER", "ADANI ELECTRICITY"]), "Bills & Utilities",
        inc * rng.uniform(0.02, 0.04), vary=0.35, recurring=0)
    return plan


def generate_data(n_users=300, months=24, seed=SEED, start="2024-11-01"):
    """Generate users, transactions and recurring labels. Returns (users, transactions, labels)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    calib = upi_calibration()
    if calib is not None:
        print(f"[generate] ticket sizes and weekend pattern calibrated from UPI 2024 for {list(calib.index)}")
        PROCESSED.mkdir(parents=True, exist_ok=True)
        calib.round(4).to_csv(PROCESSED / "upi_calibration.csv")
    if N_HF_MERCHANTS:
        print(f"[generate] {N_HF_MERCHANTS} merchant names added from public dataset #1 "
              f"({len(hf_holdout_merchants())} new merchants held out for the external test)")
    base = pd.Series(HCES_URBAN_SHARES)
    base = base / base.sum()
    users, rows, labels = [], [], []

    for u in range(n_users):
        uid = f"U{u:04d}"
        arch = ARCHETYPES[u % len(ARCHETYPES)]
        tier = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
        inc = float(np.clip(rng.lognormal(np.log(65000), 0.5), 18000, 400000))
        if arch == "Student/Early Career":
            inc = float(np.clip(rng.lognormal(np.log(28000), 0.3), 12000, 60000))
        inc *= {1: 1.15, 2: 1.0, 3: 0.85}[tier]
        inc = round(inc, -2)
        user = dict(user_id=uid, monthly_income=inc, city_tier=tier, archetype=arch)
        users.append(user)
        save_rate = rng.uniform(*SAVINGS_RANGE[arch])
        shares = base.copy()
        for c, f in ARCHETYPE_SHIFT[arch].items():
            shares[c] *= f
        shares *= rng.lognormal(0, 0.15, len(shares))  # personal taste
        shares /= shares.sum()
        plan = _recurring_plan(rng, user, months)
        for p in plan:
            labels.append(dict(user_id=uid, merchant=p["merchant"], is_recurring=p["recurring"]))
        fav = {c: rng.choice(MERCHANTS[c], size=min(3, len(MERCHANTS[c])), replace=False) for c in MERCHANTS}
        trend = rng.normal(0.004, 0.004)
        ar = pd.Series(0.0, index=shares.index)
        salary_day = int(rng.choice([1, 1, 28, 30]))
        employer = rng.choice(EMPLOYERS)

        for mi in range(months):
            mstart = start + pd.DateOffset(months=mi)
            ndays = mstart.days_in_month
            m_inc = inc * (1 + trend) ** mi

            def add_row(day, desc, amt, typ, cat):
                d = mstart + pd.Timedelta(days=int(np.clip(day, 1, ndays)) - 1)
                rows.append(dict(user_id=uid, date=d, description=desc, amount=round(float(amt), 2),
                                 type=typ, category=cat))

            # salary (+ jitter), occasional bonus and small other income
            sd = salary_day + int(rng.integers(-2, 3))
            add_row(sd, rng.choice([f"NEFT CR SALARY {employer} {MONTHS[mstart.month - 1]}",
                                    f"SAL CREDIT {employer} {_ref(rng, 8)}"]), m_inc, "credit", "Salary")
            if mstart.month == 3 and rng.random() < 0.5:
                add_row(rng.integers(20, 28), f"NEFT CR BONUS {employer}", m_inc * rng.uniform(0.5, 1.5),
                        "credit", "Other Income")
            if rng.random() < 0.3:
                add_row(rng.integers(1, 28), rng.choice(["INTEREST CREDIT SB", "REFUND AMAZON " + _ref(rng),
                                                         f"UPI/{rng.choice(PEOPLE)}/{_ref(rng)}/received"]),
                        rng.uniform(100, 3000), "credit", "Other Income")

            # recurring streams
            recurring_spend = pd.Series(0.0, index=EXPENSE_CATEGORIES)
            for p in plan:
                if not (p["start"] <= mi < p["end"]):
                    continue
                amt = p["amount"] * (1 + trend) ** mi if p["category"] in ("Bills & Utilities",) and p["vary"] else p["amount"]
                if p["category"] == "Subscriptions" and mi >= months * 0.6 and sum(map(ord, p["merchant"])) % 3 == 0:
                    amt = round(amt * 1.15)  # price hike
                if p["period"] == 30:
                    days = [] if rng.random() < 0.04 else [p["day"] + int(rng.integers(-2, 3))]  # occasional miss
                elif p["period"] == 28:
                    first = (mi * 30) % 28
                    days = [d for d in range(1 + (28 - first) % 28, ndays + 1, 28)]
                elif p["period"] == -1:  # irregular: 0-2 purchases at random days
                    days = list(rng.integers(1, ndays + 1, rng.choice([0, 1, 1, 2])))
                else:  # quarterly
                    days = [p["day"]] if mi % 3 == 0 else []
                for d in days:
                    a = amt * (rng.lognormal(0, p["vary"]) if p["vary"] else rng.uniform(0.995, 1.005) if p["category"] != "Subscriptions" else 1)
                    mode = "UPI" if p["merchant"] in PEOPLE else None
                    add_row(d, make_description(rng, p["merchant"], p["category"], mstart, mode), a, "debit", p["category"])
                    recurring_spend[p["category"]] += a

            # variable spending: income x (1 - savings) x shares, with AR(1) noise
            spend_budget = m_inc * (1 - save_rate) - recurring_spend[[c for c in recurring_spend.index
                                                                      if c not in NON_EXPENSE]].sum()
            spend_budget = max(spend_budget, m_inc * 0.15)
            ar = 0.5 * ar + rng.normal(0, 0.18, len(ar))
            season = pd.Series(1.0, index=shares.index)
            if mstart.month in (10, 11):
                season["Shopping"] = 1.35
            if mstart.month in (5, 12):
                season["Travel"] = 1.6
            if mstart.month in (6, 7):
                season["Education"] = 1.5
            budgets = spend_budget * shares * np.exp(ar) * season
            for cat, b in budgets.items():
                b = b - recurring_spend.get(cat, 0) * 0.5 if cat in ("Bills & Utilities", "Personal Care", "Rent") else b
                if b <= 50 or (cat == "Rent" and any(p["category"] == "Rent" for p in plan)):
                    continue
                if cat == "Rent":
                    b *= 0.3  # no rent stream (owns home): small maintenance payments only
                # weekend pattern: from UPI 2024 when available (it shows ~no weekend effect), else uniform
                wr = float(calib.loc[cat, "weekend_ratio"]) if calib is not None and cat in calib.index else 1.0
                dw = np.array([wr if (mstart + pd.Timedelta(days=i)).dayofweek >= 5 else 1.0 for i in range(ndays)])
                dw = dw / dw.sum()
                if calib is not None and cat in calib.index:
                    # number of payments and their spread follow the real UPI ticket-size distribution
                    c = calib.loc[cat]
                    n = int(np.clip(rng.poisson(max(b / c["mean_ticket"], 1)), 1, 35))
                    draws = rng.lognormal(np.log(c["median"]), c["log_sd"], n)
                    parts = draws * b / draws.sum()
                else:
                    n = {"Food & Dining": 10, "Groceries": 8, "Transport": 9, "Shopping": 3, "Others": 3}.get(cat, 2)
                    n = max(1, int(rng.poisson(n)))
                    parts = rng.dirichlet(np.ones(n)) * b
                for amt in parts:
                    if amt < 20:
                        continue
                    merch = rng.choice(fav[cat]) if rng.random() < 0.7 else rng.choice(MERCHANTS[cat])
                    if rng.random() < 0.05:  # ambiguous person-to-person payment
                        desc = f"UPI/{rng.choice(PEOPLE)}/{_ref(rng)}/{rng.choice(['Payment', 'paid', 'UPI'])}"
                    else:
                        desc = make_description(rng, merch, cat, mstart)
                    add_row(int(rng.choice(ndays, p=dw)) + 1, desc, amt, "debit", cat)

            # one-off big expenses
            if rng.random() < 0.06:
                cat, m = [("Shopping", "CROMA"), ("Health", "MAX HOSPITAL"), ("Travel", "INDIGO")][rng.integers(0, 3)]
                add_row(rng.integers(1, ndays + 1), make_description(rng, m, cat, mstart),
                        m_inc * rng.uniform(0.3, 0.8), "debit", cat)

    users = pd.DataFrame(users)
    tx = pd.DataFrame(rows).sort_values(["user_id", "date"]).reset_index(drop=True)
    tx.insert(0, "txn_id", np.arange(1, len(tx) + 1))
    labels = pd.DataFrame(labels).drop_duplicates(["user_id", "merchant"])
    validate_generated(users, tx, labels, base)
    return users, tx, labels


def validate_generated(users, tx, labels, target_shares):
    assert (tx["amount"] > 0).all(), "negative or zero amounts found"
    assert tx["category"].isin(ALL_CATEGORIES).all()
    spend = tx[(tx["type"] == "debit") & tx["category"].isin(target_shares.index)]
    actual = spend.groupby("category")["amount"].sum()
    actual = actual / actual.sum()
    diff = (actual - target_shares).abs().max()
    print(f"[validate] max abs difference between actual and HCES-based target shares: {diff:.3f} "
          f"(archetype shifts and recurring streams move it on purpose)")
    assert diff < 0.15, "category shares drifted too far from targets"
    merged = labels.merge(users[["user_id"]], on="user_id")
    assert len(merged) == len(labels), "labels reference unknown users"
    print(f"[validate] users={len(users)} txns={len(tx):,} recurring labels={len(labels)} "
          f"(positive share {labels['is_recurring'].mean():.2f})")


# --------------------------------------------------------------------------------------
# 2.4 Cleaning
# --------------------------------------------------------------------------------------
_MODE_PATTERNS = [("UPI", r"\bupi\b|^upi|upi[/-]"), ("NEFT", r"\bneft\b"), ("IMPS", r"\bimps\b"),
                  ("ACH", r"\bn?ach\b"), ("POS", r"\bpos\b"), ("ATM", r"\batm\b"), ("CARD", r"\*order|\becom\b")]
_STOP = {"txn", "upi", "neft", "imps", "pos", "ach", "nach", "d", "dr", "cr", "p2m", "payment", "order", "ecom", "paid",
         "wdl", "received"}
_PAYEES = KNOWN_MERCHANTS + PEOPLE  # known merchants + payee names (so "NEHAJOSHI" and "NEHA JOSHI" match)
_KNOWN_LOWER = {m.lower(): m for m in _PAYEES}
_KNOWN_NOSPACE = {m.lower().replace(" ", ""): m for m in _PAYEES}


def payment_mode(desc):
    d = str(desc).lower()
    for mode, pat in _MODE_PATTERNS:
        if re.search(pat, d):
            return mode
    return "OTHER"


def clean_description(desc):
    d = str(desc).lower()
    d = re.sub(r"[/\-*_,:#]", " ", d)
    d = re.sub(r"\S+@\S+", " ", d)                  # VPAs like name@ybl
    d = re.sub(r"\b\w*\d{3,}\w*\b", " ", d)          # reference numbers / ids
    toks = [t for t in re.sub(r"[^a-z0-9 ]", " ", d).split() if t not in _STOP]
    return " ".join(toks) if toks else "unknown"


def extract_merchant(clean):
    """Longest known merchant found in the cleaned text, else fuzzy match, else first two tokens."""
    toks = clean.split()
    best = None
    for n in (3, 2, 1):
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i:i + n])
            hit = _KNOWN_LOWER.get(g) or _KNOWN_NOSPACE.get(g.replace(" ", ""))
            if hit and (best is None or len(hit) > len(best)):
                best = hit
        if best:
            return best
    try:
        from rapidfuzz import fuzz, process
        cand = [k for k in _KNOWN_LOWER if len(k) >= 5]
        res = (process.extractOne(clean, cand, scorer=fuzz.ratio, score_cutoff=80)
               or process.extractOne(clean, cand, scorer=fuzz.token_set_ratio, score_cutoff=88))
        if res:
            return _KNOWN_LOWER[res[0]]
    except ImportError:
        pass
    return " ".join(toks[:2]).upper() if toks else "UNKNOWN"


def clean_transactions(df, verbose=True):
    """Apply every cleaning rule; returns cleaned DataFrame and prints a data-quality summary."""
    report = {"rows_in": len(df)}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad = df["date"].isna() | (df["date"] > pd.Timestamp.today() + pd.Timedelta(days=366))
    report["dropped_bad_date"] = int(bad.sum())
    df = df[~bad]
    n = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "txn_id"])
    report["dropped_duplicates"] = n - len(df)
    df["description"] = df["description"].fillna("UNKNOWN").astype(str)
    report["dropped_missing_amount"] = int(df["amount"].isna().sum())
    df = df[df["amount"].notna()]
    if "type" not in df:
        df["type"] = np.where(df["amount"] < 0, "debit", "credit")
    df["amount"] = df["amount"].abs()
    df = df[df["amount"] > 0]
    df["log_amount"] = np.log1p(df["amount"])
    df["payment_mode"] = df["description"].map(payment_mode)
    df["clean_desc"] = df["description"].map(clean_description)
    df["merchant"] = df["clean_desc"].map(extract_merchant)
    q = df.groupby("user_id")["amount"].transform(lambda s: s.quantile(0.99)) if "user_id" in df else df["amount"].quantile(0.99)
    df["is_large"] = df["amount"] > q  # flagged only, never deleted
    df["day_of_week"] = df["date"].dt.dayofweek
    report["rows_out"] = len(df)
    report["flagged_large"] = int(df["is_large"].sum())
    if verbose:
        print("[clean] data-quality summary:", report)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2.5 Optional: real bank statement import
# --------------------------------------------------------------------------------------
def import_bank_csv(path, date_col="Date", desc_col="Narration", debit_col="Withdrawal Amt.",
                    credit_col="Deposit Amt.", user_id="me"):
    """Map a net-banking CSV/Excel export into our schema. Column names are configurable."""
    path = Path(path)
    raw = pd.read_excel(path) if path.suffix in (".xls", ".xlsx") else pd.read_csv(path)
    debit = pd.to_numeric(raw[debit_col], errors="coerce").fillna(0)
    credit = pd.to_numeric(raw[credit_col], errors="coerce").fillna(0)
    out = pd.DataFrame({
        "user_id": user_id,
        "date": pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce"),
        "description": raw[desc_col].astype(str),
        "amount": np.where(debit > 0, debit, credit),
        "type": np.where(debit > 0, "debit", "credit"),
    })
    return out[out["amount"] > 0].reset_index(drop=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    load_public()
    users, tx, labels = generate_data()
    users.to_csv(RAW / "users.csv", index=False)
    tx.to_csv(RAW / "transactions.csv", index=False)
    labels.to_csv(RAW / "recurring_labels.csv", index=False)
    clean = clean_transactions(tx)
    clean.to_csv(PROCESSED / "transactions_clean.csv", index=False)
    match = clean.merge(labels, on=["user_id", "merchant"], how="inner")
    print(f"[main] saved raw and processed files. Recurring-label merchants matched in cleaned data: "
          f"{match[['user_id', 'merchant']].drop_duplicates().shape[0]}/{len(labels)}")


if __name__ == "__main__":
    main()
