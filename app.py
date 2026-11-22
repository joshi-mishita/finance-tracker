"""Smart Finance Tracker - Streamlit app.  Run: python -m streamlit run app.py"""
import html
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import services as S
from src.data_prep import ALL_CATEGORIES, EXPENSE_CATEGORIES, NON_EXPENSE

st.set_page_config(page_title="Smart Finance Tracker", page_icon=":material/account_balance_wallet:", layout="wide")
RESULTS = Path(__file__).parent / "results"
MLFLOW_URL = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000")

# ====================================================================================== design tokens
INK, PAPER, RULE, MARIGOLD = "#16233F", "#F2F5F0", "#DCE2D9", "#E6A23C"
CREDIT, DEBIT, MUTED = "#1E7B5A", "#B9432F", "#5B6576"
CAT_COLORS = {
    "Food & Dining": "#E6A23C", "Groceries": "#1E7B5A", "Transport": "#3B6EA8", "Shopping": "#B9432F",
    "Bills & Utilities": "#16233F", "Rent": "#6B5B95", "Entertainment": "#D9776B", "Health": "#2A9D8F",
    "Education": "#8A6D3B", "Travel": "#5FA8D3", "Subscriptions": "#B56EA4", "EMI & Loans": "#7A3E3E",
    "Investments": "#4C8C4A", "Personal Care": "#D4B262", "Others": "#9AA3AE", "Salary": "#1E7B5A",
    "Other Income": "#6FAF8F",
}

CSS = f"""
<style>
@font-face {{font-family: "Bricolage Grotesque"; src: url("app/static/fonts/BricolageGrotesque.woff2") format("woff2");
             font-weight: 200 800; font-display: swap;}}
@font-face {{font-family: "IBM Plex Sans"; src: url("app/static/fonts/IBMPlexSans.woff2") format("woff2");
             font-weight: 100 700; font-display: swap;}}
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{font-family: "IBM Plex Sans", system-ui, sans-serif;}}
span[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] *, .material-symbols-rounded {{font-family: "Material Symbols Rounded" !important;}}
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"], div[data-baseweb="base-input"] {{background-color: #F8FAF7 !important;}}
div[data-baseweb="input"], div[data-baseweb="select"] > div {{border: 1px solid {RULE} !important; border-radius: 8px !important;}}
.stApp {{background: {PAPER};}}
.block-container {{padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}}
header[data-testid="stHeader"] {{background: transparent;}}
h1, h2, h3, h4 {{font-family: "Bricolage Grotesque", "IBM Plex Sans", sans-serif; color: {INK}; letter-spacing: -.01em;}}

/* page header */
.ph {{margin: 0 0 1.4rem 0;}}
.ph h1 {{font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin: 0; padding: 0;}}
.ph p {{color: {MUTED}; margin: .35rem 0 0 0; font-size: 1rem; max-width: 70ch;}}

/* ledger strip: the one bold element */
.ledger {{display: grid; background: #FFFFFF; border: 1px solid {RULE}; border-bottom: 4px double {INK};
          border-radius: 12px 12px 4px 4px; margin: 0 0 1.4rem 0; overflow: hidden;}}
.ledger .cell {{padding: 1rem 1.25rem 1.05rem; border-left: 1px solid {RULE}; min-width: 0;}}
.ledger .cell:first-child {{border-left: none;}}
.ledger .lbl {{color: {MUTED}; font-size: .86rem; margin-bottom: .3rem;}}
.ledger .val {{font-family: "Bricolage Grotesque", sans-serif; font-size: 1.85rem; font-weight: 650; color: {INK};
               font-variant-numeric: tabular-nums; line-height: 1.15; white-space: nowrap; overflow: hidden;
               text-overflow: ellipsis;}}
.ledger .val.credit {{color: {CREDIT};}} .ledger .val.debit {{color: {DEBIT};}}
.ledger .note {{font-size: .82rem; color: {MUTED}; margin-top: .25rem; font-variant-numeric: tabular-nums;}}
.ledger .note .up {{color: {CREDIT}; font-weight: 600;}} .ledger .note .down {{color: {DEBIT}; font-weight: 600;}}
@media (max-width: 760px) {{ .ledger {{grid-template-columns: 1fr 1fr !important;}}
  .ledger .cell:nth-child(3) {{border-left: none;}} .ledger .cell:nth-child(n+3) {{border-top: 1px solid {RULE};}} }}

/* panels = keyed bordered containers */
div[class*="st-key-panel_"] {{background: #FFFFFF; border-color: {RULE} !important; border-radius: 12px !important;
                              padding: 1.15rem 1.25rem !important;}}
.sec {{font-family: "Bricolage Grotesque", sans-serif; font-size: 1.12rem; font-weight: 650; color: {INK}; margin: 0 0 .15rem 0;}}
.sec-note {{color: {MUTED}; font-size: .85rem; margin: 0 0 .6rem 0;}}

/* list rows inside panels */
.row {{display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; padding: .55rem 0;
       border-bottom: 1px solid #EDF0EA; font-size: .93rem;}}
.row:last-child {{border-bottom: none;}}
.row .name {{color: {INK}; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}}
.row .meta {{color: {MUTED}; font-size: .82rem;}}
.row .amt {{font-variant-numeric: tabular-nums; color: {INK}; font-weight: 600; white-space: nowrap;}}

/* pills */
.pill {{display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .76rem; font-weight: 600;
        vertical-align: middle; white-space: nowrap;}}
.pill.review {{background: #FBEFD9; color: #8A5A10;}} .pill.ok {{background: #E1F0E8; color: #17603F;}}
.pill.sim {{background: #E6EAF2; color: {INK};}} .pill.warn {{background: #F7E3DE; color: #8E2F1F;}}

.persona-name {{font-family: "Bricolage Grotesque", sans-serif; font-size: 1.7rem; font-weight: 700; color: {INK};
                line-height: 1.15; margin: .1rem 0 .4rem 0;}}
.persona-tip {{border-left: 3px solid {MARIGOLD}; padding: .2rem 0 .2rem .75rem; color: #2E3A4E; margin-top: .6rem;}}
.empty {{padding: 1.6rem 1.4rem .4rem;}}
.empty h3 {{margin: 0 0 .3rem 0; font-size: 1.35rem;}}
.empty p {{color: {MUTED}; margin: 0 0 .6rem 0; max-width: 60ch;}}

/* tabs, buttons, inputs */
button[data-baseweb="tab"] {{font-weight: 500;}}
div[data-baseweb="tab-highlight"] {{background-color: {MARIGOLD} !important; height: 3px;}}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{border-radius: 8px; font-weight: 600;}}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{border: 1px solid {RULE}; border-radius: 10px; overflow: hidden;}}
div[data-testid="stProgress"] div[role="progressbar"] > div > div {{background-color: {CREDIT};}}
.stApp code {{color: {INK}; background: #E9EEE6; font-size: .82em; padding: .05rem .3rem; border-radius: 4px;}}
*:focus-visible {{outline: 2px solid {MARIGOLD} !important; outline-offset: 2px;}}

/* sidebar */
section[data-testid="stSidebar"] {{background: {INK};}}
section[data-testid="stSidebar"] * {{color: #DCE3EE;}}
section[data-testid="stSidebar"] .brand {{display: flex; align-items: center; gap: .6rem; margin: .2rem 0 1.4rem 0;}}
section[data-testid="stSidebar"] .brand .mark {{width: 30px; height: 30px; border-radius: 7px; background: {MARIGOLD};
     display: grid; place-items: center; color: {INK}; font-family: "Bricolage Grotesque"; font-weight: 800; font-size: .95rem;}}
section[data-testid="stSidebar"] .brand .word {{font-family: "Bricolage Grotesque"; font-weight: 700; font-size: 1.12rem; color: #FFFFFF;}}
section[data-testid="stSidebar"] .who {{background: rgba(255,255,255,.06); border-radius: 10px; padding: .7rem .8rem; margin-bottom: 1rem;}}
section[data-testid="stSidebar"] .who b {{color: #FFFFFF; display: block; font-size: .95rem;}}
section[data-testid="stSidebar"] .who span {{color: #9FB0C8; font-size: .82rem;}}
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]),
section[data-testid="stSidebar"] div[data-testid="stRadio"], section[data-testid="stSidebar"] div[data-testid="stRadio"] > div,
section[data-testid="stSidebar"] div[role="radiogroup"] {{gap: 2px; width: 100% !important; max-width: 100%;}}
section[data-testid="stSidebar"] div[role="radiogroup"] > div {{width: 100%;}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{width: 100%; padding: .55rem .8rem; border-radius: 8px;
     margin: 0; cursor: pointer;}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {{display: none;}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {{background: rgba(255,255,255,.05);}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {{background: rgba(255,255,255,.09);
     box-shadow: inset 3px 0 0 {MARIGOLD};}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p {{color: #FFFFFF; font-weight: 600;}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{font-size: .95rem;}}
section[data-testid="stSidebar"] button[kind="secondary"] {{background: transparent !important; border: 1px solid rgba(255,255,255,.22) !important;}}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {{border-color: {MARIGOLD} !important;}}
section[data-testid="stSidebar"] button[kind="secondary"] p {{color: #E7ECF3 !important;}}
section[data-testid="stSidebar"] hr {{border-color: rgba(255,255,255,.12);}}
section[data-testid="stSidebar"] .side-note {{color: #8796AD; font-size: .78rem; line-height: 1.45;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ====================================================================================== UI helpers
@st.cache_resource
def get_models():
    return S.load_models()


_PANELS = [0]


def panel(height="content"):
    """White bordered panel (a keyed container so CSS can style it)."""
    _PANELS[0] += 1
    return st.container(border=True, height=height, key=f"panel_{_PANELS[0]}")


def esc(x):
    return html.escape(str(x))


def page_header(title, subtitle=""):
    st.markdown(f'<div class="ph"><h1>{esc(title)}</h1>' + (f"<p>{esc(subtitle)}</p>" if subtitle else "") + "</div>",
                unsafe_allow_html=True)


def ledger(cells):
    """cells: list of (label, value, tone, note_html). tone in {'', 'credit', 'debit'}."""
    parts = "".join(f'<div class="cell"><div class="lbl">{esc(l)}</div><div class="val {t}">{esc(v)}</div>'
                    f'<div class="note">{n}</div></div>' for l, v, t, n in cells)
    st.markdown(f'<div class="ledger" style="grid-template-columns: repeat({len(cells)}, minmax(0, 1fr));">{parts}</div>',
                unsafe_allow_html=True)


def delta(curr, prev, money=True, good_up=True, suffix=""):
    """Small 'vs last month' line, coloured by whether the change is good."""
    d = curr - prev
    if abs(d) < 1e-9:
        return f"no change{suffix}"
    good = (d > 0) == good_up
    txt = (("+" if d > 0 else "-") + S.inr(abs(d))) if money else f"{d * 100:+.1f} pts"
    return f'<span class="{"up" if good else "down"}">{txt}</span>{suffix}'


def section(title, note=""):
    st.markdown(f'<div class="sec">{esc(title)}</div>' + (f'<div class="sec-note">{note}</div>' if note else ""),
                unsafe_allow_html=True)


def pill(text, tone="ok"):
    return f'<span class="pill {tone}">{esc(text)}</span>'


def rows_html(items):
    """items: list of (name, meta, amount_text, extra_html)."""
    return "".join(f'<div class="row"><div style="min-width:0"><div class="name">{esc(n)}</div>'
                   f'<div class="meta">{m}</div></div><div class="amt">{a} {x}</div></div>' for n, m, a, x in items)


def chart(fig, h=330, legend=True):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=10, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#2E3A4E"),
                      legend=dict(orientation="h", y=-0.18, x=0, title=None, font=dict(size=11)), showlegend=legend,
                      hoverlabel=dict(font_family="IBM Plex Sans", bgcolor="#FFFFFF"), title=None)
    fig.update_xaxes(showgrid=False, linecolor=RULE, title=None)
    fig.update_yaxes(gridcolor="#E7EBE4", zeroline=False, title=None)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def empty_panel(title, text, user=None, models=None, info=None, key="empty_demo"):
    with panel():
        st.markdown(f'<div class="empty"><h3>{esc(title)}</h3><p>{esc(text)}</p></div>', unsafe_allow_html=True)
        if user is not None:
            c = st.columns([1, 3])
            if c[0].button("Load demo data", type="primary", key=key):
                load_demo(user, models, info)


def load_demo(user, models, info):
    with st.spinner("Loading and categorising 24 months of demo history..."):
        try:
            n = S.load_demo(user, models, info)
            st.toast(f"Loaded {n:,} demo transactions.")
            st.rerun()
        except Exception as e:
            st.error(str(e))


def latest_month(tx):
    return tx["date"].dt.to_period("M").max() if not tx.empty else pd.Timestamp(date.today()).to_period("M")


# ====================================================================================== pages
def page_dashboard(user, models, info):
    u = S.get_user(user) or {}
    first = (u.get("full_name") or user).split()[0]
    tx = S.load_transactions(user)
    if tx.empty:
        page_header(f"Welcome, {first}", "Your dashboard fills in once there are transactions to analyse.")
        return empty_panel("No transactions yet",
                           "Load 24 months of demo history to explore every page, or add your own on the Transactions "
                           "page (one at a time or as a bank-statement CSV).", user, models, info)
    m = latest_month(tx)
    cur, prev = S.kpis(tx, m), S.kpis(tx, m - 1)
    page_header(f"Welcome back, {first}", f"{m.strftime('%B %Y')} at a glance, compared with {(m - 1).strftime('%B')}.")
    vs = f" vs {(m - 1).strftime('%b')}"
    ledger([("Money in", S.inr(cur["income"]), "credit", delta(cur["income"], prev["income"], suffix=vs)),
            ("Money out", S.inr(cur["expenses"]), "debit", delta(cur["expenses"], prev["expenses"], good_up=False, suffix=vs)),
            ("Saved", S.inr(cur["savings"]), "credit" if cur["savings"] >= 0 else "debit",
             delta(cur["savings"], prev["savings"], suffix=vs)),
            ("Savings rate", f"{cur['rate']:.0%}", "", delta(cur["rate"], prev["rate"], money=False, suffix=vs))])

    left, right = st.columns([1, 1.45], gap="medium")
    with left, panel():
        section("Where the money went", f"Spending by category, {m.strftime('%B')}")
        month_tx = tx[(tx["date"].dt.to_period("M") == m) & (tx["type"] == "debit")]
        by_cat = month_tx.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
        fig = px.pie(by_cat, names="category", values="amount", hole=0.62, color="category", color_discrete_map=CAT_COLORS)
        fig.update_traces(textinfo="none", sort=False, marker=dict(line=dict(color="#FFFFFF", width=2)),
                          hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>")
        fig.add_annotation(text=f"<b>{S.inr(by_cat['amount'].sum())}</b><br><span style='font-size:11px'>spent</span>",
                           showarrow=False, font=dict(family="Bricolage Grotesque", size=18, color=INK))
        chart(fig, 330)
    with right, panel():
        section("Month by month", "Money in, money out and investments over the last 12 months")
        d = tx.assign(month=tx["date"].dt.to_period("M"))
        d["kind"] = d.apply(lambda r: "Money in" if r["type"] == "credit" else
                            ("Investments" if r["category"] in NON_EXPENSE else "Money out"), axis=1)
        months = sorted(d["month"].unique())[-12:]
        trend = d[d["month"].isin(months)].groupby(["month", "kind"])["amount"].sum().reset_index()
        trend["month"] = trend["month"].dt.strftime("%b %y")
        fig = px.bar(trend, x="month", y="amount", color="kind", barmode="group",
                     category_orders={"month": [m_.strftime("%b %y") for m_ in months],
                                      "kind": ["Money in", "Money out", "Investments"]},
                     color_discrete_map={"Money in": CREDIT, "Money out": DEBIT, "Investments": INK})
        fig.update_traces(hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>", marker_line_width=0)
        fig.update_layout(bargap=0.25, bargroupgap=0.08)
        chart(fig, 330)

    a, b, c = st.columns(3, gap="medium")
    with a, panel(height=330):
        section("Top merchants", "All-time spend")
        p = S.patterns(tx)
        if p:
            st.markdown(rows_html([(name.title(), f"{int(r['count'])} payments", S.inr(r["sum"]), "")
                                   for name, r in p["merchants"].head(6).iterrows()]), unsafe_allow_html=True)
    with b, panel(height=330):
        section("Upcoming bills", "Next 30 days")
        subs = S.get_subscriptions(user)
        if subs.empty and "subscription" in models:
            S.detect_subscriptions(user, models, info)
            subs = S.get_subscriptions(user)
        items = []
        if not subs.empty:
            subs["next_due_date"] = pd.to_datetime(subs["next_due_date"])
            up = subs[subs["next_due_date"] <= pd.Timestamp(date.today() + timedelta(days=30))]
            today = pd.Timestamp(date.today())
            items = [(r.merchant.title(),
                      f"Due {r.next_due_date:%d %b}" if r.next_due_date >= today else f"Was due {r.next_due_date:%d %b}",
                      S.inr(r.avg_amount),
                      pill("Autopay", "sim") if r.autopay else (pill("Overdue", "warn") if r.next_due_date < today else ""))
                     for r in up.head(6).itertuples()]
        if items:
            st.markdown(rows_html(items), unsafe_allow_html=True)
        else:
            st.caption("Nothing due. Open Subscriptions to detect recurring payments." if subs.empty else
                       "Nothing due in the next 30 days.")
    with c, panel(height=330):
        per = S.persona(user, models, info)
        section("Your spending persona")
        if per and not per["cold_start"]:
            st.markdown(f'<div class="persona-name">{esc(per["name"])}</div><div class="sec-note">{esc(per["description"])}</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("Available after 3 months of transactions.")
        goals = S.get_goals(user)
        if not goals.empty:
            st.markdown('<div class="sec" style="font-size:1rem;margin-top:.6rem">Goals</div>', unsafe_allow_html=True)
            for g in goals.head(3).itertuples():
                st.progress(min(g.saved_amount / g.target_amount, 1.0),
                            text=f"{g.name}: {S.inr(g.saved_amount)} of {S.inr(g.target_amount)}")


def page_transactions(user, models, info):
    page_header("Transactions", "Every payment is categorised automatically with a confidence score. "
                                "Change any category in the table and it is saved straight away.")
    tab_list, tab_add, tab_csv = st.tabs(["All transactions", "Add a transaction", "Import a CSV"])
    with tab_list:
        tx = S.load_transactions(user)
        if tx.empty:
            empty_panel("No transactions yet", "Add one in the next tab, import a bank statement, or load the demo history.",
                        user, models, info, key="tx_demo")
        else:
            with panel():
                f1, f2, f3, f4 = st.columns([2.2, 2, 2.2, 1.3])
                q = f1.text_input("Search", "", placeholder="Merchant or description")
                dmin, dmax = tx["date"].min().date(), tx["date"].max().date()
                rng = f2.date_input("Dates", (max(dmin, dmax - timedelta(days=90)), dmax), min_value=dmin, max_value=dmax)
                cats = f3.multiselect("Categories", ALL_CATEGORIES, placeholder="All categories")
                f4.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
                only_review = f4.toggle("Needs review")
                amax = float(tx["amount"].max())
                lo, hi = st.slider("Amount (₹)", 0.0, amax, (0.0, amax), format="₹%.0f")
            v = tx.copy()
            if q:
                v = v[v["description"].str.contains(q, case=False, regex=False)]
            if isinstance(rng, tuple) and len(rng) == 2:
                v = v[(v["date"].dt.date >= rng[0]) & (v["date"].dt.date <= rng[1])]
            if cats:
                v = v[v["category"].isin(cats)]
            if only_review:
                v = v[v["needs_review"] == 1]
            v = v[(v["amount"] >= lo) & (v["amount"] <= hi)]
            n_rev = int(tx["needs_review"].sum())
            st.markdown(f'{pill(f"{n_rev} need review", "review")} &nbsp;<span class="sec-note">Showing {len(v):,} of '
                        f'{len(tx):,} transactions</span>', unsafe_allow_html=True)
            show = v[["id", "date", "description", "amount", "type", "category", "confidence", "needs_review"]].copy()
            show["status"] = show["needs_review"].map({1: "Needs review", 0: "OK"})
            show["date"] = show["date"].dt.date
            edited = st.data_editor(
                show.drop(columns=["needs_review"]), hide_index=True, width="stretch", height=480,
                disabled=["id", "date", "description", "amount", "type", "confidence", "status"],
                column_config={
                    "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                    "description": st.column_config.TextColumn("Description", width="medium"),
                    "category": st.column_config.SelectboxColumn("Category", options=ALL_CATEGORIES, required=True),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent",
                                                                  width="small"),
                    "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                    "type": st.column_config.TextColumn("Type", width="small"),
                    "status": st.column_config.TextColumn("Status", width="small"), "id": None}, key=f"editor_{user}")
            changed = edited[edited["category"].values != show["category"].values]
            if len(changed):
                for r in changed.itertuples():
                    S.update_category(r.id, r.category)
                st.toast(f"Saved {len(changed)} category change(s).")
                st.rerun()
    with tab_add, panel():
        if "categorizer" not in models:
            st.warning("The categorizer is not trained yet. Run `python -m src.train`.")
        desc = st.text_input("Description", placeholder="As it appears on your statement, e.g. UPI/SWIGGY/123456/Payment")
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Amount (₹)", min_value=0.0, value=250.0, step=50.0)
        when = c2.date_input("Date", date.today())
        ttype = c3.selectbox("Type", ["debit", "credit"], format_func=lambda t: "Money out" if t == "debit" else "Money in")
        if desc and "categorizer" in models:
            cat, conf, review = S.suggest_category(desc, amt, when, ttype, models, info)
            st.markdown(f'Suggested category: <b>{esc(cat)}</b> &nbsp;'
                        f'{pill(f"{conf:.0%} confident" + (", please check" if review else ""), "review" if review else "ok")}',
                        unsafe_allow_html=True)
        else:
            cat, conf, review = "Others", 0.0, True
        final = st.selectbox("Category", ALL_CATEGORIES, index=ALL_CATEGORIES.index(cat))
        if st.button("Add transaction", type="primary", disabled=not desc):
            row = pd.DataFrame([dict(date=when, description=desc, amount=amt, type=ttype, category=final,
                                     predicted_category=cat, confidence=conf, needs_review=int(review and final == cat))])
            S.insert_transactions(user, row, "manual")
            st.toast("Transaction added.")
    with tab_csv, panel():
        st.markdown('<div class="sec-note">Use a CSV with columns <code>date, description, amount</code> (optional '
                    '<code>type</code>), or your bank\'s export with <code>Date, Narration, Withdrawal Amt., Deposit Amt.</code>. '
                    'Every row is categorised automatically.</div>', unsafe_allow_html=True)
        file = st.file_uploader("Statement CSV", type=["csv"])
        if file and st.button("Import transactions", type="primary"):
            with st.spinner("Categorising..."):
                try:
                    res = S.import_csv(file, user, models, info)
                    st.success(f"Imported {res['imported']:,} of {res['rows_in']:,} rows. {res['needs_review']} need review.")
                    s = pd.Series(res["by_category"]).sort_values()
                    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                                           marker_color=[CAT_COLORS.get(c, MUTED) for c in s.index]))
                    chart(fig, 40 + 26 * len(s), legend=False)
                except Exception as e:
                    st.error(str(e))


def page_subscriptions(user, models, info):
    page_header("Subscriptions", "Recurring payments found in your history. Autopay here is a simulation: "
                                 "it records the payment in the app, and no money moves.")
    for m, d, a in S.run_autopay(user):
        st.toast(f"Autopay recorded: {m} {S.inr(a)} on {d:%d %b}")
    with st.spinner("Looking for recurring payments..."):
        S.detect_subscriptions(user, models, info)
    subs = S.get_subscriptions(user)
    if subs.empty:
        return empty_panel("No recurring payments found",
                           "Recurring payments show up after at least two payments to the same merchant.", user, models,
                           info, key="sub_demo")
    monthly = (subs["avg_amount"] * 30.4 / subs["frequency_days"].clip(lower=1)).sum()
    ledger([("Recurring payments", f"{len(subs)}", "", f"{int((subs['autopay'] == 1).sum())} on autopay"),
            ("Per month", S.inr(monthly), "debit", "Average across all cycles"),
            ("Per year", S.inr(subs["yearly_cost"].sum()), "debit", "If nothing changes")])
    with panel():
        widths = [2.3, 1.1, 1.0, 1.4, 1.1, 0.8, 0.8, 2.4]
        hdr = st.columns(widths)
        for h, t in zip(hdr, ["Merchant", "Amount", "Every", "Next due", "Per year", "Match", "Autopay", "Is this recurring?"]):
            h.markdown(f'<span class="sec-note">{t}</span>', unsafe_allow_html=True)
        for r in subs.itertuples():
            col = st.columns(widths, vertical_alignment="center")
            col[0].markdown(f'<div class="name" style="font-weight:600;color:{INK}">{esc(r.merchant.title())}</div>'
                            f'<div class="sec-note" style="margin:0">{esc(r.category)}</div>', unsafe_allow_html=True)
            col[1].markdown(f"**{S.inr(r.avg_amount)}**")
            col[2].write(f"{r.frequency_days:.0f} days")
            due = pd.Timestamp(r.next_due_date)
            col[3].markdown(f"{due:%d %b}" + (" &nbsp;" + pill("Overdue", "warn") if due < pd.Timestamp(date.today()) else ""),
                            unsafe_allow_html=True)
            col[4].write(S.inr(r.yearly_cost))
            col[5].write(f"{r.confidence:.0%}")
            ap = col[6].toggle("Autopay", value=bool(r.autopay), key=f"ap_{r.id}", label_visibility="collapsed")
            if ap != bool(r.autopay):
                S.set_subscription(r.id, autopay=int(ap))
                st.toast(f"Autopay {'turned on' if ap else 'turned off'} for {r.merchant.title()}.")
                st.rerun()
            if r.status == "confirmed":
                col[7].markdown(pill("Confirmed", "ok"), unsafe_allow_html=True)
            else:
                b1, b2 = col[7].columns(2)
                if b1.button("Confirm", key=f"c_{r.id}", help="Yes, this is a recurring payment"):
                    S.set_subscription(r.id, status="confirmed"); st.rerun()
                if b2.button("Dismiss", key=f"d_{r.id}", help="Not a subscription"):
                    S.set_subscription(r.id, status="dismissed", autopay=0); st.rerun()


def page_forecast(user, models, info):
    page_header("Forecast and budget", "Next month's spending per category, predicted from your own history.")
    fc = S.forecast_next_month(user, models, info)
    if fc is None:
        return empty_panel("Not enough history yet", "Forecasts need at least 3 complete months of transactions.",
                           user, models, info, key="fc_demo")
    tm = pd.Period(fc.attrs["target_month"]).strftime("%B %Y")
    sv = S.savings_forecast(user, fc)
    ledger([("Expected spend", S.inr(sv["spend"]), "debit", f"{tm}, all categories"),
            ("Expected income", S.inr(sv["income"]), "credit", "Average of the last 3 months"),
            ("Expected savings", S.inr(sv["savings"]), "credit" if sv["savings"] >= 0 else "debit", "Income minus spend"),
            ("Planned investments", S.inr(sv["investments"]), "", "SIPs and deposits, counted as savings")])
    t1, t2 = st.tabs(["Forecast", "Budget"])
    with t1, panel():
        section(f"Predicted spend for {tm}", "Bars show the forecast; the whiskers show the likely range "
                                             "(10th to 90th percentile). The tick marks your last 3-month average.")
        f = fc[fc["forecast"] > 0].sort_values("forecast")
        fig = go.Figure(go.Bar(y=f["category"], x=f["forecast"], orientation="h",
                               marker_color=[CAT_COLORS.get(c, MUTED) for c in f["category"]],
                               error_x=dict(type="data", symmetric=False, array=f["high"] - f["forecast"],
                                            arrayminus=f["forecast"] - f["low"], color="#5B6576", thickness=1.2, width=4),
                               name="Forecast", hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
        fig.add_scatter(y=f["category"], x=f["avg_3m"], mode="markers", name="Last 3-month average",
                        marker=dict(color=INK, symbol="line-ns-open", size=16, line=dict(width=2.5)),
                        hovertemplate="%{y}, 3-month average: ₹%{x:,.0f}<extra></extra>")
        fig.update_xaxes(showgrid=True, gridcolor="#E7EBE4", tickprefix="₹")
        fig.update_yaxes(showgrid=False)
        chart(fig, 60 + 30 * len(f))
        t = fc.rename(columns={"category": "Category", "forecast": "Forecast", "low": "Low", "high": "High",
                               "last_month": "Last month", "avg_3m": "3-month average"})
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(c, format="₹%.0f") for c in ["Forecast", "Low", "High", "Last month", "3-month average"]})
    with t2, panel():
        rb = S.recommend_budget(user, fc)
        section("Recommended budget", "Essentials get a 5% buffer. Discretionary spending is reduced only when your "
                                      "goals need more savings than the forecast leaves.")
        if rb.attrs["goal_need"] > 0:
            msg = (f"Your goals need about {S.inr(rb.attrs['goal_need'])} a month. "
                   + (f"Dining, shopping, entertainment and travel are cut by {rb.attrs['cut']:.0%} to close a "
                      f"{S.inr(rb.attrs['shortfall'])} gap." if rb.attrs["cut"] > 0 else "Your forecast savings already cover that."))
            st.info(msg)
        saved = S.get_budget(user)
        rb["my_budget"] = rb["category"].map(lambda c: saved.get(c)).fillna(rb["recommended"])
        ed = st.data_editor(rb, hide_index=True, width="stretch", disabled=["category", "forecast", "recommended", "reason"],
                            column_config={"category": "Category", "reason": st.column_config.TextColumn("Why", width="large"),
                                           "my_budget": st.column_config.NumberColumn("My budget", min_value=0, step=500, format="₹%d"),
                                           "forecast": st.column_config.NumberColumn("Forecast", format="₹%d"),
                                           "recommended": st.column_config.NumberColumn("Recommended", format="₹%d")})
        if st.button("Save budget", type="primary"):
            S.save_budget(user, dict(zip(ed["category"], ed["my_budget"].astype(float))))
            st.toast("Budget saved.")
    with t2, panel():
        section("This month so far", "Spending against your budget")
        tx = S.load_transactions(user)
        cur = tx[(tx["date"].dt.to_period("M") == pd.Timestamp(date.today()).to_period("M")) & (tx["type"] == "debit")]
        spent = cur.groupby("category")["amount"].sum()
        cols = st.columns(2, gap="large")
        items = [(c, b) for c, b in sorted(dict(zip(ed["category"], ed["my_budget"])).items(), key=lambda x: -x[1]) if b > 0]
        for i, (cat, b) in enumerate(items):
            s = float(spent.get(cat, 0))
            cols[i % 2].progress(min(s / b, 1.0), text=f"{cat}: {S.inr(s)} of {S.inr(b)}" + (" (over budget)" if s > b else ""))


def page_goals(user, models, info):
    page_header("Goals", "Set a target and see when you will reach it at your forecast savings rate.")
    goals = S.get_goals(user)
    with st.expander("Create a goal", expanded=goals.empty):
        with st.form("goal"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Goal", "Emergency fund")
            target = c2.number_input("Target amount (₹)", min_value=1000.0, value=100000.0, step=5000.0, format="%.0f")
            saved = c1.number_input("Saved so far (₹)", min_value=0.0, value=0.0, step=1000.0, format="%.0f")
            when = c2.date_input("Target date", date.today() + timedelta(days=365))
            if st.form_submit_button("Create goal", type="primary"):
                S.add_goal(user, name, target, saved, when)
                st.toast("Goal created.")
                st.rerun()
    if goals.empty:
        return
    fc = S.forecast_next_month(user, models, info)
    monthly = S.savings_forecast(user, fc)["savings"] if fc is not None else 0
    n = max(len(goals), 1)
    st.markdown(f'<div class="sec-note">Forecast savings next month: <b>{S.inr(monthly)}</b>, split evenly across '
                f'{n} goal{"s" if n != 1 else ""} for the estimates below.</div>', unsafe_allow_html=True)
    cols = st.columns(2, gap="medium")
    for i, g in enumerate(goals.itertuples()):
        with cols[i % 2], panel():
            section(g.name, f"Target {pd.Timestamp(g.target_date):%d %b %Y}")
            pct = min(g.saved_amount / g.target_amount, 1.0)
            st.markdown(f'<div class="persona-name" style="font-size:1.5rem">{S.inr(g.saved_amount)} '
                        f'<span class="sec-note" style="font-family:IBM Plex Sans">of {S.inr(g.target_amount)} ({pct:.0%})</span></div>',
                        unsafe_allow_html=True)
            st.progress(pct)
            msg, ok = S.goal_feasibility(g.target_amount, g.saved_amount, monthly / n, g.target_date)
            (st.success if ok else st.warning)(msg)
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            new = c1.number_input("Saved so far (₹)", value=float(g.saved_amount), step=1000.0, format="%.0f", key=f"g_{g.id}")
            if new != g.saved_amount:
                S.update_goal_saved(g.id, new); st.rerun()
            if c2.button("Delete goal", key=f"del_{g.id}"):
                S.delete_goal(g.id); st.rerun()


def page_insights(user, models, info):
    page_header("Insights", "Your spending persona, weekly patterns and anything unusual.")
    tx = S.load_transactions(user)
    if tx.empty:
        return empty_panel("No data yet", "Insights appear once there are transactions.", user, models, info, key="in_demo")
    per = S.persona(user, models, info)
    left, right = st.columns([1, 1.9], gap="medium")
    with left, panel(height=600):
        section("Spending persona")
        if per is None:
            st.caption("The persona model is not trained yet.")
        elif per["cold_start"]:
            st.caption(f"Personas need at least 3 months of data. You have {per['n_months']}.")
        else:
            st.markdown(f'<div class="persona-name">{esc(per["name"])}</div><div class="sec-note">{esc(per["description"])}</div>'
                        f'<div class="persona-tip">{esc(per["tip"])}</div>', unsafe_allow_html=True)
            u_, a_ = per["user"], per["avg"]
            st.markdown('<div class="sec" style="font-size:1rem;margin-top:1.4rem">What sets you apart</div>'
                        '<div class="sec-note">You, compared with a typical member of this persona</div>' + rows_html([
                            ("Savings rate", f"Typical: {a_['savings_rate']:.0%}", f"{u_['savings_rate']:.0%}", ""),
                            ("Average payment", f"Typical: {S.inr(a_['avg_txn_size'])}", S.inr(u_["avg_txn_size"]), ""),
                            ("Payments per month", f"Typical: {a_['txns_per_month']:.0f}", f"{u_['txns_per_month']:.0f}", ""),
                            ("Month-to-month swing", f"Typical: {a_['monthly_cv']:.0%}", f"{u_['monthly_cv']:.0%}", "")]),
                        unsafe_allow_html=True)
    with right, panel(height=600):
        section("You compared with your persona", "Share of spending per category")
        if per and not per["cold_start"]:
            shares = pd.DataFrame({"category": EXPENSE_CATEGORIES,
                                   "You": [per["user"][f"share_{c}"] for c in EXPENSE_CATEGORIES],
                                   f"Typical {per['name']}": [per["avg"][f"share_{c}"] for c in EXPENSE_CATEGORIES]})
            shares = shares[(shares.iloc[:, 1:] > 0.005).any(axis=1)].sort_values("You")
            long = shares.melt("category", var_name="who", value_name="share")
            fig = px.bar(long, y="category", x="share", color="who", barmode="group", orientation="h",
                         color_discrete_sequence=[MARIGOLD, "#B9C2CF"])
            fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor="#E7EBE4")
            fig.update_yaxes(showgrid=False)
            fig.update_traces(hovertemplate="%{y}: %{x:.1%}<extra></extra>")
            fig.update_layout(bargap=0.3, bargroupgap=0.05)
            chart(fig, 520)
    p = S.patterns(tx)
    if not p:
        return
    subs = S.get_subscriptions(user)
    wt = S.weekend_test(tx)
    ratio = p["weekend_daily"] / max(p["weekday_daily"], 1) - 1
    ledger([("Average weekend day", S.inr(p["weekend_daily"]), "", f"{ratio:+.0%} compared with weekdays"),
            ("Average weekday", S.inr(p["weekday_daily"]), "", "Monday to Friday"),
            ("Recurring share of spend", f"{S.recurring_share(tx, subs):.0%}", "", "Payments to detected subscriptions"),
            ("Weekend test", "Different" if wt and wt["significant"] else "No difference", "",
             f"Mann-Whitney U, p = {wt['p_value']:.3f}" if wt else "Not enough days")])
    a, b = st.columns(2, gap="medium")
    with a, panel():
        section("Average spend by weekday")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wk = p["by_weekday"].reindex(order).fillna(0)
        fig = go.Figure(go.Bar(x=[d[:3] for d in order], y=wk.values,
                               marker_color=[INK if i < 5 else MARIGOLD for i in range(7)],
                               hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
        chart(fig, 280, legend=False)
    with b, panel():
        section("Fastest-growing categories", "Last 3 months compared with the 3 months before")
        if len(p["growth"]):
            g = p["growth"].head(6).sort_values()
            fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h",
                                   marker_color=[DEBIT if v > 0 else CREDIT for v in g.values],
                                   hovertemplate="%{y}: %{x:+.0%}<extra></extra>"))
            fig.update_xaxes(tickformat="+.0%")
            chart(fig, 280, legend=False)
        else:
            st.caption("Needs 6 months of history.")
    with panel():
        section("Unusual transactions", "Last 3 months: payments far above your usual amount for that category "
                                        "(above Q3 + 3 × IQR of your own history)")
        un = S.unusual_transactions(tx)
        if un.empty:
            st.caption("Nothing unusual.")
        else:
            st.dataframe(un.assign(date=un["date"].dt.date), hide_index=True, width="stretch", column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"), "description": "Description",
                "category": "Category", "amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
                "times_typical": st.column_config.NumberColumn("Times your usual", format="%.1f×")})


def page_admin(user, models, info):
    page_header("Models", "Active model versions and how they scored on held-out test data in the last training run.")
    if not info:
        return empty_panel("No models registered", "Run python -m src.data_prep and python -m src.train, then reload.")
    fb = S.feedback_stats()
    ledger([("Categorised transactions", f"{fb['n']:,}", "", "All users of this app"),
            ("Corrected by users", f"{fb['correction_rate']:.1%}", "", "Saved for future retraining"),
            ("Flagged for review", f"{fb['review_rate']:.1%}", "", "Below the confidence threshold"),
            ("Average confidence", f"{fb['avg_confidence']:.2f}", "", "Categorizer")])
    st.markdown(f'<div class="sec-note">Metrics are measured on synthetic and simulated data, not real-world performance. '
                f'Experiment history: <a href="{MLFLOW_URL}" target="_blank">MLflow</a> '
                f'(start it with <code>python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db</code>).</div>',
                unsafe_allow_html=True)
    figs = {"categorizer": ["categorizer_confusion_matrix.png", "categorizer_per_class_f1.png", "categorizer_confidence_curve.png"],
            "subscription": ["subscription_confusion_matrix.png", "subscription_roc_pr.png", "subscription_shap.png"],
            "forecaster": ["forecaster_baseline_vs_model.png", "forecaster_pred_vs_actual.png", "forecaster_residuals.png"],
            "persona": ["persona_pca.png", "persona_k_selection.png", "persona_centroid_heatmap.png"]}
    names = {"categorizer": "Categorizer", "subscription": "Subscription detector", "forecaster": "Forecaster",
             "persona": "Persona clustering"}
    tabs = st.tabs([names.get(n, n) for n in info])
    for tab, (name, e) in zip(tabs, info.items()):
        with tab:
            ledger([("Version", f"v{e['version']}", "", esc(e["file"])), ("Algorithm", e["algorithm"], "", ""),
                    ("Primary metric", e["primary_metric"], "", ""), ("Trained", e["trained_at"][:10], "",
                                                                     f"git {esc(e.get('git_commit', '?'))}")])
            with panel():
                section("Test metrics against the baseline")
                mt = pd.DataFrame({"Model": e["test_metrics"], "Baseline": e.get("baseline_metrics", {})})
                mt.index = [i.replace("_", " ").capitalize() for i in mt.index]
                st.dataframe(mt.map(lambda v: "-" if v is None or (isinstance(v, float) and pd.isna(v))
                                    else (f"{v:.4f}" if isinstance(v, (int, float)) else str(v))), width="stretch")
            comp = RESULTS / f"model_comparison_{name}.csv"
            if comp.exists():
                with panel():
                    section("All candidates")
                    st.dataframe(pd.read_csv(comp).round(4), hide_index=True, width="stretch")
            cols = st.columns(3, gap="medium")
            for col, f in zip(cols, figs.get(name, [])):
                if (RESULTS / f).exists():
                    with col, panel():
                        st.image(str(RESULTS / f), width="stretch")


# ====================================================================================== authentication
AUTH_CSS = f"""
<style>
section[data-testid="stSidebar"], div[data-testid="collapsedControl"], button[data-testid="stExpandSidebarButton"] {{display: none;}}
.block-container {{max-width: 1180px; padding-top: 2.4rem;}}
.auth-hero {{background: {INK}; border-radius: 20px; padding: 2.4rem 2.4rem 2rem; color: #DCE3EE; min-height: 600px;
             display: flex; flex-direction: column;}}
.auth-hero .brand {{display: flex; align-items: center; gap: .6rem; margin-bottom: 2.6rem;}}
.auth-hero .mark {{width: 32px; height: 32px; border-radius: 8px; background: {MARIGOLD}; display: grid; place-items: center;
                   color: {INK}; font-family: "Bricolage Grotesque"; font-weight: 800;}}
.auth-hero .word {{font-family: "Bricolage Grotesque"; font-weight: 700; font-size: 1.15rem; color: #FFFFFF;}}
.auth-hero h1 {{font-family: "Bricolage Grotesque"; color: #FFFFFF; font-size: 2.9rem; line-height: 1.02; font-weight: 750;
                margin: 0 0 .9rem 0; letter-spacing: -.025em; padding: 0;}}
.auth-hero .lead {{color: #B8C4D6; font-size: 1.04rem; line-height: 1.6; max-width: 46ch; margin-bottom: 1.8rem;}}
.passbook {{background: #FBFCFA; border-radius: 12px; padding: .5rem 1.1rem; color: {INK}; margin-top: auto;}}
.passbook .pb-row {{display: grid; grid-template-columns: 1fr auto; gap: .2rem 1rem; height: 52px; align-content: center;}}
.passbook .nar {{font-family: "IBM Plex Sans"; font-size: .84rem; color: #3A4659; white-space: nowrap; overflow: hidden;
                 text-overflow: ellipsis; font-variant-numeric: tabular-nums;}}
.passbook .cat {{font-size: .84rem; font-weight: 600; text-align: right; white-space: nowrap;}}
.passbook .bar {{grid-column: 1 / -1; height: 3px; background: #E3E8E0; border-radius: 2px; overflow: hidden;}}
.passbook .bar i {{display: block; height: 100%;}}
.auth-hero .cap {{color: #8796AD; font-size: .8rem; margin-top: .7rem;}}
.auth-side {{padding: 3.2rem .4rem 0 .6rem;}}
.auth-side h2 {{font-family: "Bricolage Grotesque"; font-size: 1.9rem; color: {INK}; margin: 0 0 .3rem 0; padding: 0;}}
.auth-side p {{color: {MUTED}; margin: 0 0 1.2rem 0;}}
div[data-testid="stForm"] {{background: #FFFFFF; border: 1px solid {RULE}; border-radius: 14px; padding: 1.3rem 1.3rem .5rem;}}
.auth-foot {{color: #8A93A2; font-size: .78rem; margin-top: 1rem; line-height: 1.5;}}
</style>"""

PREVIEW = ["UPI/SWIGGY/402291/Payment", "ACH D- HDFC BANK EMI 88213004", "NEFT-HOUSE RENT-NOV", "POS 4471 DMART PUNE",
           "UPI/RAHUL SHARMA/551204/paid"]


def passbook_preview(models, info):
    """Live demo of the categorizer on statement-style narrations."""
    rows = []
    for d in PREVIEW:
        if "categorizer" in models:
            cat, conf, review = S.suggest_category(d, 500, date.today(), "debit", models, info)
        else:
            cat, conf, review = "-", 0.0, False
        color = MARIGOLD if review else CREDIT
        label = f"Needs review ({conf:.0%})" if review else f"{cat} ({conf:.0%})"
        rows.append(f'<div class="pb-row"><div class="nar">{esc(d)}</div><div class="cat" style="color:{"#9A6410" if review else CREDIT}">'
                    f'{esc(label)}</div><div class="bar"><i style="width:{conf * 100:.0f}%;background:{color}"></i></div></div>')
    return "".join(rows)


def sign_in(username):
    st.session_state["user"] = username
    st.session_state.pop("page", None)


def page_auth(models, info):
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    left, right = st.columns([1.12, 1], gap="large")
    with left:
        st.markdown(f"""
<div class="auth-hero">
  <div class="brand"><div class="mark">S</div><div class="word">Smart Finance Tracker</div></div>
  <h1>Your bank statement, sorted.</h1>
  <div class="lead">Import UPI, NEFT and card narrations and every payment gets a category. Recurring bills are
  spotted for you, and next month's spending is forecast from your own history.</div>
  <div class="passbook">{passbook_preview(models, info)}</div>
  <div class="cap">Live output of the categorization model on sample narrations. Low-confidence guesses are flagged
  for review instead of being filed silently.</div>
</div>""", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="auth-side"><h2>Welcome</h2><p>Sign in, or create an account to start tracking.</p></div>',
                    unsafe_allow_html=True)
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("signin_form", clear_on_submit=False):
                username = st.text_input("Username", key="si_username", autocomplete="username")
                password = st.text_input("Password", type="password", key="si_password", autocomplete="current-password")
                submitted = st.form_submit_button("Sign in", type="primary", width="stretch", key="si_submit")
            if submitted:
                ok, msg = S.authenticate(username, password)
                if ok:
                    sign_in(username.strip().lower())
                    st.rerun()
                st.error(msg)
        with tab_up:
            with st.form("signup_form", clear_on_submit=False):
                c1, c2 = st.columns(2)
                full_name = c1.text_input("Full name", key="su_name", autocomplete="name")
                username = c2.text_input("Username", key="su_username", autocomplete="username",
                                         help="3-20 characters: lowercase letters, numbers, '.' or '_'")
                income = st.number_input("Monthly income (₹)", min_value=0.0, value=50000.0, step=5000.0, format="%.0f", key="su_income")
                c3, c4 = st.columns(2)
                password = c3.text_input("Password", type="password", key="su_password", autocomplete="new-password",
                                         help=f"At least {S.MIN_PASSWORD} characters, with a letter and a number")
                confirm = c4.text_input("Confirm password", type="password", key="su_confirm", autocomplete="new-password")
                submitted = st.form_submit_button("Create account", type="primary", width="stretch", key="su_submit")
            if submitted:
                username = username.strip().lower()
                problems = S.validate_signup(username, full_name, password, confirm)
                if problems:
                    for p in problems:
                        st.error(p)
                else:
                    S.register_user(username, full_name, password, income)
                    sign_in(username)
                    st.rerun()
        st.markdown('<div class="auth-foot">Passwords are stored as salted PBKDF2-SHA256 hashes in a local SQLite file. '
                    'This is a student project, so avoid uploading real statements on a shared computer.</div>',
                    unsafe_allow_html=True)


# ====================================================================================== top navigation / routing
NAV_CSS = f"""
<style>
section[data-testid="stSidebar"], button[data-testid="stExpandSidebarButton"] {{display: none;}}
header[data-testid="stHeader"] {{background: rgba(255,255,255,.92) !important; backdrop-filter: saturate(160%) blur(8px);
     border-bottom: 1px solid {RULE}; height: 3.6rem;}}
header[data-testid="stHeader"] a, header[data-testid="stHeader"] button p {{font-weight: 500;}}
header a[aria-current="page"], header a[aria-current="page"]:hover {{background: transparent !important; color: {INK} !important;
     box-shadow: inset 0 -2px 0 {MARIGOLD}; border-radius: 0 !important;}}
header a[aria-current="page"] p, header a[aria-current="page"] span {{font-weight: 650 !important; color: {INK} !important;}}
header a {{color: #3D4A5E !important;}}
.block-container {{padding-top: 4.6rem;}}
div[class*="st-key-usermenu"] {{position: fixed; top: .55rem; right: 1.4rem; z-index: 999991; width: auto !important;}}
div[class*="st-key-usermenu"] button {{border-radius: 999px; border: 1px solid {RULE}; background: #FFFFFF;}}
</style>"""

PAGES = [("Dashboard", "dashboard", page_dashboard), ("Transactions", "transactions", page_transactions),
         ("Subscriptions", "subscriptions", page_subscriptions), ("Forecast", "forecast", page_forecast),
         ("Goals", "goals", page_goals), ("Insights", "insights", page_insights), ("Models", "models", page_admin)]


def user_menu(username, user, models, info):
    with st.container(key="usermenu"):
        name = user.get("full_name") or username
        with st.popover(name, width="content"):
            st.markdown(f'<div class="sec" style="font-size:1rem">{esc(name)}</div><div class="sec-note">@{esc(username)}</div>',
                        unsafe_allow_html=True)
            if st.button("Load demo data", width="stretch", help="Replaces your data with a generated 24-month history"):
                load_demo(username, models, info)
            if st.button("Sign out", width="stretch", type="primary"):
                st.session_state.clear()
                st.rerun()


def main():
    _PANELS[0] = 0
    S.init_db()
    models, info = get_models()
    st.logo("static/logo.svg", size="large")
    username = st.session_state.get("user")
    user = S.get_user(username) if username else None
    if not user:
        st.session_state.pop("user", None)
        return page_auth(models, info)
    st.markdown(NAV_CSS, unsafe_allow_html=True)
    pages = [st.Page(lambda f=f: f(username, models, info), title=t, url_path=u, default=(i == 0))
             for i, (t, u, f) in enumerate(PAGES)]
    nav = st.navigation(pages, position="top")
    user_menu(username, user, models, info)
    if not models:
        st.warning("Models not found. Run python -m src.data_prep and python -m src.train.")
    nav.run()


main()
