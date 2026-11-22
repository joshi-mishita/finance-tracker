"""Smart Finance Tracker - Streamlit app.  Run: streamlit run app.py"""
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import services as S
from src.data_prep import ALL_CATEGORIES, EXPENSE_CATEGORIES, NON_EXPENSE

st.set_page_config(page_title="Smart Finance Tracker", page_icon="💰", layout="wide")
RESULTS = Path(__file__).parent / "results"
MLFLOW_URL = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000")
PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
CAT_COLORS = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(ALL_CATEGORIES)}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.app-header {background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%); color: white; padding: 18px 24px;
             border-radius: 16px; margin-bottom: 18px; box-shadow: 0 6px 18px rgba(79,70,229,.25);}
.app-header h1 {color: white; font-size: 1.6rem; margin: 0;}
.app-header p {margin: 2px 0 0 0; opacity: .85;}
div[data-testid="stMetric"] {background: #FFFFFF; border-radius: 14px; padding: 14px 16px;
             box-shadow: 0 2px 10px rgba(17,24,39,.06); border: 1px solid #EEF0F6;}
.card {background: #FFFFFF; border-radius: 14px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(17,24,39,.06);
       border: 1px solid #EEF0F6; margin-bottom: 12px;}
.badge {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;}
.badge-review {background: #FEF3C7; color: #92400E;} .badge-ok {background: #D1FAE5; color: #065F46;}
.badge-sim {background: #E0E7FF; color: #3730A3;}
.persona {font-size: 1.3rem; font-weight: 700; color: #4F46E5;}
.muted {color: #6B7280; font-size: .9rem;}
section[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #EEF0F6;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return S.load_models()


def header(title, subtitle=""):
    st.markdown(f'<div class="app-header"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def empty_state(msg):
    st.info(f"🗂️ {msg}")


def fig_style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.15))
    return fig


def latest_month(tx):
    return tx["date"].dt.to_period("M").max() if not tx.empty else pd.Timestamp(date.today()).to_period("M")


# ====================================================================================== pages
def page_dashboard(user, models, info):
    header("Dashboard", f"Welcome back, {user}")
    tx = S.load_transactions(user)
    if tx.empty:
        return empty_state("No transactions yet. Add some on the Transactions page or click **Load demo data**.")
    m = latest_month(tx)
    cur, prev = S.kpis(tx, m), S.kpis(tx, m - 1)
    st.caption(f"Showing **{m.strftime('%B %Y')}** (latest month with data) vs {(m - 1).strftime('%B %Y')}")
    c = st.columns(4)
    c[0].metric("Income", S.inr(cur["income"]), S.inr(cur["income"] - prev["income"]))
    c[1].metric("Expenses", S.inr(cur["expenses"]), S.inr(cur["expenses"] - prev["expenses"]), delta_color="inverse")
    c[2].metric("Net savings", S.inr(cur["savings"]), S.inr(cur["savings"] - prev["savings"]))
    c[3].metric("Savings rate", f"{cur['rate']:.0%}", f"{(cur['rate'] - prev['rate']) * 100:+.1f} pts")

    left, right = st.columns([1, 1.4])
    month_tx = tx[(tx["date"].dt.to_period("M") == m) & (tx["type"] == "debit")]
    by_cat = month_tx.groupby("category")["amount"].sum().reset_index()
    fig = px.pie(by_cat, names="category", values="amount", hole=0.55, color="category", color_discrete_map=CAT_COLORS,
                 title="Spending by category")
    left.plotly_chart(fig_style(fig), width="stretch")
    d = tx.assign(month=tx["date"].dt.to_period("M").astype(str))
    d["kind"] = d.apply(lambda r: "Income" if r["type"] == "credit" else ("Investments" if r["category"] in NON_EXPENSE
                                                                          else "Expenses"), axis=1)
    trend = d.groupby(["month", "kind"])["amount"].sum().reset_index().tail(36)
    fig = px.bar(trend, x="month", y="amount", color="kind", barmode="group", title="Monthly trend",
                 color_discrete_map={"Income": "#10B981", "Expenses": "#EF4444", "Investments": "#4F46E5"})
    right.plotly_chart(fig_style(fig), width="stretch")

    a, b, c3 = st.columns(3)
    with a:
        st.markdown("**🏪 Top merchants (all time)**")
        p = S.patterns(tx)
        if p:
            mt = p["merchants"].reset_index().rename(columns={"sum": "spent", "count": "txns"})
            mt["spent"] = mt["spent"].map(S.inr)
            st.dataframe(mt, hide_index=True, width="stretch")
    with b:
        st.markdown("**📅 Upcoming bills (next 30 days)**")
        subs = S.get_subscriptions(user)
        if not subs.empty:
            subs["next_due_date"] = pd.to_datetime(subs["next_due_date"])
            up = subs[subs["next_due_date"] <= pd.Timestamp(date.today() + timedelta(days=30))]
            for r in up.head(6).itertuples():
                tag = '<span class="badge badge-sim">Autopay (simulated)</span>' if r.autopay else ""
                st.markdown(f"{r.merchant} — **{S.inr(r.avg_amount)}** on {r.next_due_date:%d %b} {tag}",
                            unsafe_allow_html=True)
            if up.empty:
                st.caption("Nothing due in the next 30 days.")
        else:
            st.caption("Open the Subscriptions page to detect recurring payments.")
    with c3:
        st.markdown("**🎯 Goals**")
        goals = S.get_goals(user)
        for g in goals.itertuples():
            st.progress(min(g.saved_amount / g.target_amount, 1.0), text=f"{g.name}: {S.inr(g.saved_amount)} / {S.inr(g.target_amount)}")
        if goals.empty:
            st.caption("No goals yet.")
        per = S.persona(user, models, info)
        if per and not per["cold_start"]:
            st.markdown(f'<div class="card">Your persona<br><span class="persona">{per["name"]}</span></div>',
                        unsafe_allow_html=True)


def page_transactions(user, models, info):
    header("Transactions", "Auto-categorised with a confidence score - fix anything that looks wrong")
    tab_list, tab_add, tab_csv = st.tabs(["📋 All transactions", "➕ Add transaction", "📤 Upload CSV"])
    with tab_list:
        tx = S.load_transactions(user)
        if tx.empty:
            empty_state("No transactions yet.")
        else:
            f1, f2, f3, f4 = st.columns([2, 2, 2, 1.4])
            q = f1.text_input("Search description", "")
            dmin, dmax = tx["date"].min().date(), tx["date"].max().date()
            rng = f2.date_input("Date range", (max(dmin, dmax - timedelta(days=90)), dmax), min_value=dmin, max_value=dmax)
            cats = f3.multiselect("Category", ALL_CATEGORIES)
            only_review = f4.toggle("Needs review only")
            amax = float(tx["amount"].max())
            lo, hi = st.slider("Amount (₹)", 0.0, amax, (0.0, amax))
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
            st.markdown(f'<span class="badge badge-review">⚠ {n_rev} need review</span> &nbsp; '
                        f'<span class="muted">{len(v)} shown of {len(tx)}</span>', unsafe_allow_html=True)
            show = v[["id", "date", "description", "amount", "type", "category", "confidence", "needs_review", "source"]].copy()
            show["status"] = show["needs_review"].map({1: "⚠️ Needs review", 0: "✓"})
            show["date"] = show["date"].dt.date
            edited = st.data_editor(
                show.drop(columns=["needs_review"]), hide_index=True, width="stretch", height=460,
                disabled=["id", "date", "description", "amount", "type", "confidence", "source", "status"],
                column_config={
                    "category": st.column_config.SelectboxColumn("Category ✏️", options=ALL_CATEGORIES, required=True),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                    "amount": st.column_config.NumberColumn("Amount (₹)", format="%.2f"),
                    "id": None}, key=f"editor_{user}")
            changed = edited[edited["category"].values != show["category"].values]
            if len(changed):
                for r in changed.itertuples():
                    S.update_category(r.id, r.category)
                st.toast(f"Saved {len(changed)} category correction(s) ✅")
                st.rerun()
    with tab_add:
        if "categorizer" not in models:
            st.warning("Categorizer not trained yet - run `python -m src.train`.")
        desc = st.text_input("Description", placeholder="e.g. UPI/SWIGGY/123456/Payment")
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Amount (₹)", min_value=0.0, value=250.0, step=50.0)
        when = c2.date_input("Date", date.today())
        ttype = c3.selectbox("Type", ["debit", "credit"])
        if desc and "categorizer" in models:
            cat, conf, review = S.suggest_category(desc, amt, when, ttype, models, info)
            badge = "badge-review" if review else "badge-ok"
            st.markdown(f'Suggested category: <b>{cat}</b> <span class="badge {badge}">confidence {conf:.0%}'
                        f'{" - needs review" if review else ""}</span>', unsafe_allow_html=True)
        else:
            cat, conf, review = "Others", 0.0, True
        final = st.selectbox("Category", ALL_CATEGORIES, index=ALL_CATEGORIES.index(cat))
        if st.button("Add transaction", type="primary", disabled=not desc):
            row = pd.DataFrame([dict(date=when, description=desc, amount=amt, type=ttype, category=final,
                                     predicted_category=cat, confidence=conf, needs_review=int(review and final == cat))])
            S.insert_transactions(user, row, "manual")
            st.toast("Transaction added ✅")
    with tab_csv:
        st.caption("CSV with columns `date, description, amount` (optional `type`), or a bank export with "
                   "`Date, Narration, Withdrawal Amt., Deposit Amt.`. Everything is categorised automatically.")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file and st.button("Import", type="primary"):
            with st.spinner("Categorising..."):
                try:
                    res = S.import_csv(file, user, models, info)
                    st.success(f"Imported {res['imported']} of {res['rows_in']} rows. {res['needs_review']} need review.")
                    st.bar_chart(pd.Series(res["by_category"]))
                except Exception as e:
                    st.error(str(e))


def page_subscriptions(user, models, info):
    header("Subscriptions", "Recurring payments detected by ML • autopay is a simulation, no real money moves")
    paid = S.run_autopay(user)
    for m, d, a in paid:
        st.toast(f"Simulated autopay: {m} {S.inr(a)} on {d}")
    with st.spinner("Detecting recurring payments..."):
        S.detect_subscriptions(user, models, info)
    subs = S.get_subscriptions(user)
    if subs.empty:
        return empty_state("No recurring payments detected yet (need at least 2 payments to the same merchant).")
    active = subs[subs["status"] != "dismissed"]
    monthly = (active["avg_amount"] * 30.4 / active["frequency_days"].clip(lower=1)).sum()
    c = st.columns(3)
    c[0].metric("Recurring payments", len(active))
    c[1].metric("Monthly recurring cost", S.inr(monthly))
    c[2].metric("Yearly recurring cost", S.inr(active["yearly_cost"].sum()))
    st.markdown("")
    hdr = st.columns([2.2, 1.2, 1, 1.3, 1.3, 1, 1.6, 1.4])
    for h, t in zip(hdr, ["Merchant", "Amount", "Every", "Next due", "Yearly cost", "Confidence", "Autopay", "Status"]):
        h.markdown(f"**{t}**")
    for r in subs.itertuples():
        col = st.columns([2.2, 1.2, 1, 1.3, 1.3, 1, 1.6, 1.4])
        col[0].markdown(f"{r.merchant}<br><span class='muted'>{r.category}</span>", unsafe_allow_html=True)
        col[1].write(S.inr(r.avg_amount))
        col[2].write(f"{r.frequency_days:.0f} days")
        col[3].write(pd.Timestamp(r.next_due_date).strftime("%d %b %Y"))
        col[4].write(S.inr(r.yearly_cost))
        col[5].write(f"{r.confidence:.0%}")
        ap = col[6].toggle("Autopay (Simulated)", value=bool(r.autopay), key=f"ap_{r.id}")
        if ap != bool(r.autopay):
            S.set_subscription(r.id, autopay=int(ap))
            st.toast(f"Simulated autopay {'on' if ap else 'off'} for {r.merchant}")
            st.rerun()
        if r.status == "confirmed":
            col[7].markdown('<span class="badge badge-ok">Confirmed</span>', unsafe_allow_html=True)
        else:
            b1, b2 = col[7].columns(2)
            if b1.button("✓", key=f"c_{r.id}", help="Confirm"):
                S.set_subscription(r.id, status="confirmed"); st.rerun()
            if b2.button("✕", key=f"d_{r.id}", help="Dismiss (not a subscription)"):
                S.set_subscription(r.id, status="dismissed", autopay=0); st.rerun()


def page_forecast(user, models, info):
    header("Forecast & Budget", "Next-month spending per category, predicted from your own history")
    fc = S.forecast_next_month(user, models, info)
    if fc is None:
        return empty_state("Need at least 3 complete months of transactions to forecast.")
    tm = fc.attrs["target_month"]
    sv = S.savings_forecast(user, fc)
    c = st.columns(4)
    c[0].metric(f"Forecast spend ({tm})", S.inr(sv["spend"]))
    c[1].metric("Expected income", S.inr(sv["income"]))
    c[2].metric("Savings forecast", S.inr(sv["savings"]))
    c[3].metric("Planned investments", S.inr(sv["investments"]))
    t1, t2 = st.tabs(["📈 Forecast", "🧾 Budget"])
    with t1:
        f = fc[fc["forecast"] > 0]
        fig = go.Figure(go.Bar(x=f["category"], y=f["forecast"], marker_color=[CAT_COLORS[c] for c in f["category"]],
                               error_y=dict(type="data", symmetric=False, array=f["high"] - f["forecast"],
                                            arrayminus=f["forecast"] - f["low"]), name="Forecast"))
        fig.add_scatter(x=f["category"], y=f["avg_3m"], mode="markers", name="Last 3-month avg",
                        marker=dict(color="#111827", symbol="line-ew-open", size=18))
        fig.update_layout(title=f"Predicted spend for {tm} (bars: 10th-90th percentile range)", yaxis_title="₹")
        st.plotly_chart(fig_style(fig, 420), width="stretch")
        t = fc.copy()
        for col in ["forecast", "low", "high", "last_month", "avg_3m"]:
            t[col] = t[col].map(S.inr)
        st.dataframe(t, hide_index=True, width="stretch")
    with t2:
        rb = S.recommend_budget(user, fc)
        if rb.attrs["goal_need"] > 0:
            st.info(f"Your goals need about {S.inr(rb.attrs['goal_need'])}/month. "
                    + (f"Discretionary categories are cut by {rb.attrs['cut']:.0%} to close a gap of {S.inr(rb.attrs['shortfall'])}."
                       if rb.attrs["cut"] > 0 else "Your forecast savings already cover that. 🎉"))
        saved = S.get_budget(user)
        rb["my_budget"] = rb["category"].map(lambda c: saved.get(c)).fillna(rb["recommended"])
        ed = st.data_editor(rb, hide_index=True, width="stretch", disabled=["category", "forecast", "recommended", "reason"],
                            column_config={"my_budget": st.column_config.NumberColumn("My budget ✏️ (₹)", min_value=0, step=500),
                                           "forecast": st.column_config.NumberColumn("Forecast (₹)", format="%d"),
                                           "recommended": st.column_config.NumberColumn("Recommended (₹)", format="%d")})
        if st.button("Save my budget", type="primary"):
            S.save_budget(user, dict(zip(ed["category"], ed["my_budget"].astype(float))))
            st.toast("Budget saved ✅")
        st.markdown("#### This month so far")
        tx = S.load_transactions(user)
        cur = tx[(tx["date"].dt.to_period("M") == pd.Timestamp(date.today()).to_period("M")) & (tx["type"] == "debit")]
        spent = cur.groupby("category")["amount"].sum()
        budget = dict(zip(ed["category"], ed["my_budget"]))
        for cat, b in sorted(budget.items(), key=lambda x: -x[1]):
            if b <= 0:
                continue
            s = float(spent.get(cat, 0))
            st.progress(min(s / b, 1.0), text=f"{cat}: {S.inr(s)} of {S.inr(b)}" + (" ⚠️ over budget" if s > b else ""))


def page_goals(user, models, info):
    header("Goals", "Save for what matters - feasibility uses your forecast savings")
    with st.expander("➕ New goal", expanded=S.get_goals(user).empty):
        with st.form("goal"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Goal name", "Emergency fund")
            target = c2.number_input("Target amount (₹)", min_value=1000.0, value=100000.0, step=5000.0)
            saved = c1.number_input("Already saved (₹)", min_value=0.0, value=0.0, step=1000.0)
            when = c2.date_input("Target date", date.today() + timedelta(days=365))
            if st.form_submit_button("Create goal", type="primary"):
                S.add_goal(user, name, target, saved, when)
                st.toast("Goal created 🎯")
                st.rerun()
    goals = S.get_goals(user)
    if goals.empty:
        return empty_state("No goals yet.")
    fc = S.forecast_next_month(user, models, info)
    monthly = S.savings_forecast(user, fc)["savings"] if fc is not None else 0
    n = max(len(goals), 1)
    st.caption(f"Predicted savings next month: **{S.inr(monthly)}**, split evenly across {n} goal(s) for the estimate below.")
    for g in goals.itertuples():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"### {g.name}")
            c1.progress(min(g.saved_amount / g.target_amount, 1.0),
                        text=f"{S.inr(g.saved_amount)} of {S.inr(g.target_amount)} • target {g.target_date}")
            msg, ok = S.goal_feasibility(g.target_amount, g.saved_amount, monthly / n, g.target_date)
            (c1.success if ok else c1.warning)(msg)
            new = c2.number_input("Saved so far", value=float(g.saved_amount), step=1000.0, key=f"g_{g.id}")
            if new != g.saved_amount:
                S.update_goal_saved(g.id, new); st.rerun()
            if c2.button("Delete", key=f"del_{g.id}"):
                S.delete_goal(g.id); st.rerun()


def page_insights(user, models, info):
    header("Insights", "Your spending persona and patterns")
    tx = S.load_transactions(user)
    if tx.empty:
        return empty_state("No data yet.")
    per = S.persona(user, models, info)
    if per is None:
        st.warning("Persona model not trained yet.")
    elif per["cold_start"]:
        st.info(f"Not enough data yet - personas need 3+ months (you have {per['n_months']}).")
    else:
        c1, c2 = st.columns([1, 2])
        c1.markdown(f'<div class="card">Your spending persona<br><span class="persona">{per["name"]}</span>'
                    f'<p>{per["description"]}</p><p>💡 <b>Tip:</b> {per["tip"]}</p></div>', unsafe_allow_html=True)
        shares = pd.DataFrame({"category": EXPENSE_CATEGORIES,
                               "You": [per["user"][f"share_{c}"] for c in EXPENSE_CATEGORIES],
                               f"Average {per['name']}": [per["avg"][f"share_{c}"] for c in EXPENSE_CATEGORIES]})
        fig = px.bar(shares.melt("category", var_name="who", value_name="share"), x="category", y="share", color="who",
                     barmode="group", title="Your category shares vs your persona",
                     color_discrete_sequence=["#4F46E5", "#CBD5E1"])
        fig.update_yaxes(tickformat=".0%")
        c2.plotly_chart(fig_style(fig), width="stretch")
    p = S.patterns(tx)
    if not p:
        return
    st.markdown("#### Patterns")
    subs = S.get_subscriptions(user)
    c = st.columns(3)
    c[0].metric("Avg weekend day spend", S.inr(p["weekend_daily"]), f"{p['weekend_daily'] / max(p['weekday_daily'], 1) - 1:+.0%} vs weekday")
    c[1].metric("Avg weekday spend", S.inr(p["weekday_daily"]))
    c[2].metric("Recurring share of spending", f"{S.recurring_share(tx, subs):.0%}")
    wt = S.weekend_test(tx)
    if wt:
        verdict = ("**is** statistically different" if wt["significant"] else "is **not** statistically different")
        st.markdown(f'<div class="card">📊 <b>Hypothesis test</b> (Mann-Whitney U, two-sided, α = 0.05): your daily spend on '
                    f'weekends (median {S.inr(wt["weekend_median"])}, n={wt["n_weekend"]}) {verdict} from weekdays '
                    f'(median {S.inr(wt["weekday_median"])}, n={wt["n_weekday"]}); p = {wt["p_value"]:.4f}.</div>',
                    unsafe_allow_html=True)
    a, b = st.columns(2)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wk = p["by_weekday"].reindex(order).fillna(0).reset_index()
    wk.columns = ["day", "avg"]
    a.plotly_chart(fig_style(px.bar(wk, x="day", y="avg", title="Average spend by weekday",
                                    color_discrete_sequence=["#4F46E5"])), width="stretch")
    if len(p["growth"]):
        g = p["growth"].head(6).reset_index()
        g.columns = ["category", "growth"]
        fig = px.bar(g, x="growth", y="category", orientation="h", title="Fastest-growing categories (last 3 vs previous 3 months)",
                     color_discrete_sequence=["#F59E0B"])
        fig.update_xaxes(tickformat="+.0%")
        b.plotly_chart(fig_style(fig), width="stretch")
    st.markdown("#### 🚨 Unusual transactions (last 3 months)")
    un = S.unusual_transactions(tx)
    if un.empty:
        st.caption("Nothing unusual - no payment above Q3 + 3×IQR of your own history for its category.")
    else:
        un = un.assign(date=un["date"].dt.date, amount=un["amount"].map(S.inr),
                       times_typical=un["times_typical"].map(lambda x: f"{x:.1f}× typical"))
        st.dataframe(un, hide_index=True, width="stretch")


def page_admin(user, models, info):
    header("Admin / Models", "Active model versions and their test metrics (from the last training run)")
    if not info:
        return empty_state("No models registered. Run `python -m src.data_prep` and `python -m src.train`.")
    st.markdown(f"🔬 Experiment tracking: [open MLflow UI]({MLFLOW_URL}) (start with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`)")
    st.caption("All metrics are measured on synthetic data generated by this project - not real-world performance.")
    figs = {"categorizer": ["categorizer_confusion_matrix.png", "categorizer_per_class_f1.png", "categorizer_confidence_curve.png"],
            "subscription": ["subscription_confusion_matrix.png", "subscription_roc_pr.png", "subscription_shap.png"],
            "forecaster": ["forecaster_baseline_vs_model.png", "forecaster_pred_vs_actual.png", "forecaster_residuals.png"],
            "persona": ["persona_pca.png", "persona_k_selection.png", "persona_centroid_heatmap.png"]}
    fb = S.feedback_stats()
    with st.container(border=True):
        st.markdown("**Feedback from the app (all users)** - corrections are stored for future retraining "
                    "(automated retraining is out of scope)")
        c = st.columns(4)
        c[0].metric("Categorised transactions", f"{fb['n']:,}")
        c[1].metric("User correction rate", f"{fb['correction_rate']:.1%}")
        c[2].metric("Needs-review rate", f"{fb['review_rate']:.1%}")
        c[3].metric("Avg confidence", f"{fb['avg_confidence']:.2f}")
        if len(fb["top_corrections"]):
            st.dataframe(fb["top_corrections"], hide_index=True, width="stretch")
    tabs = st.tabs([n.title() for n in info])
    for tab, (name, e) in zip(tabs, info.items()):
        with tab:
            c = st.columns(4)
            c[0].metric("Active version", f"v{e['version']}")
            c[1].metric("Algorithm", e["algorithm"][:28])
            c[2].metric("Primary metric", e["primary_metric"])
            c[3].metric("Trained", e["trained_at"][:10])
            mt = pd.DataFrame({"test (model)": e["test_metrics"], "test (baseline)": e.get("baseline_metrics", {})})
            st.dataframe(mt.map(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else ("" if pd.isna(v) else str(v))),
                         width="stretch")
            comp = RESULTS / f"model_comparison_{name}.csv"
            if comp.exists():
                st.markdown("**All candidates**")
                st.dataframe(pd.read_csv(comp).round(4), hide_index=True, width="stretch")
            cols = st.columns(3)
            for col, f in zip(cols, figs.get(name, [])):
                if (RESULTS / f).exists():
                    col.image(str(RESULTS / f), width="stretch")
            st.caption(f"File: models/{e['file']} • git commit {e.get('git_commit', '?')}")


# ====================================================================================== sidebar / routing
def main():
    S.init_db()
    models, info = get_models()
    with st.sidebar:
        st.markdown("## 💰 Smart Finance")
        username = st.text_input("Username", key="username", placeholder="type a username").strip().lower()
        if not username:
            st.caption("No password needed - this is a demo login.")
        user = S.get_user(username) if username else None
        if username and not user:
            inc = st.number_input("Monthly income (₹)", min_value=0.0, value=50000.0, step=5000.0)
            if st.button("Create account", type="primary"):
                S.create_user(username, inc)
                st.rerun()
        if user:
            st.success(f"Signed in as **{username}**")
            if st.button("📦 Load demo data", help="Replaces your data with 24 months of a generated user's history"):
                with st.spinner("Loading and categorising demo history..."):
                    try:
                        n = S.load_demo(username, models, info)
                        st.toast(f"Loaded {n} demo transactions ✅")
                    except Exception as e:
                        st.error(str(e))
            st.divider()
            page = st.radio("Navigate", ["🏠 Dashboard", "💳 Transactions", "🔁 Subscriptions", "📈 Forecast & Budget",
                                         "🎯 Goals", "🧠 Insights", "⚙️ Admin / Models"], label_visibility="collapsed")
        else:
            page = None
        if not models:
            st.warning("Models not found. Run `python -m src.data_prep && python -m src.train`.")
    if not user:
        header("Smart Finance Tracker", "ML-powered expense categorisation, subscription detection, forecasting and personas")
        st.markdown("👈 **Type a username in the sidebar to start.** New users are created automatically.")
        return
    pages = {"🏠 Dashboard": page_dashboard, "💳 Transactions": page_transactions, "🔁 Subscriptions": page_subscriptions,
             "📈 Forecast & Budget": page_forecast, "🎯 Goals": page_goals, "🧠 Insights": page_insights,
             "⚙️ Admin / Models": page_admin}
    pages[page](username, models, info)


main()
