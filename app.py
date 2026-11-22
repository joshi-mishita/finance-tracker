"""Smart Finance Tracker - Streamlit app.  Run: python -m streamlit run app.py"""
import html
import math
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import services as S
from src.data_prep import ALL_CATEGORIES, DISCRETIONARY, ESSENTIALS, EXPENSE_CATEGORIES, NON_EXPENSE

st.set_page_config(page_title="Smart Finance Tracker", page_icon=":material/account_balance_wallet:", layout="wide")
RESULTS = Path(__file__).parent / "results"
MLFLOW_URL = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000")

# ====================================================================================== design tokens
BG, PANEL, LINE, TEXT, MUTED = "#0B0E0A", "#121711", "#232B21", "#EEF4E8", "#8E9B88"
LIME, LIME2, PINK, CYAN, PURPLE, YELLOW = "#8CF542", "#62D226", "#FF4FD8", "#2ED3E6", "#A78BFA", "#FACC15"
POS, NEG = LIME, "#FF6BC9"
CAT_COLORS = {
    "Food & Dining": "#FF9F43", "Groceries": "#8CF542", "Transport": "#2ED3E6", "Shopping": "#FF4FD8",
    "Bills & Utilities": "#A78BFA", "Rent": "#7C83FD", "Entertainment": "#FACC15", "Health": "#34D399",
    "Education": "#F472B6", "Travel": "#38BDF8", "Subscriptions": "#C084FC", "EMI & Loans": "#FB7185",
    "Investments": "#4ADE80", "Personal Care": "#FDE68A", "Others": "#94A3B8", "Salary": "#8CF542",
    "Other Income": "#BEF264",
}

CSS = f"""
<style>
@font-face {{font-family: "Archivo Black"; src: url("app/static/fonts/ArchivoBlack.woff2") format("woff2"); font-display: swap;}}
@font-face {{font-family: "Poppins"; src: url("app/static/fonts/Poppins-Regular.woff2") format("woff2"); font-weight: 400; font-display: swap;}}
@font-face {{font-family: "Poppins"; src: url("app/static/fonts/Poppins-Medium.woff2") format("woff2"); font-weight: 500; font-display: swap;}}
@font-face {{font-family: "Poppins"; src: url("app/static/fonts/Poppins-SemiBold.woff2") format("woff2"); font-weight: 600; font-display: swap;}}
@font-face {{font-family: "Poppins"; src: url("app/static/fonts/Poppins-Bold.woff2") format("woff2"); font-weight: 700; font-display: swap;}}
@font-face {{font-family: "Poppins"; src: url("app/static/fonts/Poppins-ExtraBold.woff2") format("woff2"); font-weight: 800; font-display: swap;}}
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{font-family: "Poppins", system-ui, sans-serif;}}
span[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] *, .material-symbols-rounded {{font-family: "Material Symbols Rounded" !important;}}
.stApp {{background-color: {BG};
  background-image: linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px);
  background-size: 34px 34px;}}
header[data-testid="stHeader"] {{background: transparent;}}
.block-container {{padding-top: 2rem; padding-bottom: 3rem; max-width: 1340px;}}
h1, h2, h3 {{color: {TEXT};}}

/* page title */
.pt h1 {{font-family: "Archivo Black", "Poppins", sans-serif; font-size: 3rem; line-height: 1; letter-spacing: -.01em;
         text-transform: uppercase; margin: 0; padding: 0; color: #FFFFFF;}}
.meta {{display: flex; align-items: center; gap: .55rem; margin: .7rem 0 1.5rem 0; color: {MUTED}; font-size: .74rem;
        font-weight: 700; letter-spacing: .14em; text-transform: uppercase;}}
.dot {{width: 7px; height: 7px; border-radius: 50%; background: {LIME}; box-shadow: 0 0 10px {LIME};}}
.cap {{color: {MUTED}; font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;}}
.chip {{display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .72rem; font-weight: 700;
        background: rgba(140,245,66,.12); color: {LIME}; letter-spacing: .04em; white-space: nowrap;}}
.chip.pink {{background: rgba(255,79,216,.13); color: {PINK};}} .chip.cyan {{background: rgba(46,211,230,.13); color: {CYAN};}}
.chip.yellow {{background: rgba(250,204,21,.13); color: {YELLOW};}} .chip.dim {{background: #1C2319; color: {MUTED};}}
.phead {{display: flex; justify-content: space-between; align-items: center; margin-bottom: .9rem;}}
.muted, p.muted, .muted p {{color: {MUTED}; font-size: .86rem !important; line-height: 1.6;}}
div[class*="st-key-setgoal"] button {{background: {LIME} !important; border: none !important; box-shadow: 0 4px 0 {LIME2}, 0 10px 30px -10px rgba(140,245,66,.7);}}
div[class*="st-key-setgoal"] button p, div[class*="st-key-setgoal"] button span {{color: #0B0E0A !important; font-weight: 800; letter-spacing: .08em; text-transform: uppercase;}}

/* panels: keyed containers */
div[class*="st-key-panel_"] {{background: {PANEL}; border: 1px solid {LINE} !important; border-radius: 22px !important;
     padding: 1.3rem 1.4rem !important;}}

/* lime hero */
.hero {{position: relative; background: linear-gradient(135deg, #A4F94F 0%, #7BE433 55%, #5FCB1F 100%); border-radius: 26px;
        padding: 1.6rem 1.7rem 1.4rem; color: #0B0E0A; box-shadow: 0 0 0 1px rgba(140,245,66,.4), 0 18px 60px -18px rgba(140,245,66,.55);
        overflow: hidden; margin-bottom: 1rem;}}
.hero .cap {{color: rgba(11,14,10,.7);}}
.hero .big {{font-family: "Poppins"; font-weight: 800; font-size: clamp(2.6rem, 5vw, 4.1rem); line-height: 1.05; letter-spacing: -.03em;
             margin: .35rem 0 .2rem; font-variant-numeric: tabular-nums;}}
.hero .big small {{font-size: .95rem; font-weight: 700; letter-spacing: .05em; margin-left: .3rem;}}
.hero .sub {{font-weight: 600; font-size: .92rem; color: rgba(11,14,10,.82);}}
.hero .tiles {{display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; margin-top: 1.3rem; position: relative; z-index: 1;}}
.hero .tile {{background: rgba(11,14,10,.10); border: 1px solid rgba(11,14,10,.14); border-radius: 14px; padding: .7rem .9rem;}}
.hero .tile .cap {{font-size: .64rem;}}
.hero .tile b {{display: block; font-size: 1.25rem; font-weight: 800; margin-top: .15rem;}}
.hero .spark {{position: absolute; right: 1.5rem; top: 1.3rem; width: 30%; height: 34%; opacity: .95;}}
.hero .spark-cap {{position: absolute; right: 1.5rem; top: calc(1.3rem + 34% + .2rem); font-size: .62rem; font-weight: 700;
     letter-spacing: .12em; text-transform: uppercase; color: rgba(11,14,10,.6);}}

/* stat tiles */
.tiles2 {{display: grid; gap: .9rem; margin-bottom: 1rem;}}
.t2 {{background: {PANEL}; border: 1px solid {LINE}; border-radius: 18px; padding: 1rem 1.15rem; min-width: 0;}}
.t2 .v {{font-weight: 800; font-size: 1.75rem; color: #FFFFFF; margin: .3rem 0 .1rem; white-space: nowrap; overflow: hidden;
         text-overflow: ellipsis; font-variant-numeric: tabular-nums; letter-spacing: -.02em;}}
.t2 .v.pos {{color: {POS};}} .t2 .v.neg {{color: {NEG};}} .t2 .v.cyan {{color: {CYAN};}} .t2 .v.yellow {{color: {YELLOW};}}
.t2 .n {{color: {MUTED}; font-size: .8rem;}} .t2 .n .up {{color: {POS}; font-weight: 700;}} .t2 .n .down {{color: {NEG}; font-weight: 700;}}

/* activity rows */
.act {{display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: .62rem 0; border-bottom: 1px solid #1B2219;}}
.act:last-child {{border-bottom: none;}}
.act .nm {{color: #FFFFFF; font-weight: 700; font-size: .93rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
.act .mt {{color: {MUTED}; font-size: .78rem;}}
.act .am {{font-weight: 700; font-size: .95rem; white-space: nowrap; font-variant-numeric: tabular-nums;}}
.am.pos {{color: {POS};}} .am.neg {{color: {NEG};}} .am.plain {{color: #FFFFFF;}}

/* bubbles (spending split) */
.bubbles {{display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 1rem; padding: .4rem 0;}}
.bub {{border-radius: 50%; display: grid; place-items: center; text-align: center; color: #FFFFFF; font-weight: 800;
       box-shadow: inset 0 -8px 18px rgba(0,0,0,.25), 0 10px 30px -10px currentColor;}}
.bub span {{display: block; font-size: .62rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .9;}}

/* budget cards */
.bud {{background: {PANEL}; border: 1px solid {LINE}; border-radius: 18px; padding: 1rem 1.1rem; margin-bottom: .8rem;
       display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: .2rem 1rem; align-items: center;}}
.bud .t {{color: #FFFFFF; font-weight: 700;}}
.bud .bar {{height: 6px; background: #222A20; border-radius: 99px; overflow: hidden; margin: .55rem 0;}}
.bud .bar i {{display: block; height: 100%; border-radius: 99px;}}
.bud .amt {{text-align: right; font-weight: 800; color: #FFFFFF; font-size: 1.15rem; white-space: nowrap;}}
.bud .amt small {{display: block; color: {MUTED}; font-weight: 600; font-size: .78rem;}}

/* milestones */
.ms {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .6rem; margin: 1.2rem 0 1rem;}}
.ms > div {{border: 1px solid {LINE}; border-radius: 14px; padding: .55rem .7rem; background: #151B13;}}
.ms > div.on {{border-color: rgba(140,245,66,.55); background: rgba(140,245,66,.08);}}
.ms > div .cap {{font-size: .62rem;}} .ms > div.on .cap {{color: {LIME};}}
.ms > div b {{display: block; color: #FFFFFF; font-size: 1rem; font-weight: 700; white-space: nowrap;}}

/* badges */
.badges {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin-bottom: 1rem;}}
.bd {{background: {PANEL}; border: 1px solid {LINE}; border-radius: 18px; padding: 1rem; text-align: center;}}
.bd .ic {{width: 54px; height: 54px; border-radius: 50%; margin: .2rem auto .7rem; display: grid; place-items: center;
          background: #1C2319; color: #44513F; font-weight: 800; font-size: 1.3rem;}}
.bd.on {{border-color: rgba(140,245,66,.45); box-shadow: 0 0 30px -12px {LIME};}}
.bd.on .ic {{background: {LIME}; color: #0B0E0A; box-shadow: 0 0 22px -4px {LIME};}}
.bd b {{display: block; color: #FFFFFF; font-size: .92rem;}} .bd:not(.on) b {{color: #7C8A76;}}
.bd small {{color: {MUTED}; font-size: .74rem;}}

/* quick action circles */
div[class*="st-key-qa_"] button {{background: transparent !important; border: none !important; box-shadow: none !important;
     flex-direction: column; gap: .55rem; height: auto; padding: .2rem 0;}}
div[class*="st-key-qa_"] button > div, div[class*="st-key-qa_"] button > div > span {{display: flex; flex-direction: column;
     align-items: center; gap: .65rem;}}
div[class*="st-key-qa_"] button > div > span > span:first-child {{width: auto !important; height: auto !important; margin: 0 !important;}}
div[class*="st-key-qa_"] button span[data-testid="stIconMaterial"] {{width: 70px; height: 70px; border-radius: 50%; display: grid;
     place-items: center; font-size: 30px; color: #0B0E0A; box-shadow: 0 0 0 3px rgba(0,0,0,.35), 0 10px 28px -8px currentColor;
     transition: transform .15s ease;}}
div[class*="st-key-qa_"] button:hover span[data-testid="stIconMaterial"] {{transform: translateY(-2px);}}
div[class*="st-key-qa_"] button p {{color: #FFFFFF; font-weight: 700; font-size: .74rem; letter-spacing: .12em; text-transform: uppercase;}}
.st-key-qa_goal span[data-testid="stIconMaterial"] {{background: {CYAN};}}
.st-key-qa_spend span[data-testid="stIconMaterial"] {{background: {PURPLE};}}
.st-key-qa_rewards span[data-testid="stIconMaterial"] {{background: {YELLOW};}}
.st-key-qa_insights span[data-testid="stIconMaterial"] {{background: {PINK};}}

/* buttons */
.stButton > button, .stFormSubmitButton > button, div[data-testid="stPopover"] button {{border-radius: 14px; font-weight: 700;}}
button[kind="primary"], button[kind="primaryFormSubmit"] {{background: {LIME} !important; color: #0B0E0A !important; border: none !important;
     box-shadow: 0 4px 0 {LIME2}, 0 10px 30px -10px rgba(140,245,66,.7);}}
button[kind="primary"] p, button[kind="primaryFormSubmit"] p {{color: #0B0E0A !important; font-weight: 800;}}
button[kind="secondary"], button[kind="secondaryFormSubmit"] {{background: #161C14 !important; border: 1px solid #2B3428 !important;}}
div[data-baseweb="tab-highlight"] {{background-color: {LIME} !important; height: 3px;}}
button[data-baseweb="tab"] p {{font-weight: 700;}}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{border: 1px solid {LINE}; border-radius: 14px; overflow: hidden;}}
div[data-testid="stProgress"] div[role="progressbar"] > div > div {{background: linear-gradient(90deg, {LIME2}, {LIME});}}
div[data-baseweb="input"], div[data-baseweb="select"] > div {{border-radius: 12px !important;}}
*:focus-visible {{outline: 2px solid {LIME} !important; outline-offset: 2px;}}
.stApp code {{color: {LIME}; background: #1A2117;}}

/* sidebar */
section[data-testid="stSidebar"] {{background: #0E120D; border-right: 1px solid {LINE};}}
div[data-testid="stSidebarHeader"] {{padding-top: 1.4rem;}}
div[data-testid="stSidebarHeader"] img {{height: 2.2rem !important; max-width: 100%;}}
ul[data-testid="stSidebarNavItems"] {{gap: .35rem; display: flex; flex-direction: column; padding-top: .6rem;}}
a[data-testid="stSidebarNavLink"] {{border-radius: 14px; padding: .72rem 1rem !important; margin: 0 .2rem; transition: background .15s;}}
a[data-testid="stSidebarNavLink"] span {{color: #C9D3C2 !important; font-weight: 700 !important; letter-spacing: .12em;
     text-transform: uppercase; font-size: .8rem !important;}}
a[data-testid="stSidebarNavLink"]:hover {{background: #172015 !important;}}
a[data-testid="stSidebarNavLink"][aria-current="page"] {{background: {LIME} !important; box-shadow: 0 4px 0 {LIME2}, 0 10px 30px -12px {LIME};}}
a[data-testid="stSidebarNavLink"][aria-current="page"] span {{color: #0B0E0A !important;}}
div[data-testid="stSidebarNavSeparator"] {{display: none;}}
.lvl {{background: {PANEL}; border: 1px solid {LINE}; border-radius: 18px; padding: 1rem 1.05rem; margin: .4rem 0 1rem;}}
.lvl .name {{font-family: "Archivo Black", sans-serif; color: #FFFFFF; font-size: 1.2rem; text-transform: uppercase; margin: .25rem 0 .6rem;}}
.xpbar {{height: 7px; background: #222A20; border-radius: 99px; overflow: hidden;}}
.xpbar i {{display: block; height: 100%; background: linear-gradient(90deg, {LIME2}, {LIME}); box-shadow: 0 0 12px {LIME};}}
.lvl .row {{display: flex; justify-content: space-between; align-items: center; margin-top: .6rem; color: {MUTED}; font-size: .78rem;}}
.lvl hr {{border: none; border-top: 1px solid {LINE}; margin: .75rem 0 .55rem;}}
.prof {{display: flex; align-items: center; gap: .7rem; margin: .4rem 0 .7rem;}}
.ava {{width: 44px; height: 44px; border-radius: 50%; padding: 3px; background: conic-gradient({LIME}, {CYAN}, {PURPLE}, {PINK}, {LIME});}}
.ava div {{width: 100%; height: 100%; border-radius: 50%; background: #0E120D; display: grid; place-items: center; color: #FFFFFF; font-weight: 800;}}
.prof b {{color: #FFFFFF; display: block; font-size: .92rem;}} .prof span {{color: {MUTED}; font-size: .76rem;}}
/* floating AI assistant button + chat window */
.st-key-chatfab {{position: fixed; right: 28px; bottom: 26px; z-index: 999990; width: auto !important;}}
.st-key-chatfab button {{width: 62px; height: 62px; min-height: 62px; border-radius: 50% !important; padding: 0 !important;
     background: {LIME} !important; border: none !important; box-shadow: 0 0 0 4px rgba(140,245,66,.18), 0 12px 32px -6px rgba(140,245,66,.75) !important;
     transition: transform .15s ease;}}
.st-key-chatfab button:hover {{transform: translateY(-2px) scale(1.04);}}
.st-key-chatfab button span[data-testid="stIconMaterial"] {{color: #0B0E0A !important; font-size: 28px;}}
.st-key-chatfab button p {{position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0);}}
.st-key-chatfab::after {{content: ""; position: absolute; top: 4px; right: 4px; width: 12px; height: 12px; border-radius: 50%;
     background: {PINK}; border: 2px solid {BG};}}
div[role="dialog"] {{background: {PANEL} !important; border: 1px solid {LINE}; border-radius: 22px !important;}}
div[role="dialog"] h2, div[role="dialog"] [data-testid="stDialogHeader"] div {{font-family: "Archivo Black", sans-serif !important; text-transform: uppercase;
     letter-spacing: .01em; color: #FFFFFF;}}
div[class*="st-key-chatbox"] {{background: #0E120D; border: 1px solid {LINE} !important; border-radius: 16px !important;}}
div[data-testid="stChatMessage"] {{background: transparent; padding: .35rem .2rem;}}
div[data-testid="stChatMessageAvatarAssistant"] {{background: {LIME} !important; color: #0B0E0A !important;}}
div[data-testid="stChatMessageAvatarUser"] {{background: {PURPLE} !important; color: #0B0E0A !important;}}
div[class*="st-key-sug"] button {{text-align: left; justify-content: flex-start; background: #161C14 !important; border: 1px solid #2B3428 !important;
     border-radius: 12px !important; font-weight: 500 !important;}}
div[class*="st-key-sug"] button:hover {{border-color: {LIME} !important;}}
div[data-testid="stChatInput"] {{border-radius: 14px !important;}}
div[data-testid="stChatInput"] textarea, div[data-testid="stChatInput"] textarea:focus {{outline: none !important; box-shadow: none !important; border: none !important;}}
.foot {{text-align: center; color: #5E6A59; font-size: .74rem; margin-top: 2rem;}}
@media (max-width: 760px) {{ .hero .spark {{display: none;}} .badges {{grid-template-columns: repeat(2, minmax(0, 1fr));}}
  .ms {{grid-template-columns: repeat(2, minmax(0, 1fr));}} .pt h1 {{font-size: 2.2rem;}} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ====================================================================================== UI helpers
_PANELS = [0]
NAV = {}


@st.cache_resource
def get_models():
    return S.load_models()


def esc(x):
    return html.escape(str(x))


def panel(height="content"):
    _PANELS[0] += 1
    return st.container(border=True, height=height, key=f"panel_{_PANELS[0]}")


def title(text, tx=None, meta=None):
    if meta is None:
        if tx is not None and not tx.empty:
            meta = f"Your ledger &middot; ₹ &middot; {tx['date'].min():%Y-%m-%d} &rarr; {tx['date'].max():%Y-%m-%d}"
        else:
            meta = "Your ledger &middot; ₹"
    st.markdown(f'<div class="pt"><h1>{esc(text)}</h1></div><div class="meta"><span class="dot"></span>{meta}</div>',
                unsafe_allow_html=True)


def phead(label, chip="", chip_tone=""):
    st.markdown(f'<div class="phead"><span class="cap">{esc(label)}</span>'
                + (f'<span class="chip {chip_tone}">{esc(chip)}</span>' if chip else "") + "</div>", unsafe_allow_html=True)


def tiles(cells):
    """cells: list of (label, value, tone, note_html); tone in '', pos, neg, cyan, yellow."""
    parts = "".join(f'<div class="t2"><div class="cap">{esc(l)}</div><div class="v {t}">{esc(v)}</div><div class="n">{n}</div></div>'
                    for l, v, t, n in cells)
    st.markdown(f'<div class="tiles2" style="grid-template-columns: repeat({len(cells)}, minmax(0, 1fr));">{parts}</div>',
                unsafe_allow_html=True)


def delta(curr, prev, money=True, good_up=True, suffix=""):
    d = curr - prev
    if abs(d) < 1e-9:
        return f"no change{suffix}"
    good = (d > 0) == good_up
    txt = (("+" if d > 0 else "-") + S.inr(abs(d))) if money else f"{d * 100:+.1f} pts"
    return f'<span class="{"up" if good else "down"}">{txt}</span>{suffix}'


def amount_html(amount, ttype):
    sign, tone = ("+", "pos") if ttype == "credit" else ("-", "neg")
    return f'<span class="am {tone}">{sign}{S.inr(amount, 2)}</span>'


def activity_html(df):
    return "".join(f'<div class="act"><div style="min-width:0"><div class="nm">{esc(r.description.upper()[:30])}</div>'
                   f'<div class="mt">{esc(r.category)} &middot; {r.date:%d %b %Y}</div></div>{amount_html(r.amount, r.type)}</div>'
                   for r in df.itertuples())


def ring_svg(pct, size=180, color=LIME, segs=36, label="", sub="", width=None):
    """Segmented progress ring."""
    pct = max(0.0, min(1.0, pct))
    r_out, r_in = size / 2 - 4, size / 2 - (width or size * 0.13)
    cx = cy = size / 2
    on = round(pct * segs)
    lines = []
    for i in range(segs):
        a = -math.pi / 2 + 2 * math.pi * (i + 0.5) / segs
        x1, y1 = cx + r_in * math.cos(a), cy + r_in * math.sin(a)
        x2, y2 = cx + r_out * math.cos(a), cy + r_out * math.sin(a)
        col = color if i < on else "#283126"
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
                     f'stroke-width="{max(3, size / 32):.1f}" stroke-linecap="round"/>')
    txt = ""
    if label:
        txt = (f'<text x="{cx}" y="{cy + size * 0.06}" text-anchor="middle" fill="#FFFFFF" font-family="Poppins" font-weight="800" '
               f'font-size="{size * 0.2:.0f}">{esc(label)}</text>')
        if sub:
            txt += (f'<text x="{cx}" y="{cy + size * 0.17}" text-anchor="middle" fill="{MUTED}" font-family="Poppins" font-weight="700" '
                    f'font-size="{size * 0.055:.0f}" letter-spacing="2">{esc(sub.upper())}</text>')
    glow = f'style="filter: drop-shadow(0 0 8px {color}55)"' if on else ""
    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" {glow}>{"".join(lines)}{txt}</svg>'


def spark_svg(values, w=340, h=120):
    if len(values) < 2:
        return ""
    lo, hi = min(min(values), 0), max(max(values), 1)
    xs = [i * w / (len(values) - 1) for i in range(len(values))]
    ys = [h - 8 - (v - lo) / (hi - lo) * (h - 16) for v in values]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    zero = h - 8 - (0 - lo) / (hi - lo) * (h - 16)
    area = f"0,{h} " + pts + f" {w},{h}"
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#0B0E0A"/>' for x, y in zip(xs, ys))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polygon points="{area}" fill="rgba(11,14,10,.10)"/>'
            f'<line x1="0" y1="{zero:.1f}" x2="{w}" y2="{zero:.1f}" stroke="rgba(11,14,10,.25)" stroke-dasharray="4 5"/>'
            f'<polyline points="{pts}" fill="none" stroke="#0B0E0A" stroke-width="3" stroke-linejoin="round"/>{dots}</svg>')


def chart(fig, h=320, legend=True):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=8, b=6), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Poppins, sans-serif", size=12, color="#C9D3C2"),
                      legend=dict(orientation="h", y=-0.16, x=0, title=None, font=dict(size=11)), showlegend=legend,
                      hoverlabel=dict(font_family="Poppins", bgcolor="#121711", bordercolor=LINE), title=None)
    fig.update_xaxes(showgrid=False, linecolor=LINE, title=None, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#1E261C", zeroline=False, title=None, tickfont=dict(color=MUTED))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def empty_panel(heading, text, user=None, models=None, info=None, key="empty_demo"):
    with panel():
        st.markdown(f'<div style="padding:1rem .4rem .4rem"><div class="pt"><h1 style="font-size:1.6rem">{esc(heading)}</h1></div>'
                    f'<p class="muted" style="max-width:60ch;margin:.6rem 0 1rem">{esc(text)}</p></div>', unsafe_allow_html=True)
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


def go_to(name):
    if name in NAV:
        st.switch_page(NAV[name])


# ====================================================================================== pages
def page_overview(user, models, info):
    tx = S.load_transactions(user)
    title("Overview", tx)
    if tx.empty:
        return empty_panel("No transactions yet", "Load 24 months of demo history to explore every page, or add your own on "
                           "the Transactions page (one at a time or as a bank-statement CSV).", user, models, info)
    g = S.gamification(user)
    m = latest_month(tx)
    cur, prev = S.kpis(tx, m), S.kpis(tx, m - 1)
    ms = g["monthly"]
    total_saved = float(ms.sum()) if len(ms) else 0.0
    month_tx = tx[(tx["date"].dt.to_period("M") == m) & (tx["type"] == "debit")]
    goals = S.get_goals(user)
    funded = f"{goals['saved_amount'].sum() / goals['target_amount'].sum():.0%}" if not goals.empty else "No goal"

    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        st.markdown(f"""
<div class="hero">
  {spark_svg(list(ms.tail(12).values))}<div class="spark-cap">Saved per month, last 12</div>
  <div class="cap">Total saved</div>
  <div class="big">{S.inr(total_saved)}<small>INR</small></div>
  <div class="sub">&darr; {S.inr(cur['expenses'])} spent in {m.strftime('%B')} &middot; {len(month_tx)} payments</div>
  <div class="tiles">
    <div class="tile"><div class="cap">Savings rate</div><b>{cur['rate']:.1%}</b></div>
    <div class="tile"><div class="cap">Goal funded</div><b>{funded}</b></div>
    <div class="tile"><div class="cap">Saving streak</div><b>{g['streak']} month{'s' if g['streak'] != 1 else ''}</b></div>
  </div>
</div>""", unsafe_allow_html=True)
        q = st.columns(4)
        acts = [("qa_goal", "Set a goal", ":material/track_changes:", "Goals"),
                ("qa_spend", "Log a spend", ":material/receipt_long:", "Transactions"),
                ("qa_rewards", "Rewards", ":material/star:", "Rewards"),
                ("qa_insights", "Insights", ":material/bolt:", "Insights")]
        for col, (k, label, icon, dest) in zip(q, acts):
            with col:
                if st.button(label, icon=icon, key=k, width="stretch"):
                    go_to(dest)
    with right, panel(height=470):
        phead("Recent activity", "5 latest")
        st.markdown(activity_html(tx.sort_values("date", ascending=False).head(5)), unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("+ Log a spend", width="stretch", key="ov_log"):
            go_to("Transactions")
        if b2.button("Import CSV", type="primary", width="stretch", key="ov_csv"):
            go_to("Transactions")

    vs = f" vs {(m - 1).strftime('%b')}"
    tiles([("Money in", S.inr(cur["income"]), "pos", delta(cur["income"], prev["income"], suffix=vs)),
           ("Money out", S.inr(cur["expenses"]), "neg", delta(cur["expenses"], prev["expenses"], good_up=False, suffix=vs)),
           ("Saved", S.inr(cur["savings"]), "pos" if cur["savings"] >= 0 else "neg", delta(cur["savings"], prev["savings"], suffix=vs)),
           ("Level", f"{g['level_index']} of {g['n_levels']}", "yellow", f"{esc(g['level'])}, {g['xp']:,} XP")])

    a, b = st.columns([1.5, 1], gap="medium")
    with a, panel():
        phead("Month by month", "Last 12 months", "dim")
        d = tx.assign(month=tx["date"].dt.to_period("M"))
        d["kind"] = d.apply(lambda r: "Money in" if r["type"] == "credit" else
                            ("Investments" if r["category"] in NON_EXPENSE else "Money out"), axis=1)
        months = sorted(d["month"].unique())[-12:]
        trend = d[d["month"].isin(months)].groupby(["month", "kind"])["amount"].sum().reset_index()
        trend["month"] = trend["month"].dt.strftime("%b %y")
        fig = px.bar(trend, x="month", y="amount", color="kind", barmode="group",
                     category_orders={"month": [x.strftime("%b %y") for x in months], "kind": ["Money in", "Money out", "Investments"]},
                     color_discrete_map={"Money in": LIME, "Money out": PINK, "Investments": CYAN})
        fig.update_traces(hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>", marker_line_width=0)
        fig.update_layout(bargap=0.28, bargroupgap=0.1)
        chart(fig, 300)
    with b, panel():
        phead("Spending split", m.strftime("%B"), "dim")
        spend = month_tx[~month_tx["category"].isin(NON_EXPENSE)]
        tot = spend["amount"].sum() or 1
        ess = spend[spend["category"].isin(ESSENTIALS)]["amount"].sum() / tot
        life = spend[spend["category"].isin(DISCRETIONARY)]["amount"].sum() / tot
        oth = max(1 - ess - life, 0)
        bubs = [(ess, "Essentials", PURPLE), (life, "Lifestyle", CYAN), (oth, "Other", PINK)]
        st.markdown('<div class="bubbles">' + "".join(
            f'<div class="bub" style="width:{70 + 90 * v:.0f}px;height:{70 + 90 * v:.0f}px;background:{c};color:{c}">'
            f'<div style="color:#FFFFFF;font-size:{1 + v * .9:.2f}rem">{v:.0%}<span>{n}</span></div></div>' for v, n, c in bubs)
            + "</div>", unsafe_allow_html=True)
        st.markdown('<div class="muted" style="text-align:center;margin-top:.6rem">Essentials: rent, groceries, bills, EMIs, '
                    'health. Lifestyle: dining, shopping, entertainment, travel.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1, panel(height=330):
        phead("Where the money went", m.strftime("%B"), "dim")
        by_cat = month_tx.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
        fig = px.pie(by_cat, names="category", values="amount", hole=0.66, color="category", color_discrete_map=CAT_COLORS)
        fig.update_traces(textinfo="none", sort=False, marker=dict(line=dict(color=PANEL, width=3)),
                          hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>")
        fig.add_annotation(text=f"<b>{S.inr(by_cat['amount'].sum())}</b>", showarrow=False,
                           font=dict(family="Poppins", size=17, color="#FFFFFF"))
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)))
        chart(fig, 250)
    with c2, panel(height=330):
        phead("Upcoming bills", "Next 30 days", "cyan")
        subs = S.get_subscriptions(user)
        if subs.empty and "subscription" in models:
            S.detect_subscriptions(user, models, info)
            subs = S.get_subscriptions(user)
        rows = ""
        if not subs.empty:
            subs["next_due_date"] = pd.to_datetime(subs["next_due_date"])
            today = pd.Timestamp(date.today())
            up = subs[subs["next_due_date"] <= today + pd.Timedelta(days=30)]
            for r in up.head(5).itertuples():
                tag = ('<span class="chip">Autopay</span>' if r.autopay else
                       ('<span class="chip pink">Overdue</span>' if r.next_due_date < today else ""))
                rows += (f'<div class="act"><div style="min-width:0"><div class="nm">{esc(r.merchant.title())}</div>'
                         f'<div class="mt">{"Due" if r.next_due_date >= today else "Was due"} {r.next_due_date:%d %b} {tag}</div></div>'
                         f'<span class="am plain">{S.inr(r.avg_amount)}</span></div>')
        st.markdown(rows or '<div class="muted">Nothing due in the next 30 days.</div>', unsafe_allow_html=True)
    st.markdown('<div class="foot">Every number on this page is computed from your own transactions.</div>', unsafe_allow_html=True)


def page_transactions(user, models, info):
    tx = S.load_transactions(user)
    title("Transactions", tx)
    tab_list, tab_add, tab_csv = st.tabs(["All transactions", "Log a spend", "Import a CSV"])
    with tab_list:
        if tx.empty:
            empty_panel("No transactions yet", "Log one in the next tab, import a bank statement, or load the demo history.",
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
            st.markdown(f'<span class="chip yellow">{n_rev} need review</span> &nbsp;<span class="muted">Showing {len(v):,} of '
                        f'{len(tx):,} transactions. Change a category in the table and it saves straight away.</span>',
                        unsafe_allow_html=True)
            show = v[["id", "date", "description", "amount", "type", "category", "confidence", "needs_review"]].copy()
            show["status"] = show["needs_review"].map({1: "Needs review", 0: "OK"})
            show["date"] = show["date"].dt.date
            edited = st.data_editor(
                show.drop(columns=["needs_review"]), hide_index=True, width="stretch", height=500,
                disabled=["id", "date", "description", "amount", "type", "confidence", "status"],
                column_config={
                    "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                    "description": st.column_config.TextColumn("Description", width="medium"),
                    "category": st.column_config.SelectboxColumn("Category", options=ALL_CATEGORIES, required=True),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent", width="small"),
                    "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                    "type": st.column_config.TextColumn("Type", width="small"),
                    "status": st.column_config.TextColumn("Status", width="small"), "id": None}, key=f"editor_{user}")
            changed = edited[edited["category"].values != show["category"].values]
            if len(changed):
                for r in changed.itertuples():
                    S.update_category(r.id, r.category)
                st.toast(f"Saved {len(changed)} category change(s). +{len(changed) * S.XP_RULES['correction']} XP")
                st.rerun()
    with tab_add, panel():
        if "categorizer" not in models:
            st.warning("The categorizer is not trained yet. Run python -m src.train.")
        desc = st.text_input("Description", placeholder="As it appears on your statement, e.g. UPI/SWIGGY/123456/Payment")
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Amount (₹)", min_value=0.0, value=250.0, step=50.0)
        when = c2.date_input("Date", date.today())
        ttype = c3.selectbox("Type", ["debit", "credit"], format_func=lambda t: "Money out" if t == "debit" else "Money in")
        if desc and "categorizer" in models:
            cat, conf, review = S.suggest_category(desc, amt, when, ttype, models, info)
            st.markdown(f'<span class="muted">Suggested category</span> &nbsp;<b style="color:#fff">{esc(cat)}</b> &nbsp;'
                        f'<span class="chip {"yellow" if review else ""}">{conf:.0%} confident{", please check" if review else ""}</span>',
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
        st.markdown('<p class="muted">Use a CSV with columns <code>date, description, amount</code> (optional <code>type</code>), '
                    'or your bank\'s export with <code>Date, Narration, Withdrawal Amt., Deposit Amt.</code>. '
                    'Every row is categorised automatically.</p>', unsafe_allow_html=True)
        file = st.file_uploader("Statement CSV", type=["csv"])
        if file and st.button("Import transactions", type="primary"):
            with st.spinner("Categorising..."):
                try:
                    res = S.import_csv(file, user, models, info)
                    st.success(f"Imported {res['imported']:,} of {res['rows_in']:,} rows. {res['needs_review']} need review.")
                    s = pd.Series(res["by_category"]).sort_values()
                    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[CAT_COLORS.get(c, MUTED) for c in s.index]))
                    chart(fig, 40 + 26 * len(s), legend=False)
                except Exception as e:
                    st.error(str(e))


def page_subscriptions(user, models, info):
    tx = S.load_transactions(user)
    title("Subscriptions", tx, meta="Recurring payments &middot; autopay is simulated, no money moves")
    for m, d, a in S.run_autopay(user):
        st.toast(f"Autopay recorded: {m} {S.inr(a)} on {d:%d %b}")
    with st.spinner("Looking for recurring payments..."):
        S.detect_subscriptions(user, models, info)
    subs = S.get_subscriptions(user)
    if subs.empty:
        return empty_panel("No recurring payments found", "Recurring payments show up after at least two payments to the same "
                           "merchant.", user, models, info, key="sub_demo")
    monthly = (subs["avg_amount"] * 30.4 / subs["frequency_days"].clip(lower=1)).sum()
    tiles([("Recurring payments", f"{len(subs)}", "", f"{int((subs['autopay'] == 1).sum())} on autopay"),
           ("Per month", S.inr(monthly), "neg", "Average across all cycles"),
           ("Per year", S.inr(subs["yearly_cost"].sum()), "neg", "If nothing changes")])
    with panel():
        phead("Detected subscriptions", f"{len(subs)} active")
        widths = [2.3, 1.1, 1.0, 1.4, 1.1, 0.8, 0.8, 2.4]
        hdr = st.columns(widths)
        for h, t in zip(hdr, ["Merchant", "Amount", "Every", "Next due", "Per year", "Match", "Auto", "Is this recurring?"]):
            h.markdown(f'<span class="cap" style="font-size:.64rem">{t}</span>', unsafe_allow_html=True)
        today = pd.Timestamp(date.today())
        for r in subs.itertuples():
            col = st.columns(widths, vertical_alignment="center")
            col[0].markdown(f'<div class="nm" style="color:#fff;font-weight:700">{esc(r.merchant.title())}</div>'
                            f'<div class="muted" style="font-size:.78rem">{esc(r.category)}</div>', unsafe_allow_html=True)
            col[1].markdown(f'<b style="color:#fff">{S.inr(r.avg_amount)}</b>', unsafe_allow_html=True)
            col[2].write(f"{r.frequency_days:.0f} days")
            due = pd.Timestamp(r.next_due_date)
            col[3].markdown(f"{due:%d %b}" + (' &nbsp;<span class="chip pink">Overdue</span>' if due < today else ""),
                            unsafe_allow_html=True)
            col[4].write(S.inr(r.yearly_cost))
            col[5].write(f"{r.confidence:.0%}")
            ap = col[6].toggle("Autopay", value=bool(r.autopay), key=f"ap_{r.id}", label_visibility="collapsed")
            if ap != bool(r.autopay):
                S.set_subscription(r.id, autopay=int(ap))
                st.toast(f"Autopay {'turned on' if ap else 'turned off'} for {r.merchant.title()}.")
                st.rerun()
            if r.status == "confirmed":
                col[7].markdown('<span class="chip">Confirmed</span>', unsafe_allow_html=True)
            else:
                b1, b2 = col[7].columns(2)
                if b1.button("Confirm", key=f"c_{r.id}", type="primary"):
                    S.set_subscription(r.id, status="confirmed")
                    st.toast(f"Confirmed. +{S.XP_RULES['confirmed_sub']} XP")
                    st.rerun()
                if b2.button("Dismiss", key=f"d_{r.id}"):
                    S.set_subscription(r.id, status="dismissed", autopay=0)
                    st.rerun()


def page_forecast(user, models, info):
    tx = S.load_transactions(user)
    title("Forecast", tx)
    fc = S.forecast_next_month(user, models, info)
    if fc is None:
        return empty_panel("Not enough history yet", "Forecasts need at least 3 complete months of transactions.",
                           user, models, info, key="fc_demo")
    tm = pd.Period(fc.attrs["target_month"]).strftime("%B %Y")
    sv = S.savings_forecast(user, fc)
    tiles([("Expected spend", S.inr(sv["spend"]), "neg", f"{tm}, all categories"),
           ("Expected income", S.inr(sv["income"]), "pos", "Average of the last 3 months"),
           ("Expected savings", S.inr(sv["savings"]), "pos" if sv["savings"] >= 0 else "neg", "Income minus spend"),
           ("Planned investments", S.inr(sv["investments"]), "cyan", "SIPs and deposits")])
    t1, t2 = st.tabs(["Forecast", "Budget"])
    with t1, panel():
        phead(f"Predicted spend, {tm}", "10th to 90th percentile range", "dim")
        f = fc[fc["forecast"] > 0].sort_values("forecast")
        fig = go.Figure(go.Bar(y=f["category"], x=f["forecast"], orientation="h",
                               marker_color=[CAT_COLORS.get(c, MUTED) for c in f["category"]],
                               error_x=dict(type="data", symmetric=False, array=f["high"] - f["forecast"],
                                            arrayminus=f["forecast"] - f["low"], color="#6B7766", thickness=1.2, width=4),
                               name="Forecast", hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
        fig.add_scatter(y=f["category"], x=f["avg_3m"], mode="markers", name="Last 3-month average",
                        marker=dict(color="#FFFFFF", symbol="line-ns-open", size=16, line=dict(width=2.5)),
                        hovertemplate="%{y}, 3-month average: ₹%{x:,.0f}<extra></extra>")
        fig.update_xaxes(showgrid=True, gridcolor="#1E261C", tickprefix="₹")
        chart(fig, 60 + 30 * len(f))
        t = fc.rename(columns={"category": "Category", "forecast": "Forecast", "low": "Low", "high": "High",
                               "last_month": "Last month", "avg_3m": "3-month average"})
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(c, format="₹%.0f") for c in ["Forecast", "Low", "High", "Last month", "3-month average"]})
    with t2, panel():
        rb = S.recommend_budget(user, fc)
        phead("Recommended budget", "Essentials +5% buffer", "dim")
        if rb.attrs["goal_need"] > 0:
            st.info(f"Your goals need about {S.inr(rb.attrs['goal_need'])} a month. "
                    + (f"Dining, shopping, entertainment and travel are cut by {rb.attrs['cut']:.0%} to close a "
                       f"{S.inr(rb.attrs['shortfall'])} gap." if rb.attrs["cut"] > 0 else "Your forecast savings already cover that."))
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


def budget_month(tx):
    """Current month if it has spending, else the latest month with data."""
    cur = pd.Timestamp(date.today()).to_period("M")
    debits = tx[tx["type"] == "debit"]
    return cur if (debits["date"].dt.to_period("M") == cur).any() else latest_month(tx)


def page_goals(user, models, info):
    tx = S.load_transactions(user)
    goals = S.get_goals(user)
    top_l, top_r = st.columns([4, 1], vertical_alignment="center")
    with top_l:
        title("Goals", tx)
    with top_r, st.container(key="setgoal"), st.popover("+ Set a goal", width="stretch"):
        with st.form("goal"):
            name = st.text_input("Goal", "Rainy-day fund")
            target = st.number_input("Target amount (₹)", min_value=1000.0, value=100000.0, step=5000.0, format="%.0f")
            saved = st.number_input("Saved so far (₹)", min_value=0.0, value=0.0, step=1000.0, format="%.0f")
            when = st.date_input("Target date", date.today() + timedelta(days=365))
            if st.form_submit_button("Create goal", type="primary", width="stretch"):
                S.add_goal(user, name, target, saved, when)
                st.toast(f"Goal created. +{S.XP_RULES['goal']} XP")
                st.rerun()
    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        if goals.empty:
            empty_panel("No goals yet", "Use + Set a goal to create one. Progress, milestones and a finish date appear here.")
        else:
            pick = st.selectbox("Goal", goals["name"].tolist(), label_visibility="collapsed") if len(goals) > 1 else goals["name"].iat[0]
            g = goals[goals["name"] == pick].iloc[0]
            pct = min(g.saved_amount / g.target_amount, 1.0)
            fc = S.forecast_next_month(user, models, info)
            monthly = S.savings_forecast(user, fc)["savings"] if fc is not None else 0
            msg, ok = S.goal_feasibility(g.target_amount, g.saved_amount, monthly / max(len(goals), 1), g.target_date)
            with panel():
                a, b = st.columns([1.35, 1], vertical_alignment="center")
                a.markdown(f'<div class="cap">Total saved</div><div style="font-weight:800;font-size:3.1rem;color:#fff;line-height:1.1;'
                           f'letter-spacing:-.03em">{S.inr(g.saved_amount)}</div><div class="muted" style="font-weight:700">/ '
                           f'{S.inr(g.target_amount)} INR</div><div style="margin-top:.8rem"><span class="chip">{esc(g["name"])}</span> '
                           f'<span class="chip dim">by {pd.Timestamp(g.target_date):%d %b %Y}</span></div>', unsafe_allow_html=True)
                b.markdown(f'<div style="display:flex;justify-content:center">{ring_svg(pct, 190, LIME, 40, f"{pct:.0%}", "of target")}</div>',
                           unsafe_allow_html=True)
                steps = [0.25, 0.5, 0.75, 1.0]
                st.markdown('<div class="ms">' + "".join(
                    f'<div class="{"on" if pct >= s_ else ""}"><div class="cap">{s_:.0%}</div><b>{S.inr(g.target_amount * s_)}</b></div>'
                    for s_ in steps) + "</div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1.popover("Save today", width="stretch"):
                    add = st.number_input("Add to this goal (₹)", min_value=0.0, value=1000.0, step=500.0, format="%.0f", key="add_amt")
                    if st.button("Add", type="primary", key="add_btn", width="stretch"):
                        S.update_goal_saved(g.id, g.saved_amount + add)
                        st.toast(f"Added {S.inr(add)} to {g['name']}.")
                        st.rerun()
                with c2.popover("Adjust target", width="stretch"):
                    nt = st.number_input("New target (₹)", min_value=1000.0, value=float(g.target_amount), step=5000.0, format="%.0f")
                    if st.button("Save target", type="primary", width="stretch"):
                        with S.conn() as cn:
                            cn.execute("UPDATE goals SET target_amount=? WHERE id=?", (float(nt), int(g.id)))
                        st.rerun()
                    if st.button("Delete goal", width="stretch"):
                        S.delete_goal(g.id)
                        st.rerun()
                st.markdown(f'<p class="muted" style="margin-top:.8rem">{esc(msg)}</p>', unsafe_allow_html=True)
        with panel(height=300):
            g_ = S.gamification(user)
            phead("Saving months", f"{int((g_['monthly'] > 0).sum())} of {len(g_['monthly'])}")
            ms = g_["monthly"].sort_index(ascending=False).head(6)
            st.markdown("".join(
                f'<div class="act"><div><div class="nm">{p.strftime("%B %Y")}</div><div class="mt">Income minus spending</div></div>'
                f'<span class="am {"pos" if v >= 0 else "neg"}">{"+" if v >= 0 else "-"}{S.inr(abs(v))}</span></div>'
                for p, v in ms.items()) or '<div class="muted">No complete months yet.</div>', unsafe_allow_html=True)
    with right:
        bm = budget_month(tx) if not tx.empty else None
        phead(f"Budgets, {bm.strftime('%B %Y')}" if bm is not None else "Budgets", "3 active")
        saved_b = S.get_budget(user)
        fc = S.forecast_next_month(user, models, info)
        rec = S.recommend_budget(user, fc).set_index("category")["recommended"].to_dict() if fc is not None else {}
        spent = tx[(tx["type"] == "debit") & (tx["date"].dt.to_period("M") == bm)].groupby("category")["amount"].agg(["sum", "size"]) \
            if bm is not None else pd.DataFrame()
        cards = ""
        for cat, color in [("Food & Dining", PINK), ("Shopping", CYAN), ("Entertainment", YELLOW)]:
            budget = float(saved_b.get(cat) or rec.get(cat) or 0)
            s_ = float(spent["sum"].get(cat, 0)) if len(spent) else 0.0
            n_ = int(spent["size"].get(cat, 0)) if len(spent) else 0
            p_ = s_ / budget if budget else 0
            cards += (f'<div class="bud"><div style="min-width:0"><div class="t">{esc(cat)} budget</div>'
                      f'<div class="bar"><i style="width:{min(p_, 1) * 100:.0f}%;background:{color}"></i></div>'
                      f'<span class="chip dim">{esc(cat.upper())}</span> <span class="muted" style="font-size:.76rem">{n_} payments</span></div>'
                      f'<div class="amt">{S.inr(s_)}<small>/ {S.inr(budget)}</small></div>'
                      f'<div>{ring_svg(min(p_, 1), 64, color, 18, f"{p_:.0%}", width=10)}</div></div>')
        st.markdown(cards, unsafe_allow_html=True)
        with panel():
            phead("How budgets are computed")
            st.markdown('<p class="muted" style="margin:0">Budgets start from next month\'s forecast for each category: essentials '
                        'get a 5% buffer, and dining, shopping, entertainment and travel are trimmed only when your goals need more '
                        'savings than the forecast leaves. Edit your own on the Forecast page.</p>', unsafe_allow_html=True)
        if st.button("Edit budgets", width="stretch"):
            go_to("Forecast")


def page_rewards(user, models, info):
    tx = S.load_transactions(user)
    title("Rewards", tx, meta="Levels, streaks and badges &middot; earned from your real activity")
    g = S.gamification(user)
    span = max(g["next_xp"] - g["floor"], 1)
    prog = (g["xp"] - g["floor"]) / span if g["next_level"] else 1.0
    a, b = st.columns([1.5, 1], gap="medium")
    with a:
        to_next = f" &middot; {g['next_xp'] - g['xp']:,} XP to {esc(g['next_level'])}" if g["next_level"] else " &middot; top level reached"
        st.markdown(f"""
<div class="hero">
  <div class="cap">Current level &middot; {g['level_index']} of {g['n_levels']}</div>
  <div class="big" style="font-family:'Archivo Black';font-size:clamp(2.2rem,4.2vw,3.4rem);text-transform:uppercase">{esc(g['level'])}</div>
  <div class="sub">{g['xp']:,} XP{to_next}</div>
  <div style="height:10px;background:rgba(11,14,10,.15);border-radius:99px;margin-top:1rem;overflow:hidden">
    <div style="height:100%;width:{prog * 100:.0f}%;background:#0B0E0A;border-radius:99px"></div></div>
  <div class="tiles">
    <div class="tile"><div class="cap">Saving streak</div><b>{g['streak']} month{'s' if g['streak'] != 1 else ''}</b></div>
    <div class="tile"><div class="cap">Badges</div><b>{g['unlocked']} / {len(g['badges'])}</b></div>
    <div class="tile"><div class="cap">Total XP</div><b>{g['xp']:,}</b></div>
  </div>
</div>""", unsafe_allow_html=True)
    with b, panel(height=330):
        phead("Where your XP came from")
        st.markdown("".join(f'<div class="act"><div class="nm" style="font-weight:600">{esc(k)}</div>'
                            f'<span class="am pos">+{v:,}</span></div>' for k, v in g["parts"].items()), unsafe_allow_html=True)
    phead("Badges", f"{g['unlocked']} unlocked")
    st.markdown('<div class="badges">' + "".join(
        f'<div class="bd {"on" if on else ""}"><div class="ic">{"&#10003;" if on else "?"}</div><b>{esc(n)}</b><small>{esc(d)}</small></div>'
        for n, d, on in g["badges"]) + "</div>", unsafe_allow_html=True)
    with panel():
        phead("How to earn XP")
        r = S.XP_RULES
        st.markdown(f'<p class="muted" style="margin:0">+{r["saving_month"]} for every month you save money, +{r["correction"]} for '
                    f'each category you correct (it teaches the model), +{r["confirmed_sub"]} for each subscription you confirm, '
                    f'+{r["goal"]} for each goal you set and +{r["goal_reached"]} when you reach one. Levels: '
                    + ", ".join(f"{n} ({x:,} XP)" for n, x in S.LEVELS) + ".</p>", unsafe_allow_html=True)


def page_insights(user, models, info):
    tx = S.load_transactions(user)
    title("Insights", tx)
    if tx.empty:
        return empty_panel("No data yet", "Insights appear once there are transactions.", user, models, info, key="in_demo")
    per = S.persona(user, models, info)
    left, right = st.columns([1, 1.8], gap="medium")
    with left, panel(height=590):
        phead("Spending persona", "Clustering")
        if per is None:
            st.caption("The persona model is not trained yet.")
        elif per["cold_start"]:
            st.caption(f"Personas need at least 3 months of data. You have {per['n_months']}.")
        else:
            st.markdown(f'<div class="pt"><h1 style="font-size:2.1rem;color:{LIME}">{esc(per["name"])}</h1></div>'
                        f'<p class="muted" style="margin:.6rem 0">{esc(per["description"])}</p>'
                        f'<div style="border-left:3px solid {PINK};padding:.2rem 0 .2rem .8rem;color:#E4ECDD">{esc(per["tip"])}</div>',
                        unsafe_allow_html=True)
            u_, a_ = per["user"], per["avg"]
            st.markdown('<div class="cap" style="margin:1.4rem 0 .3rem">You vs a typical member</div>' + "".join(
                f'<div class="act"><div><div class="nm">{n}</div><div class="mt">Typical: {t}</div></div><span class="am plain">{v}</span></div>'
                for n, t, v in [("Savings rate", f"{a_['savings_rate']:.0%}", f"{u_['savings_rate']:.0%}"),
                                ("Average payment", S.inr(a_["avg_txn_size"]), S.inr(u_["avg_txn_size"])),
                                ("Payments per month", f"{a_['txns_per_month']:.0f}", f"{u_['txns_per_month']:.0f}"),
                                ("Month-to-month swing", f"{a_['monthly_cv']:.0%}", f"{u_['monthly_cv']:.0%}")]),
                unsafe_allow_html=True)
    with right, panel(height=590):
        phead("Your category shares vs your persona")
        if per and not per["cold_start"]:
            shares = pd.DataFrame({"category": EXPENSE_CATEGORIES, "You": [per["user"][f"share_{c}"] for c in EXPENSE_CATEGORIES],
                                   f"Typical {per['name']}": [per["avg"][f"share_{c}"] for c in EXPENSE_CATEGORIES]})
            shares = shares[(shares.iloc[:, 1:] > 0.005).any(axis=1)].sort_values("You")
            fig = px.bar(shares.melt("category", var_name="who", value_name="share"), y="category", x="share", color="who",
                         barmode="group", orientation="h", color_discrete_sequence=[LIME, "#3A4636"])
            fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor="#1E261C")
            fig.update_traces(hovertemplate="%{y}: %{x:.1%}<extra></extra>")
            fig.update_layout(bargap=0.3, bargroupgap=0.05)
            chart(fig, 500)
    p = S.patterns(tx)
    if not p:
        return
    subs = S.get_subscriptions(user)
    wt = S.weekend_test(tx)
    ratio = p["weekend_daily"] / max(p["weekday_daily"], 1) - 1
    tiles([("Average weekend day", S.inr(p["weekend_daily"]), "", f"{ratio:+.0%} compared with weekdays"),
           ("Average weekday", S.inr(p["weekday_daily"]), "", "Monday to Friday"),
           ("Recurring share", f"{S.recurring_share(tx, subs):.0%}", "cyan", "Payments to detected subscriptions"),
           ("Weekend test", "Different" if wt and wt["significant"] else "No difference", "yellow",
            f"Mann-Whitney U, p = {wt['p_value']:.3f}" if wt else "Not enough days")])
    a, b = st.columns(2, gap="medium")
    with a, panel():
        phead("Average spend by weekday")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wk = p["by_weekday"].reindex(order).fillna(0)
        fig = go.Figure(go.Bar(x=[d[:3] for d in order], y=wk.values, marker_color=[PURPLE if i < 5 else PINK for i in range(7)],
                               hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
        chart(fig, 270, legend=False)
    with b, panel():
        phead("Fastest-growing categories", "Last 3 vs previous 3 months", "dim")
        if len(p["growth"]):
            gr = p["growth"].head(6).sort_values()
            fig = go.Figure(go.Bar(x=gr.values, y=gr.index, orientation="h", marker_color=[PINK if v > 0 else LIME for v in gr.values],
                                   hovertemplate="%{y}: %{x:+.0%}<extra></extra>"))
            fig.update_xaxes(tickformat="+.0%")
            chart(fig, 270, legend=False)
        else:
            st.caption("Needs 6 months of history.")
    with panel():
        phead("Unusual transactions", "Above Q3 + 3 x IQR", "pink")
        un = S.unusual_transactions(tx)
        if un.empty:
            st.caption("Nothing unusual in the last 3 months.")
        else:
            st.dataframe(un.assign(date=un["date"].dt.date), hide_index=True, width="stretch", column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"), "description": "Description",
                "category": "Category", "amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
                "times_typical": st.column_config.NumberColumn("Times your usual", format="%.1fx")})


def page_models(user, models, info):
    title("Models", meta="Active versions &middot; held-out test metrics &middot; synthetic and simulated data")
    if not info:
        return empty_panel("No models registered", "Run python -m src.data_prep and python -m src.train, then reload.")
    fb = S.feedback_stats()
    tiles([("Categorised", f"{fb['n']:,}", "", "Transactions, all users"),
           ("Corrected by users", f"{fb['correction_rate']:.1%}", "pos", "Saved for future retraining"),
           ("Flagged for review", f"{fb['review_rate']:.1%}", "yellow", "Below the confidence threshold"),
           ("Average confidence", f"{fb['avg_confidence']:.2f}", "cyan", "Categorizer")])
    st.markdown(f'<p class="muted">Experiment history: <a href="{MLFLOW_URL}" target="_blank" style="color:{LIME}">MLflow</a> '
                f'(start it with <code>python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db</code>).</p>',
                unsafe_allow_html=True)
    figs = {"categorizer": ["categorizer_confusion_matrix.png", "categorizer_per_class_f1.png", "categorizer_confidence_curve.png"],
            "subscription": ["subscription_confusion_matrix.png", "subscription_roc_pr.png", "subscription_shap.png"],
            "forecaster": ["forecaster_baseline_vs_model.png", "forecaster_pred_vs_actual.png", "forecaster_residuals.png"],
            "persona": ["persona_pca.png", "persona_k_selection.png", "persona_centroid_heatmap.png"]}
    names = {"categorizer": "Categorizer", "subscription": "Subscription detector", "forecaster": "Forecaster", "persona": "Persona"}
    tabs = st.tabs([names.get(n, n) for n in info])
    for tab, (name, e) in zip(tabs, info.items()):
        with tab:
            tiles([("Version", f"v{e['version']}", "pos", esc(e["file"])), ("Algorithm", e["algorithm"], "", ""),
                   ("Primary metric", e["primary_metric"], "cyan", ""), ("Trained", e["trained_at"][:10], "", f"git {esc(e.get('git_commit', '?'))}")])
            with panel():
                phead("Test metrics against the baseline")
                mt = pd.DataFrame({"Model": e["test_metrics"], "Baseline": e.get("baseline_metrics", {})})
                mt.index = [i.replace("_", " ").capitalize() for i in mt.index]
                st.dataframe(mt.map(lambda v: "-" if v is None or (isinstance(v, float) and pd.isna(v))
                                    else (f"{v:.4f}" if isinstance(v, (int, float)) else str(v))), width="stretch")
            comp = RESULTS / f"model_comparison_{name}.csv"
            if comp.exists():
                with panel():
                    phead("All candidates")
                    st.dataframe(pd.read_csv(comp).round(4), hide_index=True, width="stretch")
            cols = st.columns(3, gap="medium")
            for col, f in zip(cols, figs.get(name, [])):
                if (RESULTS / f).exists():
                    with col, panel():
                        st.image(str(RESULTS / f), width="stretch")


# ====================================================================================== authentication
AUTH_CSS = f"""
<style>
section[data-testid="stSidebar"], button[data-testid="stExpandSidebarButton"], div[data-testid="collapsedControl"] {{display: none;}}
.block-container {{max-width: 1200px; padding-top: 2.6rem;}}
.ah {{position: relative; background: linear-gradient(135deg, #A4F94F 0%, #7BE433 55%, #5FCB1F 100%); border-radius: 30px;
      padding: 2.3rem 2.3rem 2rem; color: #0B0E0A; min-height: 610px; display: flex; flex-direction: column;
      box-shadow: 0 0 0 1px rgba(140,245,66,.4), 0 30px 80px -30px rgba(140,245,66,.6); overflow: hidden;}}
.ah .brand {{display: flex; align-items: center; gap: .6rem; font-family: "Archivo Black"; font-size: 1.15rem; text-transform: uppercase;}}
.ah .mk {{width: 34px; height: 34px; border-radius: 50%; background: #0B0E0A; color: {LIME}; display: grid; place-items: center; font-size: 1.1rem;}}
.ah h1 {{font-family: "Archivo Black"; font-size: clamp(2.6rem, 5vw, 3.9rem); line-height: .98; text-transform: uppercase; margin: 2.2rem 0 1rem;
         padding: 0; color: #0B0E0A; letter-spacing: -.01em;}}
.ah .lead {{font-weight: 600; font-size: 1.02rem; max-width: 44ch; color: rgba(11,14,10,.8);}}
.pbk {{background: #0B0E0A; border-radius: 20px; padding: .5rem 1.1rem; margin-top: auto; box-shadow: 0 20px 40px -20px rgba(0,0,0,.6);}}
.pbk .r {{display: grid; grid-template-columns: minmax(0,1fr) auto; gap: .25rem 1rem; padding: .62rem 0; border-bottom: 1px solid #1B2219;}}
.pbk .r:last-child {{border-bottom: none;}}
.pbk .n {{color: #C9D3C2; font-size: .82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
.pbk .c {{font-size: .8rem; font-weight: 700; white-space: nowrap;}}
.pbk .b {{grid-column: 1 / -1; height: 4px; background: #1E261C; border-radius: 99px; overflow: hidden;}}
.pbk .b i {{display: block; height: 100%; border-radius: 99px;}}
.ah .note {{font-size: .78rem; font-weight: 600; color: rgba(11,14,10,.65); margin-top: .7rem;}}
.side {{padding: 3.4rem .3rem 0 .6rem;}}
.side h2 {{font-family: "Archivo Black"; font-size: 2.3rem; text-transform: uppercase; color: #fff; margin: 0 0 .3rem; padding: 0;}}
.side p {{color: {MUTED}; margin: 0 0 1.3rem;}}
div[data-testid="stForm"] {{background: {PANEL}; border: 1px solid {LINE}; border-radius: 22px; padding: 1.4rem 1.4rem .6rem;}}
.afoot {{color: #5E6A59; font-size: .76rem; margin-top: 1rem; line-height: 1.5;}}
</style>"""

PREVIEW = ["UPI/SWIGGY/402291/Payment", "ACH D- HDFC BANK EMI 88213004", "NEFT-HOUSE RENT-NOV", "POS 4471 DMART PUNE",
           "UPI/RAHUL SHARMA/551204/paid"]


def passbook_preview(models, info):
    rows = []
    for d in PREVIEW:
        if "categorizer" in models:
            cat, conf, review = S.suggest_category(d, 500, date.today(), "debit", models, info)
        else:
            cat, conf, review = "-", 0.0, False
        color = YELLOW if review else LIME
        label = f"Needs review {conf:.0%}" if review else f"{cat} {conf:.0%}"
        rows.append(f'<div class="r"><div class="n">{esc(d)}</div><div class="c" style="color:{color}">{esc(label)}</div>'
                    f'<div class="b"><i style="width:{conf * 100:.0f}%;background:{color}"></i></div></div>')
    return "".join(rows)


def sign_in(username):
    st.session_state["user"] = username


def page_auth(models, info):
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"""
<div class="ah">
  <div class="brand"><div class="mk">&uarr;</div>Smart Finance Tracker</div>
  <h1>Your bank statement, sorted.</h1>
  <div class="lead">UPI, NEFT and card narrations go in. Categories, subscriptions, a forecast and your saving streak come out.</div>
  <div class="pbk">{passbook_preview(models, info)}</div>
  <div class="note">Live output of the categorisation model. Unsure guesses are flagged for review.</div>
</div>""", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="side"><h2>Welcome</h2><p>Sign in, or create an account to start your first saving streak.</p></div>',
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
        st.markdown('<div class="afoot">Passwords are stored as salted PBKDF2-SHA256 hashes in a local SQLite file. This is a student '
                    'project, so avoid uploading real statements on a shared computer.</div>', unsafe_allow_html=True)


# ====================================================================================== AI assistant
SUGGESTIONS = ["Where did most of my money go last month?", "Which subscriptions cost me the most per year?",
               "Am I on track for my goals?", "How could I save ₹2,000 more next month?"]


@st.dialog("Ask your money", width="large")
def chat_dialog(username, models, info):
    key, model = S.groq_settings()
    if not key:
        st.markdown('<p class="muted">The assistant needs a free Groq API key. Create one at console.groq.com, then put it '
                    'in <code>.streamlit/secrets.toml</code> on your computer:</p>', unsafe_allow_html=True)
        st.code('GROQ_API_KEY = "gsk_your_key_here"', language="toml")
        st.markdown('<p class="muted">Restart the app afterwards. The file is in .gitignore, so the key never reaches GitHub. '
                    'Running with Docker? Put the same line, without spaces or quotes, in a <code>.env</code> file next to docker-compose.yml.</p>',
                    unsafe_allow_html=True)
        return
    hist = st.session_state.setdefault("chat", [])
    box = st.container(height=420, key="chatbox")
    question = None
    with box:
        if not hist:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.markdown("Hi! I can answer questions about **your** spending, bills, forecast and goals, using the figures "
                            "in this app. Try one of these:")
            for i, s_ in enumerate(SUGGESTIONS):
                if st.button(s_, key=f"sug{i}", width="stretch"):
                    question = s_
        for m in hist:
            with st.chat_message(m["role"], avatar=":material/auto_awesome:" if m["role"] == "assistant" else ":material/person:"):
                st.markdown(m["content"])
    typed = st.chat_input("Ask about your spending, bills or goals")
    question = typed or question
    if question:
        hist.append({"role": "user", "content": question})
        with box:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(question)
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                try:
                    with st.spinner("Reading your numbers..."):
                        context = S.finance_context(username, models, info)
                    reply = st.write_stream(S.ask_groq_stream(hist, context, key, model))
                    hist.append({"role": "assistant", "content": reply if isinstance(reply, str) else "".join(reply)})
                except S.GroqError as e:
                    hist.pop()
                    st.error(str(e))
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.markdown(f'<span class="muted" style="font-size:.76rem !important">Answers use your figures from this app, sent to Groq '
                f'({esc(model)}). General guidance, not professional financial advice.</span>', unsafe_allow_html=True)
    if hist and c2.button("Clear chat", width="stretch"):
        st.session_state["chat"] = []
        st.rerun(scope="fragment")


def chat_fab(username, models, info):
    with st.container(key="chatfab"):
        if st.button("Ask AI", icon=":material/chat:", key="aibtn", help="Ask about your money"):
            chat_dialog(username, models, info)


# ====================================================================================== navigation / routing
PAGES = [("Overview", "overview", page_overview, ":material/arrow_upward:"),
         ("Transactions", "transactions", page_transactions, ":material/receipt_long:"),
         ("Subscriptions", "subscriptions", page_subscriptions, ":material/autorenew:"),
         ("Forecast", "forecast", page_forecast, ":material/insights:"),
         ("Goals", "goals", page_goals, ":material/track_changes:"),
         ("Rewards", "rewards", page_rewards, ":material/workspace_premium:"),
         ("Insights", "insights", page_insights, ":material/bar_chart:"),
         ("Models", "models", page_models, ":material/model_training:")]


def sidebar(username, user, models, info):
    g = S.gamification(username)
    span = max(g["next_xp"] - g["floor"], 1)
    prog = (g["xp"] - g["floor"]) / span if g["next_level"] else 1.0
    name = user.get("full_name") or username
    with st.sidebar:
        st.markdown(f"""
<div class="lvl">
  <div class="cap" style="font-size:.64rem">Current level</div>
  <div class="name">{esc(g['level'])}</div>
  <div class="xpbar"><i style="width:{prog * 100:.0f}%"></i></div>
  <div class="row"><span>{g['xp']:,} / {g['next_xp']:,} XP</span><span class="chip">{g['level_index']}/{g['n_levels']}</span></div>
  <hr>
  <div class="row" style="margin:0"><span>{g['streak']}-month streak</span><span>{g['unlocked']} badges</span></div>
</div>
<div class="prof"><div class="ava"><div>{esc(name[:1].upper())}</div></div><div><b>{esc(name)}</b><span>@{esc(username)}</span></div></div>
""", unsafe_allow_html=True)
        with st.popover("Account", width="stretch"):
            if st.button("Load demo data", width="stretch", help="Replaces your data with a generated 24-month history"):
                load_demo(username, models, info)
            if st.button("Sign out", type="primary", width="stretch"):
                st.session_state.clear()
                st.rerun()
        if not models:
            st.warning("Models not found. Run python -m src.data_prep and python -m src.train.")


def main():
    _PANELS[0] = 0
    S.init_db()
    models, info = get_models()
    username = st.session_state.get("user")
    user = S.get_user(username) if username else None
    if not user:
        st.session_state.pop("user", None)
        return page_auth(models, info)
    st.logo("static/logo.svg", size="large")
    pages = []
    for i, (t, u, f, icon) in enumerate(PAGES):
        p = st.Page(lambda f=f: f(username, models, info), title=t, url_path=u, icon=icon, default=(i == 0))
        NAV[t] = p
        pages.append(p)
    nav = st.navigation(pages, position="sidebar")
    sidebar(username, user, models, info)
    chat_fab(username, models, info)
    nav.run()


main()
