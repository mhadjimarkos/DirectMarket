# DirectMarket — Pro Dashboard
# Clean structure, tabs, KPIs, watchlist, downloads, spinners, better empty states.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from urllib.parse import urlparse

@st.cache_data(show_spinner=False, ttl=60*60)
def get_logo_url(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}

    url = (info or {}).get("logo_url")
    if url and isinstance(url, str):
        return url  # yfinance logo is already sized well

    website = (info or {}).get("website")
    if website and isinstance(website, str):
        try:
            host = urlparse(website if website.startswith("http") else "https://" + website).netloc.split(":")[0]
            if host:
                return f"https://logo.clearbit.com/{host}?size=40"  # <= force 40px
        except Exception:
            pass
    return ""


    # 2) Derive company domain and use Clearbit
    website = (info or {}).get("website") or (info or {}).get("longBusinessSummaryWebsite")
    if website and isinstance(website, str):
        try:
            host = urlparse(website if website.startswith("http") else "https://" + website).netloc
            host = host.split(":")[0]
            if host:
                return f"https://logo.clearbit.com/{host}"
        except Exception:
            pass

    # 3) Nothing found → empty string (table will show blank)
    return ""

def add_logos(df: pd.DataFrame, col_name="ticker") -> pd.DataFrame:
    out = df.copy()
    out["logo"] = out[col_name].astype(str).apply(get_logo_url)
    return out

# ---------- CONFIG ----------
st.set_page_config(
    page_title="DirectMarket — Social Sentiment & Price",
    page_icon="",
    layout="wide",
    menu_items={"About": "DirectMarket — see what the crowd feels before the market moves."}
)

# ---------- STYLES ----------
try:
    with open("assets/style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ---------- CONSTANTS ----------
DEFAULT_TICKERS = ("AAPL","TSLA","NVDA","MSFT","AMZN","META","GOOGL","AMD","NFLX","PLTR","SPY","QQQ","SMH")
PULSE_NOTE = "Pulse is a 0–100 demo score; connect Reddit keys to enable live mentions."

# ---------- HELPERS ----------
@st.cache_data(show_spinner=False, ttl=60*15)
def get_price_data(ticker: str, days: int = 7) -> pd.DataFrame:
    """Live OHLCV via yfinance (cached)."""
    try:
        interval = "1h" if days <= 30 else "1d"
        period = f"{days}d"
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"index": "Datetime"})
        if "Datetime" in df.columns and hasattr(df["Datetime"].iloc[0], "tzinfo") and df["Datetime"].dt.tz is not None:
            df["Datetime"] = df["Datetime"].dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

def demo_trending_top(n: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = np.array(DEFAULT_TICKERS)
    rng.shuffle(tickers)
    picks = tickers[:n]
    mentions = rng.integers(120, 3500, size=n)
    pulse = np.clip(rng.normal(55, 12, size=n), 0, 100)
    df = pd.DataFrame({"ticker": picks, "mentions": mentions, "pulse": pulse})
    return df.sort_values(["pulse","mentions"], ascending=[False, False]).reset_index(drop=True)

def demo_sentiment_series(start: pd.Timestamp, end: pd.Timestamp, freq: str = "h", seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt_index = pd.date_range(start=start, end=end, freq=freq)
    base = 50 + 15*np.sin(np.linspace(0, 4*np.pi, len(dt_index)))
    noise = rng.normal(0, 6, size=len(dt_index))
    return pd.DataFrame({"Datetime": dt_index, "sentiment": np.clip(base + noise, 0, 100)})

def kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"<div class='kpi'><h3>{label}</h3><div class='val'>{value}</div><div class='small'>{sub}</div></div>",
        unsafe_allow_html=True,
    )

def fmt_money(x): 
    try: return f"${float(x):,.2f}"
    except: return "–"

def fmt_pct(x): 
    try: return f"{float(x):+.2f}%"
    except: return "–"

def price_pulse_chart(price_df: pd.DataFrame, senti_df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df["Datetime"], y=price_df["Close"],
        mode="lines", name=f"{ticker} Price"
    ))
    fig.add_trace(go.Scatter(
        x=senti_df["Datetime"], y=senti_df["sentiment"],
        mode="lines", name="Pulse (0–100)", yaxis="y2"
    ))
    fig.update_layout(
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(title="Time"),
        yaxis=dict(title="Price", side="left", gridcolor="rgba(255,255,255,.06)"),
        yaxis2=dict(title="Pulse (0–100)", overlaying="y", side="right", rangemode="tozero",
                    gridcolor="rgba(255,255,255,.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ---------- SESSION ----------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL","NVDA","TSLA","MSFT"]

# ---------- HERO / TOP NAV ----------
nv1, nv2, nv3, nv4 = st.columns([6,1.2,1.2,1.6])
with nv1:
    st.markdown(
        "<div class='hero'>"
        "<img src='assets/logo.svg' width='42' style='opacity:.95'/>"
        "<div>"
        "<h2>DirectMarket <span class='mp-badge'>PRO</span></h2>"
        "<div class='sub'>Real-time crowd sentiment meets live price action.</div>"
        "</div></div>",
        unsafe_allow_html=True
    )
with nv2: st.link_button("About", "https://github.com", use_container_width=True)
with nv3: st.link_button("Pricing", "https://github.com", use_container_width=True)
with nv4: st.link_button("Get updates", "https://www.beehiiv.com", use_container_width=True)
st.markdown("---")

# ---------- SIDEBAR ----------
st.sidebar.markdown("### ⚙️ Settings")
mode = st.sidebar.selectbox("Data mode", ["Demo (no API keys)"], index=0)
days = st.sidebar.slider("Window (days)", 3, 90, 7)
topn = st.sidebar.slider("Trending list size", 5, 20, 10)
st.sidebar.markdown("---")
st.sidebar.markdown("**Watchlist**")
w_add = st.sidebar.text_input("Add ticker", placeholder="e.g., AMD").upper().strip()
c1, c2 = st.sidebar.columns([1,1])
with c1:
    if st.sidebar.button("Add"):
        if w_add and w_add not in st.session_state.watchlist:
            st.session_state.watchlist.append(w_add)
with c2:
    if st.sidebar.button("Clear"):
        st.session_state.watchlist = []
st.sidebar.caption("Tip: add 3–10 tickers you care about.")
st.sidebar.markdown("---")
ad_toggle = st.sidebar.checkbox("Show affiliate banner", value=True)

# ---------- TABS ----------
tab_overview, tab_watchlist, tab_about = st.tabs(["Overview", "Watchlist", "About"])

# ===== OVERVIEW =====
with tab_overview:
    # --- Trending section (full width, fixed height) ---
    st.subheader("🔥 Trending now")

box = st.container(border=True)
with box:
    trending = demo_trending_top(n=topn)
    trending = add_logos(trending, col_name="ticker")

    st.dataframe(
    trending[["logo", "ticker", "mentions", "pulse"]],
    width="stretch",
    height=420,
    hide_index=True,
    column_config={
        "logo": st.column_config.ImageColumn("Company", width="small"),
        "ticker": st.column_config.TextColumn("ticker"),
        "mentions": st.column_config.NumberColumn("mentions", format="%,d"),
        # nice in-cell bar without pandas Styler
        "pulse": st.column_config.ProgressColumn(
            "pulse",
            min_value=0,
            max_value=100,
            format="%.0f",
        ),
    },
)


st.download_button(
    "Download CSV",
    data=trending.drop(columns=["logo"]).to_csv(index=False).encode(),
    file_name=f"directmarket_trending_{datetime.now():%Y%m%d}.csv",
    mime="text/csv", use_container_width=True,
)
st.caption(PULSE_NOTE)


st.markdown("### 📊 Price ↔ Sentiment")
cc1, cc2, cc3 = st.columns([3,1,1])
default_ticker = trending["ticker"].iloc[0] if not trending.empty else "AAPL"
with cc1: ticker = st.text_input("Search ticker", value=default_ticker).upper().strip()
with cc2: days_sel = st.selectbox("Window", [7,14,30,60,90], index=0)
with cc3: run_btn = st.button("Analyze", use_container_width=True)

if run_btn and ticker:
    with st.spinner("Fetching price data…"):
        price_df = get_price_data(ticker, days=int(days_sel))

    if price_df.empty:
        st.warning("No price data returned. Try another ticker or change the window.")
    else:
        start_ts = price_df["Datetime"].min()
        end_ts   = price_df["Datetime"].max()
        freq     = "h" if int(days_sel) <= 30 else "D"
        senti_df = demo_sentiment_series(start_ts, end_ts, freq=freq, seed=abs(hash(ticker)) % (2**32))
        senti_df = senti_df.set_index("Datetime").reindex(price_df["Datetime"]).interpolate().reset_index()

        card = st.container(border=True)
        with card:
            st.markdown(f"#### {ticker}")
            st.plotly_chart(price_pulse_chart(price_df, senti_df, ticker), use_container_width=True)

            latest = price_df["Close"].iloc[-1]
            first  = price_df["Close"].iloc[0]
            pct    = 100.0 * (latest - first) / first
            vol    = price_df["Close"].pct_change().std() * 100 * (252**0.5)
            pulse  = float(senti_df["sentiment"].iloc[-1])

            k1, k2, k3 = st.columns(3)
            with k1: kpi("Last Price", fmt_money(latest), f"{fmt_pct(pct)} over {days_sel}d")
            with k2: kpi("Latest Pulse", f"{pulse:,.1f}", "demo series")
            with k3: kpi("Volatility (ann.)", fmt_pct(vol), "rough estimate")



# ===== WATCHLIST =====
with tab_watchlist:
    st.subheader("⭐ Your Watchlist")
    if not st.session_state.watchlist:
        st.info("No tickers yet. Add some from the sidebar.")
    else:
        rows = []
        with st.spinner("Refreshing quotes…"):
            for t in st.session_state.watchlist:
                df = get_price_data(t, days=max(5, days))
                if df.empty:
                    rows.append({"ticker": t, "price": np.nan, "change_%": np.nan, "pulse": np.nan})
                else:
                    latest = df["Close"].iloc[-1]
                    first  = df["Close"].iloc[0]
                    pct    = 100.0 * (latest - first) / first
                    pulse  = float(np.clip(50 + (pct/2), 0, 100))  # simple demo pulse derivation
                    rows.append({"ticker": t, "price": latest, "change_%": pct, "pulse": pulse})

        wl = pd.DataFrame(rows)

        # ⬇️ add logos here
        wl["logo"] = wl["ticker"].apply(get_logo_url)

        # Safe formatters
        def fmt_or_dash(v, fmt):
            if v is None or (not np.isscalar(v)):
                return "–"
            try:
                if pd.isna(v): return "–"
            except Exception:
                pass
            try:
                return fmt(v)
            except Exception:
                return "–"

        disp = wl.copy()
        disp["price"]    = disp["price"].map(lambda v: fmt_or_dash(v, fmt_money))
        disp["change_%"] = disp["change_%"].map(lambda v: fmt_or_dash(v, lambda x: f"{x:+.2f}%"))
        disp["pulse"]    = disp["pulse"].map(lambda v: fmt_or_dash(v, lambda x: f"{x:.1f}"))

        st.dataframe(
            disp[["logo", "ticker", "price", "change_%", "pulse"]],
            width="stretch", hide_index=True,
            column_config={
                "logo": st.column_config.ImageColumn("", help="Company logo", width="small"),
                "ticker": st.column_config.TextColumn("ticker"),
                "price": st.column_config.TextColumn("price"),
                "change_%": st.column_config.TextColumn("change %"),
                "pulse": st.column_config.TextColumn("pulse"),
            },
        )
        st.caption("Change is over the selected window. Pulse is demo unless Live data is connected.")



# ---------- FOOTER ----------
st.markdown("---")
f1, f2 = st.columns([3,2])
with f1:
    st.caption(f"© {datetime.now():%Y} DirectMarket • Demo build")
with f2:
    if ad_toggle:
        st.markdown(
            '<div class="footer"><a href="#" target="_blank">'
            '<strong>Partner Broker</strong> — Start with €0 commission*</a></div>',
            unsafe_allow_html=True
        )
