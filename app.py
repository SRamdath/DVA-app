# Regime-Aware Market Conditions Analyzer (Streamlit app)
# Default pair: SPY + IEF (equity + intermediate Treasuries)
#
# Run from terminal:
#   streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import yfinance as yf
from pandas_datareader import data as pdr
import ruptures as rpt

st.set_page_config(page_title="Regime-Aware Market Analyzer", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def get_monthly_adj_close(ticker: str, start_date: str, end_date=None) -> pd.Series:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    # Robustly extract Adj Close as a SERIES (yfinance can return MultiIndex columns sometimes)
    if isinstance(df.columns, pd.MultiIndex):
        # Often columns look like ('Adj Close', 'SPY') or ('SPY','Adj Close') depending on yfinance version
        if "Adj Close" in df.columns.get_level_values(0):
            s = df["Adj Close"]
        elif "Adj Close" in df.columns.get_level_values(-1):
            s = df.xs("Adj Close", axis=1, level=-1)
        else:
            raise ValueError("Could not find 'Adj Close' in downloaded data.")
        # If still DataFrame (e.g., one column per ticker), pick the first column
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
    else:
        if "Adj Close" not in df.columns:
            raise ValueError("Could not find 'Adj Close' in downloaded data.")
        s = df["Adj Close"]
        # If this is a DataFrame for some reason, pick the first column
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

    s = s.dropna()
    s.name = ticker

    # Month-end series
    s = s.resample("M").last()
    return s

@st.cache_data(show_spinner=False)
def get_fred_monthly(series: list[str], start_date: str, end_date=None) -> pd.DataFrame:
    return pdr.DataReader(series, "fred", start_date, end_date).resample("M").last()

def max_drawdown_logret(logrets: pd.Series) -> float:
    cum = logrets.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return float(dd.min())

def detect_regimes(signal_1d: np.ndarray, pen: float, model: str = "rbf") -> np.ndarray:
    signal = signal_1d.reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(signal)
    bkps = algo.predict(pen=pen)  # includes len(signal)
    labels = np.zeros(len(signal_1d), dtype=int)

    start_i = 0
    reg_id = 0
    for end_i in bkps:
        labels[start_i:end_i] = reg_id
        reg_id += 1
        start_i = end_i
    return labels

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Settings")

default_assets = {"Equity (ticker)": "SPY", "Bond (ticker)": "IEF"}

equity = st.sidebar.text_input("Equity ticker", value=default_assets["Equity (ticker)"]).strip().upper()
bond = st.sidebar.text_input("Bond ticker", value=default_assets["Bond (ticker)"]).strip().upper()

start_date = st.sidebar.text_input("Start date (YYYY-MM-DD)", value="1990-01-01").strip()
end_date = st.sidebar.text_input("End date (YYYY-MM-DD, optional)", value="").strip()
end_dt = end_date if end_date else None

win = st.sidebar.slider("Rolling window (months)", min_value=6, max_value=36, value=12, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Regime Signal Weights")
w_eqvol = st.sidebar.slider("Weight: equity vol", 0.0, 3.0, 1.0, 0.05)
w_bdvol = st.sidebar.slider("Weight: bond vol", 0.0, 3.0, 0.5, 0.05)
w_corr = st.sidebar.slider("Weight: (negative) eq–bond corr", 0.0, 3.0, 0.75, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Change-point detector")
pen = st.sidebar.slider("Penalty (higher → fewer regimes)", 1.0, 30.0, 10.0, 0.5)
cp_model = st.sidebar.selectbox("Cost model", ["rbf", "l2"], index=0)

st.sidebar.markdown("---")
show_macro = st.sidebar.checkbox("Overlay macro context (USREC + inflation)", value=True)

# -----------------------------
# Main
# -----------------------------
st.title("Regime-Aware Market Conditions Analyzer")
st.caption("Price-first regime discovery → macro interpretation → equity/bond regime profiles (progress-report deliverable)")

try:
    with st.spinner("Downloading market data..."):
        eq_px = get_monthly_adj_close(equity, start_date, end_dt)
        bd_px = get_monthly_adj_close(bond, start_date, end_dt)

    # Monthly log returns
    eq_ret = np.log(eq_px).diff()
    eq_ret.name = "eq_ret"
    bd_ret = np.log(bd_px).diff()
    bd_ret.name = "bd_ret"

    # FRED macro
    if show_macro:
        with st.spinner("Downloading macro data (FRED)..."):
            fred_raw = get_fred_monthly(["USREC", "CPIAUCSL", "UNRATE"], start_date, end_dt)
        fred_raw.columns = ["USREC", "CPI", "UNRATE"]
        fred_raw["infl_yoy"] = (fred_raw["CPI"] / fred_raw["CPI"].shift(12) - 1.0) * 100.0
        macro = fred_raw[["USREC", "infl_yoy", "UNRATE"]]
    else:
        macro = pd.DataFrame(index=eq_ret.index)

    df = pd.concat([eq_ret, bd_ret, macro], axis=1).dropna()

    if len(df) < max(win + 24, 60):
        st.warning("Not enough data after merging. Try an earlier start date or check tickers.")
        st.stop()

    # Rolling features
    df["eq_vol"] = df["eq_ret"].rolling(win).std() * np.sqrt(12)
    df["bd_vol"] = df["bd_ret"].rolling(win).std() * np.sqrt(12)
    df["eq_bd_corr"] = df["eq_ret"].rolling(win).corr(df["bd_ret"])

    feat = df[["eq_vol", "bd_vol", "eq_bd_corr"]].dropna()
    feat_z = (feat - feat.mean()) / feat.std()

    # Regime signal (higher = more "stress": higher vol, less negative correlation)
    df_sig = (w_eqvol * feat_z["eq_vol"]) + (w_bdvol * feat_z["bd_vol"]) - (w_corr * feat_z["eq_bd_corr"])
    df_sig.name = "regime_signal"

    work = pd.concat([df.loc[df_sig.index], df_sig], axis=1).dropna()

except Exception as e:
    st.error("Data load failed — full error below:")
    st.exception(e)
    st.stop()

# Detect regimes
labels = detect_regimes(work["regime_signal"].values, pen=pen, model=cp_model)
work["regime"] = labels

# Regime summaries
def ann_mean(x): return float(x.mean() * 12)
def ann_vol(x): return float(x.std() * np.sqrt(12))

summary = (
    work.groupby("regime")
    .agg(
        start=("regime_signal", lambda x: x.index.min()),
        end=("regime_signal", lambda x: x.index.max()),
        n_months=("regime_signal", "size"),
        eq_mean=("eq_ret", ann_mean),
        eq_vol=("eq_ret", ann_vol),
        bd_mean=("bd_ret", ann_mean),
        bd_vol=("bd_ret", ann_vol),
        eq_bd_corr=("eq_ret", lambda x: float(x.corr(work.loc[x.index, "bd_ret"]))),
        eq_mdd=("eq_ret", max_drawdown_logret),
        bd_mdd=("bd_ret", max_drawdown_logret),
    )
    .reset_index()
)

if show_macro and "USREC" in work.columns:
    macro_summary = (
        work.groupby("regime")
        .agg(
            infl_yoy_avg=("infl_yoy", "mean"),
            usrec_share=("USREC", "mean"),
            unrate_avg=("UNRATE", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(macro_summary, on="regime", how="left")

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("Detected regimes over time")
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(work.index, work["regime_signal"], linewidth=1.2)
    ax.set_ylabel("Regime signal (z-scored mix)")
    ax.set_title(f"Regime signal + change-points ({equity} vs {bond})")
    ax.grid(True, alpha=0.2)

    # Shade regimes
    regs = work["regime"].values
    change_idx = np.where(np.diff(regs) != 0)[0] + 1
    bounds = np.r_[0, change_idx, len(regs)]

    for i in range(len(bounds) - 1):
        s = bounds[i]
        e = bounds[i + 1]
        ax.axvspan(work.index[s], work.index[e - 1], alpha=0.15)

    st.pyplot(fig)

    st.caption("Higher signal typically reflects higher volatility and less negative (or more positive) equity–bond correlation.")

with right:
    st.subheader("Regime profiles (summary)")
    display_cols = [
        "regime", "start", "end", "n_months",
        "eq_mean", "eq_vol", "bd_mean", "bd_vol",
        "eq_bd_corr", "eq_mdd", "bd_mdd"
    ]
    if show_macro:
        for c in ["infl_yoy_avg", "usrec_share", "unrate_avg"]:
            if c in summary.columns:
                display_cols.append(c)

    out = summary[display_cols].copy()
    for c in out.columns:
        if c not in ["regime", "start", "end", "n_months"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    st.dataframe(out, use_container_width=True, height=420)

    st.download_button(
        "Download regime summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="regime_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.markdown(
    f"""
- **Model:** rolling-feature regime signal (eq/bond vol + eq–bond corr) + change-point detection (PELT, {cp_model} cost, pen={pen}).  
- **Data:** {equity} and {bond} monthly adjusted close (Yahoo Finance) + USREC/CPI (FRED) for macro overlays.  
- **Outputs:** regime timeline + per-regime equity/bond return, vol, drawdown proxy, correlation; macro context (recession share, inflation avg).  
- **Evaluation plan:** macro alignment + rolling-window robustness + regime separation of risk/return metrics.
"""
)