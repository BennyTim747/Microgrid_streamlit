
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Tuple, Optional

# Optional ML (kept light)
try:
    from sklearn.neural_network import MLPRegressor
    SKL_OK = True
except Exception:
    SKL_OK = False

st.set_page_config(page_title="Microgrid Controller v4e (Streamlit)", layout="wide")

# ------------------------- Helpers -------------------------
DT = 0.25  # hour (15 min)
NPTS = 192 # 48h horizon (yesterday + today), index 0..191; now at 96
IDX_NOW = 96

def expand_pv_13_to_96(pv13: np.ndarray) -> np.ndarray:
    # pv13 -> hours 06..18 inclusive (13 values), linear to 96 points
    v = np.zeros(25)  # node at 0..24
    v[6:19] = pv13  # 06..18 mapped to indices 6..18; 19 is explicitly zero
    t_nodes = np.arange(25, dtype=float)
    tq = np.arange(0, 24, 0.25)
    pv_q = np.interp(tq, t_nodes, v, left=0, right=0)
    return np.clip(pv_q, 0, None)

def expand_h24_to_96(h24: np.ndarray) -> np.ndarray:
    # 24 hourly values -> 96 points (linear)
    v = np.r_[h24, h24[-1]]
    t_nodes = np.arange(25, dtype=float)
    tq = np.arange(0, 24, 0.25)
    load_q = np.interp(tq, t_nodes, v, left=v[0], right=v[-1])
    return np.clip(load_q, 0, None)

def default_pv_day(pmax=250.0) -> np.ndarray:
    h = np.linspace(0, 24, 96, endpoint=False)
    pv = np.where((h>=6)&(h<18), pmax*np.sin((h-6)/12*np.pi), 0.0)
    noise = np.random.default_rng(1).normal(0, 8, pv.shape)
    pv = np.maximum(0, pv + noise)
    # simple smoothing
    kernel = np.ones(3)/3
    pv = np.convolve(pv, kernel, mode="same")
    return pv

def dispatch_24h(pv: np.ndarray, load: np.ndarray, soc0: float,
                 eb_kwh=307.0, pcs_max=200.0, soc_min=0.10, soc_max=0.95,
                 eta_ch=1.0, eta_dis=1.0, temp_c=40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized simple rule-based dispatch (charge on surplus, discharge on deficit)"""
    assert pv.shape==(96,) and load.shape==(96,)
    Pbess = np.zeros_like(pv)
    Pgrid = np.zeros_like(pv)
    SOC = np.full_like(pv, np.nan, dtype=float)

    # derating by temperature (simple)
    if   temp_c >= 70: derate = 0.0
    elif temp_c >= 65: derate = 0.5
    elif temp_c >= 55: derate = 0.8
    else:              derate = 1.0
    Pmax_eff = pcs_max * derate

    soc = soc0
    for k in range(96):
        Ppv, Pload = pv[k], load[k]
        headroom_ch  = max(0.0, (soc_max - soc) * eb_kwh / DT)
        headroom_dis = max(0.0, (soc - soc_min) * eb_kwh / DT)

        if Ppv >= Pload:
            surplus = Ppv - Pload
            Pch = min(surplus, Pmax_eff, headroom_ch)
            pb = -Pch
        else:
            deficit = Pload - Ppv
            Pdis = min(deficit, Pmax_eff, headroom_dis)
            pb = +Pdis

        grid = Pload - Ppv - pb
        # SOC update
        soc_next = soc + ((-pb)*eta_ch - max(0.0, pb)/eta_dis)*DT/eb_kwh
        soc = min(max(soc_next, soc_min), soc_max)

        Pbess[k] = pb
        Pgrid[k] = grid
        SOC[k]   = soc * 100.0
    return Pbess, Pgrid, SOC

def compute_stats(Pgrid_future: np.ndarray, Pbess_future: np.ndarray,
                  pv_future: np.ndarray, load_future: np.ndarray) -> Tuple[float,float,float,float,float]:
    idx = slice(0,96)
    E_import = np.sum(np.maximum(0.0, Pgrid_future[idx])) * DT
    E_export = np.sum(np.maximum(0.0, -Pgrid_future[idx])) * DT
    dSOC = (np.sum(np.maximum(0.0, -Pbess_future[idx])) - np.sum(np.maximum(0.0, Pbess_future[idx]))) * DT / 307.0 * 100.0
    E_pv   = np.sum(np.maximum(0.0, pv_future[idx])) * DT
    E_load = np.sum(np.maximum(0.0, load_future[idx])) * DT
    return E_import, E_export, dSOC, E_pv, E_load

def sMAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(1e-6, np.abs(y_true) + np.abs(y_pred))
    return float(np.mean(2*np.abs(y_true - y_pred)/denom) * 100.0)

def make_time_index() -> pd.DatetimeIndex:
    # yesterday 00:00 to tomorrow 00:00 (15-min), now at today 00:00
    base0 = pd.Timestamp.now(tz="Australia/Brisbane").normalize()
    t_start = base0 - pd.Timedelta(days=1)
    times = pd.date_range(t_start, periods=192, freq="15min", tz="Australia/Brisbane")
    return times

def build_xy_from_history(hist_df: pd.DataFrame, H=5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    hist_df columns: ['time','pv_kW','load_kW'] at 15-min, continuous >= 6 days
    Return X, Y for next-day (PV then Load) regression.
    """
    if hist_df is None or len(hist_df)==0:
        return None, None
    # into daily (rows) x 96 (cols)
    pivot_pv = hist_df[['time','pv_kW']].copy()
    pivot_load = hist_df[['time','load_kW']].copy()
    pivot_pv['d'] = pivot_pv['time'].dt.normalize()
    pivot_load['d'] = pivot_load['time'].dt.normalize()

    days = sorted(list(set(pivot_pv['d']).intersection(set(pivot_load['d']))))
    if len(days) < H+1:
        return None, None
    days = days[-60:]  # cap to last 60 days

    # ensure each day has 96 entries
    pv_days = []
    load_days = []
    for d in days:
        mask_pv = (pivot_pv['d']==d)
        mask_ld = (pivot_load['d']==d)
        pv = pivot_pv.loc[mask_pv, 'pv_kW'].to_numpy()
        ld = pivot_load.loc[mask_ld, 'load_kW'].to_numpy()
        if len(pv) >= 96 and len(ld) >= 96:
            pv = pv[-96:]
            ld = ld[-96:]
            pv_days.append(pv)
            load_days.append(ld)
    if len(pv_days) < H+1:
        return None, None

    HP = np.vstack(pv_days)  # D x 96
    HL = np.vstack(load_days)
    D = HP.shape[0]
    S_full = D - H
    X_list, Y_list = [], []
    for s in range(S_full):
        dHist = slice(s, s+H)
        dNext = s+H
        histL = HL[dHist,:].reshape(-1)
        histP = HP[dHist,:].reshape(-1)
        exoL  = HL[dNext,:].reshape(-1)
        exoP  = HP[dNext,:].reshape(-1)
        x = np.concatenate([histL, histP, exoL, exoP])
        y = np.concatenate([exoP, exoL])  # predict next-day PV,Load
        X_list.append(x)
        Y_list.append(y)
    X = np.asarray(X_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    return X, Y

# ------------------------- Sidebar (inputs) -------------------------
st.sidebar.title("Controls")

with st.sidebar.expander("Initial Conditions", expanded=True):
    soc0_pct = st.slider("Initial SOC (%)", min_value=10, max_value=95, value=50, step=1)
    temp_c   = st.slider("Battery Temp (°C)", min_value=35, max_value=75, value=40, step=1)

with st.sidebar.expander("PV (06:00–18:00 hourly, kW)", expanded=True):
    default_pv13 = [0,10,40,80,130,180,220,250,220,180,120,60,10]
    pv_vals = []
    cols = st.columns(4)
    for i, hr in enumerate(range(6,19)):
        with cols[i%4]:
            pv_vals.append(st.number_input(f"{hr:02d}:00", min_value=0.0, max_value=250.0, value=float(default_pv13[i]), step=5.0, key=f"pv{hr}"))
    pv13 = np.array(pv_vals, dtype=float)
    pv_future = expand_pv_13_to_96(pv13)

with st.sidebar.expander("Load (00:00–23:00 hourly, kW)", expanded=True):
    cols = st.columns(4)
    load_vals = []
    for hr in range(24):
        with cols[hr%4]:
            load_vals.append(st.number_input(f"{hr:02d}:00", min_value=0.0, max_value=300.0, value=120.0, step=5.0, key=f"ld{hr}"))
    load24 = np.array(load_vals, dtype=float)
    load_future = expand_h24_to_96(load24)

# History CSV uploader
with st.sidebar.expander("History Import (≥6 days, 15-min)", expanded=False):
    hist_file = st.file_uploader("CSV: time, pv(MW or kW), load(MW or kW), optional price", type=["csv"], key="hist_csv")
    hist_scale_info = st.empty()
    do_train = st.checkbox("Train lightweight ANN (optional)", value=False, help="Uses scikit-learn MLP; keep data small.")

# ------------------------- Main layout -------------------------
st.title("Microgrid Controller v4e — ANN + UI (Streamlit Demo)")
st.caption("24h future dispatch with PV/Load sliders, optional ANN prediction overlay, and stats.")

times = make_time_index()

# History synthesis (if none imported) for yesterday
pv_hist = default_pv_day(250.0)
rng = np.random.default_rng(2)
load_hist = np.maximum(0, 100 + 30*np.sin(np.arange(96)/12) + 10*rng.normal(size=96))

pv_all = np.r_[pv_hist, pv_future]
load_all = np.r_[load_hist, load_future]

# Dispatch only for future 24h
Pbess_future, Pgrid_future, SOC_future = dispatch_24h(pv_future, load_future, soc0_pct/100.0, temp_c=temp_c)

# Compose series for 48h displays
Pbess_all = np.r_[np.zeros(96), Pbess_future]
Pgrid_all = np.r_[load_hist - pv_hist, Pgrid_future]
SOC_all   = np.r_[np.full(96, np.nan), SOC_future]

# ANN area
pv_pred_future = None
load_pred_future = None
if hist_file is not None:
    try:
        df = pd.read_csv(hist_file)
        cols = [c.lower() for c in df.columns]
        # Find time column
        tcol = None
        for k in ["settlement","time","timestamp","datetime","interval","date"]:
            tmatch = [c for c in df.columns if k in c.lower()]
            if tmatch:
                tcol = tmatch[0]; break
        if tcol is None:
            raise ValueError("Missing time column")
        t = pd.to_datetime(df[tcol], errors="coerce")
        # PV
        pcol = None
        for k in ["rooftop","pv","solar"]:
            m = [c for c in df.columns if k in c.lower()]
            if m: pcol = m[0]; break
        # Load
        lcol = None
        for k in ["operational demand","operational_demand","totaldemand","total demand","demand","load"]:
            m = [c for c in df.columns if k in c.lower()]
            if m: lcol = m[0]; break

        if (pcol is None) or (lcol is None):
            raise ValueError("Need PV and Load columns")

        pv_raw = pd.to_numeric(df[pcol], errors="coerce").to_numpy()
        ld_raw = pd.to_numeric(df[lcol], errors="coerce").to_numpy()

        # Basic unit heuristic: if max<2000, treat as kW; else MW->kW
        pv_kW = np.where(np.nanmax(pv_raw) < 2000, pv_raw, pv_raw*1000.0)
        ld_kW = np.where(np.nanmax(ld_raw) < 2000, ld_raw, ld_raw*1000.0)

        hist_df = pd.DataFrame({"time": t.dt.tz_localize("Australia/Brisbane", nonexistent="NaT", ambiguous="NaT"),
                                "pv_kW": pv_kW, "load_kW": ld_kW}).dropna()
        # Ensure regular 15-min by resampling
        hist_df = hist_df.set_index("time").resample("15min").mean().dropna().reset_index()

        # Scaling to microgrid (peak to 250kW PV, ~200kW p95 load)
        pv_peak = hist_df["pv_kW"].max()
        ld_q95  = hist_df["load_kW"].quantile(0.95)
        pv_scale = 250.0 / max(1.0, pv_peak)
        ld_scale = 200.0 / max(1.0, ld_q95)
        hist_df["pv_kW"]   = np.clip(hist_df["pv_kW"] * pv_scale, 0, 250)
        hist_df["load_kW"] = np.clip(hist_df["load_kW"] * ld_scale, 0, 300)
        hist_scale_info.info(f"Scaled history with pv_scale={pv_scale:.4f}, load_scale={ld_scale:.4f}")

        # If asked, train small ANN (light)
        if do_train and SKL_OK:
            X, Y = build_xy_from_history(hist_df, H=5)
            if X is not None and len(X)>0:
                # keep small model to avoid heavy compute
                n_samples = X.shape[0]
                # train/val split
                n_tr = max(1, int(0.7*n_samples))
                Xtr, Ytr = X[:n_tr], Y[:n_tr]
                Xte, Yte = X[n_tr:], Y[n_tr:] if n_tr < n_samples else (None, None)

                model = MLPRegressor(hidden_layer_sizes=(128,64), solver="lbfgs", max_iter=300, random_state=0)
                model.fit(Xtr, Ytr)
                # Build current feature using last H days + current sliders as exo
                # Reconstruct last H=5 days from hist_df
                days = sorted(hist_df["time"].dt.normalize().unique())
                days = days[-6:]  # ensure we have H+1
                pivot_pv = hist_df[["time","pv_kW"]].copy()
                pivot_load = hist_df[["time","load_kW"]].copy()
                pivot_pv["d"] = pivot_pv["time"].dt.normalize()
                pivot_load["d"] = pivot_load["time"].dt.normalize()
                hp=[]; hl=[]
                for d in days[:-1]:
                    hp.append(pivot_pv.loc[pivot_pv["d"]==d,"pv_kW"].to_numpy()[-96:])
                    hl.append(pivot_load.loc[pivot_load["d"]==d,"load_kW"].to_numpy()[-96:])
                histP = np.vstack(hp); histL = np.vstack(hl)
                exoP = pv_future.reshape(-1); exoL = load_future.reshape(-1)
                Xnow = np.concatenate([histL.reshape(-1), histP.reshape(-1), exoL, exoP]).reshape(1,-1)
                Yhat = model.predict(Xnow)[0]
                pv_pred_future = np.maximum(0.0, Yhat[:96])
                load_pred_future = np.maximum(0.0, Yhat[96:])
    except Exception as e:
        st.warning(f"History import/train warning: {e}")

# ------------------------- Charts -------------------------
import plotly.graph_objects as go

def line48(title, ylist, names, ymin=None, ymax=None):
    fig = go.Figure()
    for y, name in zip(ylist, names):
        fig.add_trace(go.Scatter(x=times, y=y, mode="lines", name=name))
    fig.add_vline(x=times[IDX_NOW], line_width=1, line_color="green")
    fig.update_layout(title=title, height=320, margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h"))
    if ymin is not None or ymax is not None:
        fig.update_yaxes(range=[ymin if ymin is not None else min([np.nanmin(y) for y in ylist])-10,
                                ymax if ymax is not None else max([np.nanmax(y) for y in ylist])+10])
    return fig

col_main, col_side = st.columns([3,2])

with col_main:
    # Main power chart
    ylist = [pv_all, load_all, Pbess_all, Pgrid_all]
    names = ["PV","Load","BESS","Grid"]
    # ANN overlay if available
    if pv_pred_future is not None:
        pv_overlay = np.r_[np.full(96, np.nan), pv_pred_future]
        ylist.append(pv_overlay); names.append("PV_ANN")
    if load_pred_future is not None:
        ld_overlay = np.r_[np.full(96, np.nan), load_pred_future]
        ylist.append(ld_overlay); names.append("Load_ANN")

    st.plotly_chart(line48("Power (PV / Load / BESS / Grid) — 48h", ylist, names), use_container_width=True)

    # SoC and Grid charts
    st.plotly_chart(line48("Battery SOC (Future 24h)", [SOC_all], ["SOC"], ymin=0, ymax=100), use_container_width=True)
    st.plotly_chart(line48("Grid Power P_grid (>0 import, <0 export)", [Pgrid_all], ["P_grid"]), use_container_width=True)

with col_side:
    # KPIs
    E_import, E_export, dSOC, E_pv, E_load = compute_stats(Pgrid_future, Pbess_future, pv_future, load_future)
    st.subheader("Next 24h KPIs")
    c1,c2 = st.columns(2)
    c1.metric("PV Energy (kWh)", f"{E_pv:.1f}")
    c2.metric("Load Energy (kWh)", f"{E_load:.1f}")
    c1.metric("Import (kWh)", f"{E_import:.1f}")
    c2.metric("Export (kWh)", f"{E_export:.1f}")
    c1.metric("ΔSOC (%)", f"{dSOC:.1f}")

    # Lamps
    st.subheader("Indicators")
    Pb_now = Pbess_all[IDX_NOW]; Pg_now = Pgrid_all[IDX_NOW]
    lamp_batt = "🟢 Discharging" if Pb_now>1e-3 else ("🔵 Charging" if Pb_now<-1e-3 else "⚪ Idle")
    lamp_grid = "🔵 Import" if Pg_now>1e-3 else ("🟡 Export" if Pg_now<-1e-3 else "⚪ Idle")
    st.write(f"**Battery:** {lamp_batt}")
    st.write(f"**Grid:** {lamp_grid}")
    st.write(f"**Temp:** {('🟥 ≥70°C Stop' if temp_c>=70 else '🟧 65–70°C Derate' if temp_c>=65 else '🟨 55–65°C Warn' if temp_c>=55 else f'🟢 Normal {temp_c:.0f}°C')}")

    # Error metrics (if ANN overlay exists and "today" has progressed)
    st.subheader("Prediction Errors")
    if (pv_pred_future is not None) or (load_pred_future is not None):
        # Assume act = slider-based future (for demo)
        k2 = 96  # using full future window for demo
        if (pv_pred_future is not None):
            rmse = float(np.sqrt(np.mean((pv_future[:k2]-pv_pred_future[:k2])**2)))
            smape = sMAPE(pv_future[:k2], pv_pred_future[:k2])
            st.write(f"**PV Err:** RMSE {rmse:.1f} kW | sMAPE {smape:.1f}%")
        if (load_pred_future is not None):
            rmse = float(np.sqrt(np.mean((load_future[:k2]-load_pred_future[:k2])**2)))
            smape = sMAPE(load_future[:k2], load_pred_future[:k2])
            st.write(f"**Load Err:** RMSE {rmse:.1f} kW | sMAPE {smape:.1f}%")
    else:
        st.write("No ANN overlay yet (import ≥6d history and tick training).")

st.info("Tips: Adjust PV/Load sliders on the left → charts update instantly. Upload a ≥6-day 15-min history to train a lightweight ANN and overlay predictions.")

