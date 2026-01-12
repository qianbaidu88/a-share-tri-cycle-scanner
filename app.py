import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="A股三周期筛选器", layout="wide")

@st.cache_data(show_spinner=False)
def get_spot_topn(topn: int = 200) -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name"])

    col_map_candidates = {
        "code": ["代码", "symbol", "股票代码"],
        "name": ["名称", "name", "股票名称"],
        "amt": ["成交额", "成交额(元)", "成交额（元）", "成交额(万元)", "成交额（万元）"],
    }

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    code_col = pick(col_map_candidates["code"])
    name_col = pick(col_map_candidates["name"])
    amt_col  = pick(col_map_candidates["amt"])

    if code_col is None or name_col is None:
        out = df.iloc[:, :2].copy()
        out.columns = ["code", "name"]
        return out.head(topn)

    out = df[[code_col, name_col] + ([amt_col] if amt_col else [])].copy()
    out = out.rename(columns={code_col: "code", name_col: "name"})
    if amt_col:
        out = out.sort_values(amt_col, ascending=False).head(topn)
    else:
        out = out.head(topn)
    return out[["code", "name"]].reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_daily_hist(symbol: str, start: str, end: str):
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
    if df is None or df.empty:
        return None
    rename = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"vol","成交额":"amt"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns}).copy()
    if "amt" not in df.columns:
        df["amt"] = np.nan
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def to_weekly(d: pd.DataFrame) -> pd.DataFrame:
    w = (d.set_index("date")
           .resample("W-FRI")
           .agg(open=("open","first"),
                high=("high","max"),
                low=("low","min"),
                close=("close","last"),
                vol=("vol","sum"),
                amt=("amt","sum"))
           .dropna()
           .reset_index())
    return w

def _slope(y: np.ndarray) -> float:
    if len(y) < 6:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])

def weekly_trend_ok(w: pd.DataFrame, ma_len: int = 20) -> bool:
    if w is None or len(w) < ma_len + 15:
        return False
    ma = w["close"].rolling(ma_len).mean()
    ma_tail = ma.dropna().values
    if len(ma_tail) < 12:
        return False
    return (w["close"].iloc[-1] > ma.iloc[-1]) and (_slope(ma_tail[-10:]) > 0)

def bb_width(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = series.rolling(n).mean()
    std = series.rolling(n).std()
    up = mid + k*std
    dn = mid - k*std
    return (up - dn) / (mid.replace(0, np.nan))

def pattern_converging(d: pd.DataFrame) -> bool:
    if len(d) < 80:
        return False
    w = bb_width(d["close"], n=20).dropna()
    if len(w) < 30:
        return False
    return (w.iloc[-1] < w.iloc[-15]) and (w.iloc[-1] < 0.25)

def _local_mins(x: np.ndarray, order: int = 3):
    mins = []
    for i in range(order, len(x)-order):
        if x[i] == np.min(x[i-order:i+order+1]):
            mins.append(i)
    return mins

def pattern_double_test(d: pd.DataFrame) -> bool:
    if len(d) < 120:
        return False
    win = d.iloc[-60:].reset_index(drop=True)
    lows = win["low"].values
    idxs = _local_mins(lows, order=3)
    if len(idxs) < 2:
        return False
    i2, i1 = idxs[-1], idxs[-2]
    if not (10 <= (i2 - i1) <= 40):
        return False
    l1, l2 = lows[i1], lows[i2]
    if l2 < l1 * 0.98:
        return False
    post_high = win["high"].iloc[i2:].max()
    return (post_high / max(l2, 1e-9) - 1.0) >= 0.08

def pattern_rounding_bottom(d: pd.DataFrame) -> bool:
    if len(d) < 120:
        return False
    y = d["close"].iloc[-60:].values.astype(float)
    rng = y.max() - y.min()
    if rng <= 0:
        return False
    y = (y - y.min()) / (rng + 1e-9)
    t = np.arange(len(y))
    a2 = np.polyfit(t, y, 2)[0]
    return (a2 > 0) and (y[-1] > y[len(y)//2])

def calc_params(d: pd.DataFrame, buy_mult: float = 1.02) -> dict:
    close = float(d["close"].iloc[-1])
    buy = close * buy_mult
    target1 = float(d["high"].iloc[-20:].max())
    stop_logic = float(d["low"].iloc[-10:].min())
    stop_price = buy - (target1 - buy) / 3.0
    stop_final = min(stop_logic, stop_price)
    rr = (target1 - buy) / max(buy - stop_final, 1e-9)
    maxbuy = (target1 - stop_final) / 4.0 + stop_final
    return dict(Close=close, Buy=buy, Target1=target1,
                Stop_logic=stop_logic, Stop_price=stop_price,
                Stop=stop_final, RR=rr, MaxBuy=maxbuy)

def position_size(account: float, risk_pct: float, buy: float, stop: float) -> int:
    risk_money = account * risk_pct
    per_share_risk = max(buy - stop, 1e-9)
    shares = int(risk_money / per_share_risk / 100) * 100
    return max(shares, 0)

st.title("A股三周期筛选器（周线方向 + 日线形态 + RR≥3 + Stop/MaxBuy/仓位）")

with st.sidebar:
    st.header("股票池（A股）")
    topn = st.number_input("按成交额Top N（建议200~800）", min_value=50, max_value=2000, value=300, step=50)

    st.header("历史区间")
    start = st.text_input("日线起始日期 YYYYMMDD", "20230101")
    end = st.text_input("日线结束日期 YYYYMMDD", "20261231")

    st.header("三周期参数")
    buy_mult = st.selectbox("Buy倍数（收盘×）", [1.02, 1.03], index=0)
    ma_len = st.number_input("周线MA长度（趋势过滤）", min_value=10, max_value=60, value=20, step=5)
    rr_min = st.number_input("RR最低要求", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

    st.header("仓位风控")
    account = st.number_input("账户资金（元）", min_value=0.0, value=200000.0, step=10000.0)
    risk_pct = st.number_input("单笔风险比例（0.01=1%）", min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f")

    st.caption("形态用量化近似；下一版可加分钟级盘中触发（水上/3分钟涨速/放量）。")

run = st.button("开始扫描", type="primary")

if run:
    uni = get_spot_topn(int(topn))
    if uni.empty:
        st.error("无法获取A股行情数据（AkShare/网络/接口）。请稍后重试或更新 akshare。")
        st.stop()

    results = []
    prog = st.progress(0)
    status = st.empty()

    total = len(uni)
    for i, row in uni.iterrows():
        code = str(row["code"])
        name = str(row["name"])
        status.write(f"扫描 {i+1}/{total}：{code} {name}")
        prog.progress((i+1)/total)

        try:
            d = get_daily_hist(code, start=start, end=end)
        except Exception:
            continue
        if d is None or len(d) < 130:
            continue

        w = to_weekly(d)
        if not weekly_trend_ok(w, ma_len=int(ma_len)):
            continue

        p1 = pattern_converging(d)
        p2 = pattern_double_test(d)
        p3 = pattern_rounding_bottom(d)
        if not (p1 or p2 or p3):
            continue

        p = calc_params(d, buy_mult=float(buy_mult))
        if p["RR"] < float(rr_min):
            continue

        shares = position_size(float(account), float(risk_pct), p["Buy"], p["Stop"])

        results.append({
            "代码": code,
            "名称": name,
            "形态": "收敛" if p1 else ("二踩" if p2 else "圆弧底"),
            "Close": round(p["Close"], 4),
            "Buy": round(p["Buy"], 4),
            "Target1": round(p["Target1"], 4),
            "Stop": round(p["Stop"], 4),
            "RR": round(p["RR"], 3),
            "MaxBuy": round(p["MaxBuy"], 4),
            "建议股数": shares,
            "建议投入(元)": round(shares * p["Buy"], 2),
        })

    if not results:
        st.warning("本轮条件下没有筛出股票。可放宽：RR阈值/形态过滤/扩大股票池TopN。")
    else:
        out = pd.DataFrame(results).sort_values("RR", ascending=False).reset_index(drop=True)
        st.success(f"筛出 {len(out)} 只（按RR排序）")
        st.dataframe(out, use_container_width=True)
        st.download_button("导出CSV", data=out.to_csv(index=False).encode("utf-8-sig"),
                           file_name="A股三周期筛选结果.csv", mime="text/csv")
