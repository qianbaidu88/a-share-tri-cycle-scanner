import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="A股三周期筛选器", layout="wide")

# ============================
# 指数成分股股票池（中证口径）
# ============================
@st.cache_data(show_spinner=False)
def get_index_universe(selected: list[str]) -> pd.DataFrame:
    """
    根据用户选择的指数，拉取成分股并合并去重。
    返回DataFrame：code, name（name可能为空，扫描时不影响）
    """
    import akshare as ak

    mapping = {
        "沪深300": "000300",
        "中证500": "000905",
        "中证1000": "000852",
        "创业板指数": "399006",
    }

    codes = set()

    for idx_name in selected:
        idx_code = mapping[idx_name]
        # 中证指数成分股（权重表），列名一般包含：成分券代码 / 成分券名称
        df = ak.index_stock_cons_weight_csindex(idx_code)
        if df is None or df.empty:
            continue

        # 兼容列名
        code_col = "成分券代码" if "成分券代码" in df.columns else ("code" if "code" in df.columns else None)
        name_col = "成分券名称" if "成分券名称" in df.columns else ("name" if "name" in df.columns else None)

        if code_col is None:
            continue

        for c in df[code_col].astype(str).tolist():
            codes.add(c)

    # 生成股票池表
    out = pd.DataFrame({"code": sorted(list(codes))})
    out["name"] = ""  # 这里不强依赖名称，扫描时可以空着
    return out


# ============================
# A股日线（AkShare）
# ============================
@st.cache_data(show_spinner=False)
def get_daily_hist(symbol: str, start: str, end: str):
    import akshare as ak

    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
    if df is None or df.empty:
        return None

    rename = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "vol", "成交额": "amt"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()

    # 兼容成交额缺失
    if "amt" not in df.columns:
        df["amt"] = np.nan

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ============================
# 周线聚合 + 趋势过滤（周线不下跌趋势）
# ============================
def to_weekly(d: pd.DataFrame) -> pd.DataFrame:
    w = (
        d.set_index("date")
        .resample("W-FRI")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            vol=("vol", "sum"),
            amt=("amt", "sum"),
        )
        .dropna()
        .reset_index()
    )
    return w


def _slope(y: np.ndarray) -> float:
    if len(y) < 6:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def weekly_trend_ok(w: pd.DataFrame, ma_len: int = 20) -> bool:
    """
    可执行版本的“周线不呈现下跌趋势”：
    - 周收盘 > 周MA
    - 周MA最近10根斜率为正
    """
    if w is None or len(w) < ma_len + 15:
        return False
    ma = w["close"].rolling(ma_len).mean()
    ma_tail = ma.dropna().values
    if len(ma_tail) < 12:
        return False
    return (w["close"].iloc[-1] > ma.iloc[-1]) and (_slope(ma_tail[-10:]) > 0)


# ============================
# 日线形态：量化近似（收敛 / 二踩 / 圆弧底）
# ============================
def bb_width(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = series.rolling(n).mean()
    std = series.rolling(n).std()
    up = mid + k * std
    dn = mid - k * std
    return (up - dn) / (mid.replace(0, np.nan))


def pattern_converging(d: pd.DataFrame) -> bool:
    # 收敛：布林带宽度收窄 + 当前带宽不大
    if len(d) < 80:
        return False
    w = bb_width(d["close"], n=20).dropna()
    if len(w) < 30:
        return False
    return (w.iloc[-1] < w.iloc[-15]) and (w.iloc[-1] < 0.25)


def _local_mins(x: np.ndarray, order: int = 3):
    mins = []
    for i in range(order, len(x) - order):
        if x[i] == np.min(x[i - order : i + order + 1]):
            mins.append(i)
    return mins


def pattern_double_test(d: pd.DataFrame) -> bool:
    # 二踩：近60日两个低点，第二个不明显更低，之后反弹>=8%
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
    if l2 < l1 * 0.98:  # 第二低点不比第一低太多（不破位）
        return False
    post_high = win["high"].iloc[i2:].max()
    return (post_high / max(l2, 1e-9) - 1.0) >= 0.08


def pattern_rounding_bottom(d: pd.DataFrame) -> bool:
    # 圆弧底：近60日收盘二次拟合U形 + 末端回升
    if len(d) < 120:
        return False
    y = d["close"].iloc[-60:].values.astype(float)
    rng = y.max() - y.min()
    if rng <= 0:
        return False
    y = (y - y.min()) / (rng + 1e-9)
    t = np.arange(len(y))
    a2 = np.polyfit(t, y, 2)[0]
    return (a2 > 0) and (y[-1] > y[len(y) // 2])


# ============================
# 交易参数（Buy / Stop / RR / MaxBuy / 仓位）
# ============================
def calc_params(d: pd.DataFrame, buy_mult: float = 1.02) -> dict:
    close = float(d["close"].iloc[-1])
    buy = close * buy_mult

    # 压力位近似：近20日最高价
    target1 = float(d["high"].iloc[-20:].max())

    # 逻辑止损近似：近10日最低价
    stop_logic = float(d["low"].iloc[-10:].min())

    # 以价定损：Buy - (Target1 - Buy)/3
    stop_price = buy - (target1 - buy) / 3.0

    # 按你的规则：选止损幅度更小的那个（更紧）
    stop_final = min(stop_logic, stop_price)

    rr = (target1 - buy) / max(buy - stop_final, 1e-9)

    # 性价比上限价： (Target1 - Stop)/4 + Stop
    maxbuy = (target1 - stop_final) / 4.0 + stop_final

    return dict(
        Close=close,
        Buy=buy,
        Target1=target1,
        Stop_logic=stop_logic,
        Stop_price=stop_price,
        Stop=stop_final,
        RR=rr,
        MaxBuy=maxbuy,
    )


def position_size(account: float, risk_pct: float, buy: float, stop: float) -> int:
    """
    用“单笔风险比例”反推建议股数（A股100股一手）
    """
    risk_money = account * risk_pct
    per_share_risk = max(buy - stop, 1e-9)
    shares = int(risk_money / per_share_risk / 100) * 100
    return max(shares, 0)


# ============================
# UI
# ============================
st.title("A股三周期筛选器（周线方向 + 日线形态 + RR≥3 + Stop/MaxBuy/仓位）")

with st.sidebar:
    st.header("股票池（指数成分）")
    index_selected = st.multiselect(
        "选择指数（可多选合并去重）",
        ["沪深300", "中证500", "中证1000", "创业板指数"],
        default=["沪深300", "中证500"],
    )

    st.header("历史区间")
    start = st.text_input("日线起始日期 YYYYMMDD", "20230101")
    end = st.text_input("日线结束日期 YYYYMMDD", "20261231")

    st.header("三周期参数")
    buy_mult = st.selectbox("Buy倍数（收盘×）", [1.02, 1.03], index=0)
    ma_len = st.number_input("周线MA长度（趋势过滤）", min_value=10, max_value=60, value=20, step=5)
    rr_min = st.number_input("RR最低要求", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

    st.header("仓位风控")
    account = st.number_input("账户资金（元）", min_value=0.0, value=200000.0, step=10000.0)
    risk_pct = st.number_input(
        "单笔风险比例（0.01=1%）", min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f"
    )

    st.caption("说明：指数成分股作为股票池；形态为量化近似；下一版可加分钟级盘中触发。")

run = st.button("开始扫描", type="primary")

if run:
    if not index_selected:
        st.warning("请至少选择一个指数。")
        st.stop()

    try:
        uni = get_index_universe(index_selected)
    except Exception as e:
        st.error(f"无法获取指数成分股：{e}")
        st.stop()

    if uni is None or uni.empty:
        st.warning("指数成分股列表为空，请稍后再试或更换指数。")
        st.stop()

    results = []
    prog = st.progress(0)
    status = st.empty()

    total = len(uni)
    for i, row in uni.iterrows():
        code = str(row["code"])
        name = str(row.get("name", ""))

        status.write(f"扫描 {i+1}/{total}：{code} {name}".strip())
        prog.progress((i + 1) / total)

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

        results.append(
            {
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
            }
        )

    if not results:
        st.warning("本轮条件下没有筛出股票。可放宽：RR阈值/形态过滤/或扩大指数选择范围。")
    else:
        out = pd.DataFrame(results).sort_values("RR", ascending=False).reset_index(drop=True)
        st.success(f"筛出 {len(out)} 只（按RR排序）")
        st.dataframe(out, use_container_width=True)

        st.download_button(
            "导出CSV",
            data=out.to_csv(index=False).encode("utf-8-sig"),
            file_name="A股三周期筛选结果_指数池.csv",
            mime="text/csv",
        )
