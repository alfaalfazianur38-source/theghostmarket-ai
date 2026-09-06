import os
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_NAME = "TheGhostMarket AI"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
TD_BASE = "https://api.twelvedata.com"

SYSTEM_PROMPT = r"""
You are TheGhostMarket AI, a professional XAUUSD market-analysis assistant.

PRIMARY STRATEGY:
When an MQ4 file is supplied, treat its actual code as the primary strategy reference.
Read it carefully and identify indicators, parameters, BUY conditions, SELL conditions,
filters, entry logic, SL/TP logic, trailing stops, money management, timeframe logic,
candle-close/tick behavior, and possible repaint behavior. Never invent MQ4 behavior.

MARKET DATA:
The application supplies recent OHLCV data and calculated indicators from Twelve Data.
Do not invent prices or indicators. If the supplied data is insufficient, say so.

MULTI-TIMEFRAME:
Use D1/H4 as HTF, H1 as MTF, and M15/M5 as LTF when available.
Evaluate trend, structure, support/resistance, breakout/retest, liquidity, momentum,
volatility, RSI, MACD, ADX and EMA context.

TRADE SELECTION:
The user may ask for trade 1, 2 or 3.
Return the best VALID setup for the requested trade number.
Do not repeat a previous setup. Do not create a trade just to fill a quota.
NO VALID TRADE SETUP is a valid result.

RISK:
Never guarantee profit. Default risk is 1% per trade.
SL must be based on structural invalidation. TP targets should use structure/liquidity.
Prefer reasonable risk/reward. Confidence is a quality score, not a probability guarantee.

NEWS:
The app may provide current web-search context. Treat it as informational risk context.
Do not fabricate news.

OUTPUT:
Return only the JSON object matching the supplied JSON schema.
"""

TRADE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "trade_signal_Theghostmachine": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "date": {"type": "string"},
                "pair": {"type": "string"},
                "trade_number": {"type": "integer"},
                "current_price": {"type": ["number", "null"]},
                "trade_type": {"type": "string", "enum": ["BUY", "SELL", "NO_TRADE"]},
                "entry_price": {"type": "string"},
                "stop_loss": {"type": ["number", "null"]},
                "take_profit_1": {"type": ["number", "null"]},
                "take_profit_2": {"type": ["number", "null"]},
                "take_profit_3": {"type": ["number", "null"]},
                "risk_reward": {"type": "string"},
                "confidence": {"type": "integer"},
                "market_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                "analysis": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "trend_detection": {"type": "string"},
                        "market_structure": {"type": "string"},
                        "volatility_level": {"type": "string"},
                        "support_levels": {"type": "array", "items": {"type": "number"}},
                        "resistance_levels": {"type": "array", "items": {"type": "number"}},
                        "liquidity_zones": {"type": "array", "items": {"type": "string"}},
                        "price_action": {"type": "string"},
                        "technical_indicators": {"type": "array", "items": {"type": "string"}},
                        "news_risk": {"type": "string"}
                    },
                    "required": [
                        "trend_detection", "market_structure", "volatility_level",
                        "support_levels", "resistance_levels", "liquidity_zones",
                        "price_action", "technical_indicators", "news_risk"
                    ]
                },
                "entry_confirmation": {"type": "array", "items": {"type": "string"}},
                "possible_outcome": {"type": "array", "items": {"type": "string"}},
                "invalidation": {"type": "string"},
                "risk_management": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "recommended_risk_percent": {"type": "number"},
                        "risk_level": {"type": "string"}
                    },
                    "required": ["recommended_risk_percent", "risk_level"]
                },
                "setup_status": {
                    "type": "string",
                    "enum": ["VALID", "INVALID", "WAIT_FOR_CONFIRMATION", "NO_VALID_TRADE_SETUP", "DATA_UNAVAILABLE"]
                }
            },
            "required": [
                "date", "pair", "trade_number", "current_price", "trade_type",
                "entry_price", "stop_loss", "take_profit_1", "take_profit_2",
                "take_profit_3", "risk_reward", "confidence", "market_bias",
                "analysis", "entry_confirmation", "possible_outcome",
                "invalidation", "risk_management", "setup_status"
            ]
        }
    },
    "required": ["trade_signal_Theghostmachine"]
}


def td_get(api_key: str, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    p["apikey"] = api_key
    r = requests.get(f"{TD_BASE}/{endpoint}", params=p, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(data.get("message", "Twelve Data error"))
    return data


def get_series(api_key: str, interval: str, outputsize: int = 220) -> pd.DataFrame:
    data = td_get(
        api_key,
        "time_series",
        {
            "symbol": "XAU/USD",
            "interval": interval,
            "outputsize": outputsize,
            "timezone": "Asia/Jakarta",
            "order": "ASC",
        },
    )
    values = data.get("values", [])
    if not values:
        raise RuntimeError(f"No XAU/USD data returned for {interval}.")
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def get_price(api_key: str) -> float:
    data = td_get(api_key, "price", {"symbol": "XAU/USD"})
    return float(data["price"])


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(s: pd.Series):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, pd.NA)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, pd.NA)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi14"] = rsi(x["close"], 14)
    x["adx14"] = adx(x, 14)
    x["macd"], x["macd_signal"], x["macd_hist"] = macd(x["close"])
    x["atr14"] = (
        pd.concat([
            x["high"] - x["low"],
            (x["high"] - x["close"].shift()).abs(),
            (x["low"] - x["close"].shift()).abs()
        ], axis=1).max(axis=1).rolling(14).mean()
    )
    return x


def compact_snapshot(df: pd.DataFrame, n: int = 40) -> Dict[str, Any]:
    x = enrich(df).tail(n).copy()
    cols = [
        "datetime", "open", "high", "low", "close",
        "ema20", "ema50", "ema200", "rsi14", "adx14",
        "macd", "macd_signal", "macd_hist", "atr14"
    ]
    cols = [c for c in cols if c in x.columns]
    x = x[cols].copy()
    x["datetime"] = x["datetime"].astype(str)
    return {
        "latest": x.iloc[-1].replace({float("nan"): None}).to_dict(),
        "recent": x.tail(12).round(4).to_dict(orient="records"),
    }


def read_mq4(uploaded) -> str:
    raw = uploaded.getvalue()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def analyze(
    openai_key: str,
    model: str,
    mq4_text: str,
    market: Dict[str, Any],
    trade_number: int,
    previous_trades: List[Dict[str, Any]],
    use_web_search: bool,
) -> Dict[str, Any]:
    client = OpenAI(api_key=openai_key)

    previous = json.dumps(previous_trades[-5:], ensure_ascii=False, indent=2)
    user_payload = {
        "task": f"Find trade opportunity #{trade_number} for XAUUSD.",
        "date": datetime.now().astimezone().isoformat(),
        "mq4_code": mq4_text,
        "market_data": market,
        "previous_trade_setups": previous,
        "requirements": [
            "Use the MQ4 code as primary strategy reference.",
            "Do not invent unavailable market values.",
            "Do not repeat previous setups.",
            "If no valid setup exists, return NO_TRADE/NO_VALID_TRADE_SETUP.",
            "Return only the JSON schema object."
        ],
    }

    kwargs = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": json.dumps(user_payload, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "trade_signal",
                "strict": True,
                "schema": TRADE_SCHEMA,
            }
        },
    }

    if use_web_search:
        kwargs["tools"] = [{"type": "web_search"}]

    response = client.responses.create(**kwargs)
    text = response.output_text
    return json.loads(text)


def build_market(api_key: str) -> Dict[str, Any]:
    intervals = ["5min", "15min", "1h", "4h", "1day"]
    result = {"symbol": "XAU/USD", "source": "Twelve Data"}
    result["current_price"] = get_price(api_key)
    for interval in intervals:
        df = get_series(api_key, interval, 220)
        result[interval] = compact_snapshot(df)
    return result



def calculate_lot_size(balance: float, risk_percent: float, entry: float, stop: float,
                       contract_size: float = 100.0, min_lot: float = 0.01,
                       max_lot: float = 100.0, lot_step: float = 0.01) -> float | None:
    """Approximate XAUUSD lot size from account risk and entry/SL distance.
    Actual broker contract specifications can differ, so treat this as an estimate.
    """
    distance = abs(entry - stop)
    if balance <= 0 or risk_percent <= 0 or distance <= 0:
        return None
    risk_money = balance * (risk_percent / 100.0)
    raw = risk_money / (distance * contract_size)
    stepped = math.floor(raw / lot_step) * lot_step
    return round(max(min_lot, min(max_lot, stepped)), 2)


def signal_summary(signal: Dict[str, Any]) -> str:
    if signal.get("trade_type") == "NO_TRADE":
        return "NO TRADE — menunggu setup yang valid."
    return (
        f"{signal.get('trade_type')} | Entry {signal.get('entry_price')} | "
        f"SL {signal.get('stop_loss')} | "
        f"TP1 {signal.get('take_profit_1')} | "
        f"TP2 {signal.get('take_profit_2')} | "
        f"TP3 {signal.get('take_profit_3')}"
    )

st.set_page_config(page_title=APP_NAME, page_icon="👻", layout="wide")
st.title("👻 TheGhostMarket AI")
st.caption("XAUUSD • MQ4 Strategy Reader • Multi-Timeframe Scanner • Risk-Aware Trade Signals")

st.markdown("""
<style>
.metric-card {
    padding: 14px 16px;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    background: rgba(128,128,128,.06);
}
.signal-buy { font-size: 28px; font-weight: 800; }
.signal-sell { font-size: 28px; font-weight: 800; }
.signal-wait { font-size: 28px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

if "trades" not in st.session_state:
    st.session_state.trades = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "latest_market" not in st.session_state:
    st.session_state.latest_market = None

with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    td_key = st.text_input("Twelve Data API Key", value=os.getenv("TWELVE_DATA_API_KEY", ""), type="password")
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    risk = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    balance = st.number_input("Account balance", min_value=0.0, value=1000.0, step=100.0)
    contract_size = st.number_input("XAU contract size", min_value=1.0, value=100.0, step=1.0)
    use_web = st.checkbox("Use web search for current news", value=True)

uploaded = st.file_uploader("Upload Theghostmachine.mq4", type=["mq4", "txt"])

if uploaded:
    mq4_text = read_mq4(uploaded)
    st.success(f"MQ4 loaded: {uploaded.name} • {len(mq4_text):,} characters")
    with st.expander("Preview MQ4"):
        st.code(mq4_text[:12000], language="cpp")
else:
    mq4_text = ""
    st.info("Upload your .mq4 strategy first.")

st.subheader("Trade scanner")
c1, c2, c3, c4 = st.columns(4)
buttons = [
    (c1, "🔎 First Trade", 1),
    (c2, "🔎 Second Trade", 2),
    (c3, "🔎 Third Trade", 3),
    (c4, "🔄 Update / Recheck", 0),
]

selected = None
for col, label, num in buttons:
    if col.button(label, use_container_width=True):
        selected = num

if selected is not None:
    if not openai_key:
        st.error("Masukkan OpenAI API Key.")
        st.stop()
    if not td_key:
        st.error("Masukkan Twelve Data API Key.")
        st.stop()
    if not mq4_text:
        st.error("Upload file .mq4 terlebih dahulu.")
        st.stop()

    trade_number = selected
    if selected == 0:
        trade_number = st.session_state.trades[-1]["trade_number"] if st.session_state.trades else 1

    try:
        with st.spinner("Mengambil data XAU/USD dan menganalisis strategi MQ4..."):
            market = build_market(td_key)
            result = analyze(
                openai_key=openai_key,
                model=model,
                mq4_text=mq4_text,
                market=market,
                trade_number=trade_number,
                previous_trades=st.session_state.trades,
                use_web_search=use_web,
            )

        signal = result["trade_signal_Theghostmachine"]
        signal["risk_management"]["recommended_risk_percent"] = risk

        # Estimate lot size when numeric entry and SL are available.
        try:
            entry = float(str(signal.get("entry_price", "")).split()[0])
            stop = float(signal["stop_loss"])
            lot = calculate_lot_size(
                balance=balance,
                risk_percent=risk,
                entry=entry,
                stop=stop,
                contract_size=contract_size,
            )
        except (TypeError, ValueError, AttributeError):
            lot = None

        signal["risk_management"]["estimated_lot_size"] = lot
        signal["risk_management"]["account_balance"] = balance

        if selected != 0:
            st.session_state.trades.append(signal)
        st.session_state.latest_result = result
        st.session_state.latest_market = market

        st.success("Analisis selesai.")

        # Dashboard summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", signal.get("trade_type", "—"))
        c2.metric("Bias", signal.get("market_bias", "—"))
        c3.metric("Confidence", f'{signal.get("confidence", 0)}%')
        c4.metric("Risk / Reward", signal.get("risk_reward", "—"))

        st.markdown("### 🎯 Trade Setup")
        st.info(signal_summary(signal))

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Entry", signal.get("entry_price", "—"))
        p2.metric("Stop Loss", signal.get("stop_loss", "—"))
        p3.metric("TP1", signal.get("take_profit_1", "—"))
        p4.metric("Estimated Lot", "—" if lot is None else str(lot))

        analysis = signal.get("analysis", {})
        with st.expander("📊 Market Analysis", expanded=True):
            a1, a2 = st.columns(2)
            with a1:
                st.write("**Trend:**", analysis.get("trend_detection", "—"))
                st.write("**Structure:**", analysis.get("market_structure", "—"))
                st.write("**Volatility:**", analysis.get("volatility_level", "—"))
                st.write("**Price Action:**", analysis.get("price_action", "—"))
            with a2:
                st.write("**Support:**", analysis.get("support_levels", []))
                st.write("**Resistance:**", analysis.get("resistance_levels", []))
                st.write("**Liquidity:**", analysis.get("liquidity_zones", []))
                st.write("**News Risk:**", analysis.get("news_risk", "—"))

        with st.expander("✅ Entry Confirmation"):
            for item in signal.get("entry_confirmation", []):
                st.write("•", item)

        with st.expander("⚠️ Invalidation & Possible Outcome"):
            st.write("**Invalidation:**", signal.get("invalidation", "—"))
            for item in signal.get("possible_outcome", []):
                st.write("•", item)

        with st.expander("🧠 Raw JSON"):
            st.json(result)

        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"xauusd_trade_{trade_number}.json",
            mime="application/json",
        )
    except Exception as e:
        st.error(f"Analisis gagal: {e}")


if st.session_state.latest_market:
    st.subheader("📡 Latest Market Snapshot")
    m = st.session_state.latest_market
    mc1, mc2 = st.columns(2)
    mc1.metric("XAU/USD", f'{m.get("current_price", "—")}')
    mc2.write("**Timeframes loaded:** M5 • M15 • H1 • H4 • D1")

with st.expander("📚 Previous trade setups"):
    if st.session_state.trades:
        st.json(st.session_state.trades)
    else:
        st.write("Belum ada trade setup.")

st.divider()
st.caption("Educational analysis only. Trading involves risk; no signal guarantees profit.")
