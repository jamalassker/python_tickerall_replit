"""
TICKERALL ADAPTIVE ACTIVE SCALPER V10.0
========================================

TickerAll REST API -> Valetax MT5

ENTRY TIMEFRAME: H1
CONFIRMATION: H4

DEMO ONLY.
No guaranteed profits or win rate.

STRATEGIES:
1. EMA trend continuation
2. Pullback recovery
3. Donchian breakout
4. Bollinger range reversal

IMPORTANT:
- Scans every discovered target symbol every cycle.
- Signal scanning continues in SIGNAL-ONLY mode.
- H4 confirmation uses completed candles only.
- All entries use completed H1 candles.
- Orders require explicit demo verification.
- No forced/random entries.
"""

import os
import time
import math
import uuid
import json
import asyncio
import logging

from datetime import datetime, timezone

import httpx
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

def env_bool(name, default=False):
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in (
        "true", "1", "yes", "on"
    )


def env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return float(default)


def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return int(default)


BASE_URL = os.getenv(
    "TICKERALL_BASE_URL",
    "https://api.tickerall.com"
).rstrip("/")

# Credentials MUST come from Railway environment variables.
api_key="cf_api_130fd0fab52f8443842e609da594c80483d71b94786c47f95b94cd3af0946154"
BROKER= "mt5"
MT5_SERVER = "ValetaxGlobal-Live3"
MT5_LOGIN = 641436570
MT5_PASSWORD = "EJam123!@"

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_TOKEN", ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID", ""
).strip()


# Trading remains disabled unless deliberately enabled.
ENABLE_TRADING = env_bool(
    "ENABLE_TRADING", False
)

ALLOW_DEMO_TRADING = env_bool(
    "ALLOW_DEMO_TRADING", False
)


# Timeframes are intentionally fixed to H1/H4.
TIMEFRAME = "H1"
CONFIRMATION_TIMEFRAME = "H4"

USE_H4_CONFIRMATION = True


# Signal quality.
MIN_SIGNAL_SCORE = env_int(
    "MIN_SIGNAL_SCORE", 3
)


# Risk.
RISK_PERCENT = env_float(
    "RISK_PERCENT", 0.20
)

MAX_POSITIONS = env_int(
    "MAX_POSITIONS", 4
)

MAX_POSITIONS_PER_SYMBOL = env_int(
    "MAX_POSITIONS_PER_SYMBOL", 1
)

MAX_DAILY_LOSS_PERCENT = env_float(
    "MAX_DAILY_LOSS_PERCENT", 3.0
)

MAX_DRAWDOWN_PERCENT = env_float(
    "MAX_DRAWDOWN_PERCENT", 8.0
)


# Execution filters.
MAX_SPREAD_ATR = env_float(
    "MAX_SPREAD_ATR", 0.65
)

ATR_SL_MULT = env_float(
    "ATR_SL_MULT", 1.8
)

ATR_TP_MULT = env_float(
    "ATR_TP_MULT", 3.0
)

MIN_RR = env_float(
    "MIN_RR", 1.6
)


# Market regime.
ADX_TREND_THRESHOLD = env_float(
    "ADX_TREND_THRESHOLD", 18
)

ADX_RANGE_THRESHOLD = env_float(
    "ADX_RANGE_THRESHOLD", 17
)


# Scanning.
SCAN_INTERVAL = env_int(
    "SCAN_INTERVAL", 300
)

SYMBOL_COOLDOWN_SECONDS = env_int(
    "SYMBOL_COOLDOWN_SECONDS", 120
)

# 1,400 hours gives approximately:
# H1: 1,400 candles
# H4: 350 candles
#
# Actual returned history depends on API/broker limits.
MAX_CANDLE_HOURS = env_int(
    "MAX_CANDLE_HOURS", 1400
)

MIN_H1_CANDLES = env_int(
    "MIN_H1_CANDLES", 220
)

MIN_H4_CANDLES = env_int(
    "MIN_H4_CANDLES", 250
)


HTTP_TIMEOUT = env_float(
    "HTTP_TIMEOUT", 30
)

# Default is deliberately large enough for the full target universe.
MAX_SYMBOLS_PER_CYCLE = env_int(
    "MAX_SYMBOLS_PER_CYCLE", 1000
)

# Limit concurrent candle requests to reduce API pressure.
MAX_CONCURRENT_SCANS = env_int(
    "MAX_CONCURRENT_SCANS", 4
)


RISK_STATE_FILE = os.getenv(
    "RISK_STATE_FILE",
    "/data/tickerall_v10_risk.json"
)

LOG_LEVEL = os.getenv(
    "LOG_LEVEL", "INFO"
).upper()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("TICKERALL_V10")


# ============================================================
# SYMBOL UNIVERSE
# ============================================================

CURRENCIES = [
    "USD", "EUR", "GBP", "JPY",
    "CHF", "CAD", "AUD", "NZD"
]

FOREX_SYMBOLS = [
    a + b
    for a in CURRENCIES
    for b in CURRENCIES
    if a != b
]

CRYPTO_SYMBOLS = [
    "BTCUSD",
    "ETHUSD"
]

METAL_GROUPS = [
    ("XAUUSD", "GOLD"),
    ("XAGUSD", "SILVER")
]

TARGET_SYMBOLS = (
    FOREX_SYMBOLS
    + CRYPTO_SYMBOLS
)


# ============================================================
# UTILITIES
# ============================================================

def number(value, default=0.0):

    try:
        if value is None:
            return default

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (ValueError, TypeError):
        return default


def normalize_symbol(symbol):

    return "".join(
        c for c in str(symbol).upper()
        if c.isalnum()
    )


def symbol_matches(actual, desired):

    actual = normalize_symbol(actual)
    desired = normalize_symbol(desired)

    return (
        actual == desired
        or actual.startswith(desired)
    )


def floor_step(value, step):

    if step <= 0:
        return 0

    return math.floor(
        (value + 1e-12) / step
    ) * step


def round_price(price, digits, tick_size):

    if tick_size > 0:
        price = round(price / tick_size) * tick_size

    return round(
        price,
        max(0, min(10, int(digits)))
    )


def valid_dataframe(df, minimum=60):

    if not isinstance(df, pd.DataFrame):
        return False

    if len(df) < minimum:
        return False

    required = (
        "open", "high", "low", "close"
    )

    return all(
        col in df.columns
        for col in required
    )


# ============================================================
# TELEGRAM
# ============================================================

async def telegram(message):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:

        async with httpx.AsyncClient(
            timeout=10
        ) as client:

            response = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message[:3900]
                }
            )

            if response.status_code >= 400:

                log.warning(
                    "Telegram HTTP %s: %s",
                    response.status_code,
                    response.text[:300]
                )

    except Exception as exc:

        log.warning(
            "Telegram error: %s",
            exc
        )


# ============================================================
# RISK STATE
# ============================================================

def load_state():

    try:

        with open(RISK_STATE_FILE, "r") as f:

            result = json.load(f)

            if isinstance(result, dict):
                return result

    except Exception:
        pass

    return {}


def save_state(data):

    try:

        directory = os.path.dirname(
            RISK_STATE_FILE
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        temp = RISK_STATE_FILE + ".tmp"

        with open(temp, "w") as f:
            json.dump(data, f)

        os.replace(
            temp,
            RISK_STATE_FILE
        )

    except Exception as exc:

        log.warning(
            "Risk state save failed: %s",
            exc
        )


class RiskManager:

    def __init__(self):

        self.state = load_state()

        self.today = datetime.now(
            timezone.utc
        ).date().isoformat()

        self.daily_halt = False

        self.drawdown_halt = bool(
            self.state.get(
                "drawdown_halt", False
            )
        )

        if self.state.get("date") != self.today:

            self.state["date"] = self.today
            self.state["daily_halt"] = False
            self.state["day_start_equity"] = 0

            self.daily_halt = False

        else:

            self.daily_halt = bool(
                self.state.get(
                    "daily_halt", False
                )
            )

        self.day_start_equity = number(
            self.state.get("day_start_equity")
        )

        self.peak_equity = number(
            self.state.get("peak_equity")
        )

        self.save()

    def save(self):

        self.state.update({

            "date": self.today,

            "daily_halt": self.daily_halt,

            "drawdown_halt": self.drawdown_halt,

            "day_start_equity": self.day_start_equity,

            "peak_equity": self.peak_equity

        })

        save_state(self.state)

    def update(self, equity):

        if equity <= 0:
            return False, "Invalid account equity"

        if self.day_start_equity <= 0:
            self.day_start_equity = equity

        if self.peak_equity <= 0:
            self.peak_equity = equity

        self.peak_equity = max(
            self.peak_equity,
            equity
        )

        daily_loss = max(
            0,
            (
                self.day_start_equity - equity
            ) / self.day_start_equity * 100
        )

        drawdown = max(
            0,
            (
                self.peak_equity - equity
            ) / self.peak_equity * 100
        )

        if daily_loss >= MAX_DAILY_LOSS_PERCENT:
            self.daily_halt = True

        if drawdown >= MAX_DRAWDOWN_PERCENT:
            self.drawdown_halt = True

        self.save()

        if self.drawdown_halt:

            return False, (
                f"Drawdown halt {drawdown:.2f}%"
            )

        if self.daily_halt:

            return False, (
                f"Daily loss halt {daily_loss:.2f}%"
            )

        return True, (
            f"Daily={daily_loss:.2f}% "
            f"DD={drawdown:.2f}%"
        )

    def calculate_lot(
        self,
        equity,
        entry,
        stop,
        spec
    ):

        tick_size = number(
            spec.get("tickSize")
        )

        tick_value = number(
            spec.get("tickValueAccount")
        )

        volume_min = number(
            spec.get("volumeMin")
        )

        volume_max = number(
            spec.get("volumeMax")
        )

        volume_step = number(
            spec.get("volumeStep")
        )

        if min(
            tick_size,
            tick_value,
            volume_min,
            volume_max,
            volume_step
        ) <= 0:

            return 0

        risk_cash = (
            equity * RISK_PERCENT / 100
        )

        risk_per_lot = (
            abs(entry - stop) / tick_size
        ) * tick_value

        if risk_per_lot <= 0:
            return 0

        lots = floor_step(
            risk_cash / risk_per_lot,
            volume_step
        )

        lots = min(
            lots,
            volume_max
        )

        if lots < volume_min:
            return 0

        if volume_step < 1:

            precision = max(
                0,
                min(
                    8,
                    int(
                        round(
                            -math.log10(volume_step)
                        )
                    )
                )
            )

        else:
            precision = 4

        return round(
            lots,
            precision
        )


# ============================================================
# TICKERALL API
# ============================================================

class TickerAllAPI:

    def __init__(self):

        self.account_id = None
        self.is_demo = False

        self.client = httpx.AsyncClient(

            base_url=BASE_URL,

            timeout=HTTP_TIMEOUT,

            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10
            ),

            headers={

                "Authorization": (
                    f"Bearer {API_KEY}"
                ),

                "Content-Type": "application/json",

                "Accept": "application/json"

            }
        )

    async def close(self):

        await self.client.aclose()

    async def request(
        self,
        method,
        path,
        params=None,
        body=None,
        write=False
    ):

        headers = {}

        if write:

            headers["Idempotency-Key"] = str(
                uuid.uuid4()
            )

        response = await self.client.request(

            method,

            path,

            params=params,

            json=body,

            headers=headers

        )

        if response.status_code >= 400:

            raise RuntimeError(

                f"TickerAll HTTP "
                f"{response.status_code} "
                f"{path}: "
                f"{response.text[:700]}"

            )

        if response.status_code == 204:
            return {}

        if not response.content:
            return {}

        return response.json()

    def account_path(self, suffix=""):

        if not self.account_id:

            raise RuntimeError(
                "No connected account"
            )

        return (
            f"/v1/accounts/"
            f"{self.account_id}{suffix}"
        )

    async def connect(self):

        missing = []

        required = {

            "TICKERALL_API_KEY": API_KEY,

            "MT5_SERVER": MT5_SERVER,

            "MT5_LOGIN": MT5_LOGIN,

            "MT5_PASSWORD": MT5_PASSWORD

        }

        for name, value in required.items():

            if not value:
                missing.append(name)

        if missing:

            raise RuntimeError(
                "Missing Railway variables: "
                + ", ".join(missing)
            )

        payload = {

            "broker": BROKER,

            "server": MT5_SERVER,

            "account": MT5_LOGIN,

            "password": MT5_PASSWORD,

            "terminalType": "MOBILE"

        }

        result = await self.request(

            "POST",

            "/v1/sessions",

            body=payload

        )

        self.account_id = result.get(
            "accountId"
        )

        self.is_demo = (
            result.get("isDemo") is True
        )

        if not self.account_id:

            raise RuntimeError(
                "No accountId returned from session"
            )

        if not self.is_demo:

            raise RuntimeError(
                "DEMO SAFETY STOP: "
                "TickerAll did not confirm "
                "isDemo=true"
            )

        log.info(
            "SESSION CONNECTED | "
            "accountId=%s | DEMO=True",
            self.account_id
        )

    async def account(self):

        result = await self.request(

            "GET",

            self.account_path()

        )

        if result.get("isDemo") is not True:

            self.is_demo = False

            raise RuntimeError(
                "Account is not verified demo"
            )

        if str(
            result.get("status", "")
        ).lower() != "online":

            raise RuntimeError(
                "Account not online: "
                f"{result.get('status')}"
            )

        return result

    async def symbols(self):

        result = await self.request(

            "GET",

            self.account_path("/symbols")

        )

        rows = result.get(
            "symbols", []
        )

        output = []

        for item in rows:

            if isinstance(item, dict):

                name = (
                    item.get("name")
                    or item.get("symbol")
                )

            else:
                name = item

            if name:
                output.append(str(name))

        return output

    async def specs(self):

        result = await self.request(

            "GET",

            self.account_path("/symbol-specs")

        )

        rows = result.get(
            "specs", []
        )

        return {

            str(item["name"]): item

            for item in rows

            if isinstance(item, dict)
            and item.get("name")

        }

    async def candles(
        self,
        symbol,
        timeframe
    ):

        result = await self.request(

            "GET",

            self.account_path("/candles"),

            params={

                "symbol": symbol,

                "timeframe": timeframe,

                "hours": MAX_CANDLE_HOURS

            }

        )

        rows = result.get(
            "candles", []
        )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        required = [
            "open", "high", "low", "close"
        ]

        if not all(
            col in df.columns
            for col in required
        ):

            return pd.DataFrame()

        for col in required:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=required
        )

        if "timestamp" in df.columns:

            df = df.sort_values(
                "timestamp"
            )

            df = df.drop_duplicates(
                "timestamp"
            )

        return df.reset_index(
            drop=True
        )

    async def quote(self, symbol):

        return await self.request(

            "GET",

            self.account_path(
                f"/symbols/{symbol}/quote"
            )

        )

    async def positions(self):

        result = await self.account()

        rows = result.get(
            "positions", []
        )

        return (
            rows
            if isinstance(rows, list)
            else []
        )

    async def place_order(
        self,
        symbol,
        side,
        volume,
        sl,
        tp
    ):

        if not (

            ENABLE_TRADING

            and ALLOW_DEMO_TRADING

            and self.is_demo

        ):

            raise RuntimeError(
                "Order blocked: "
                "demo verification or flags missing"
            )

        return await self.request(

            "POST",

            self.account_path("/orders"),

            body={

                "type": "market",

                "symbol": symbol,

                "side": side,

                "volume": volume,

                "stopLoss": sl,

                "takeProfit": tp,

                "comment": (
                    "TickerAll V10 H1 H4"
                )

            },

            write=True

        )


# ============================================================
# INDICATORS
# ============================================================

def indicators(df):

    x = df.copy()

    for period in (
        9, 21, 50, 200
    ):

        x[f"ema{period}"] = (
            x["close"].ewm(
                span=period,
                adjust=False
            ).mean()
        )

    delta = x["close"].diff()

    gain = delta.clip(
        lower=0
    ).ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    loss = (
        -delta.clip(lower=0)
    ).ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = gain / loss.replace(
        0, np.nan
    )

    x["rsi"] = (
        100 - 100 / (1 + rs)
    ).fillna(50)

    prev = x["close"].shift(1)

    tr = pd.concat([

        x["high"] - x["low"],

        (x["high"] - prev).abs(),

        (x["low"] - prev).abs()

    ], axis=1).max(axis=1)

    x["atr"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    up = x["high"].diff()

    down = -x["low"].diff()

    plus_dm = pd.Series(

        np.where(
            (up > down) & (up > 0),
            up,
            0
        ),

        index=x.index

    )

    minus_dm = pd.Series(

        np.where(
            (down > up) & (down > 0),
            down,
            0
        ),

        index=x.index

    )

    atr_safe = x["atr"].replace(
        0, np.nan
    )

    plus_di = (

        100 * plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean() / atr_safe

    )

    minus_di = (

        100 * minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean() / atr_safe

    )

    dx = (

        100 * (
            plus_di - minus_di
        ).abs()

        / (
            plus_di + minus_di
        ).replace(0, np.nan)

    )

    x["plus_di"] = plus_di.fillna(0)

    x["minus_di"] = minus_di.fillna(0)

    x["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean().fillna(0)

    x["bb_mid"] = (
        x["close"].rolling(20).mean()
    )

    std = x["close"].rolling(
        20
    ).std()

    x["bb_upper"] = (
        x["bb_mid"] + 2 * std
    )

    x["bb_lower"] = (
        x["bb_mid"] - 2 * std
    )

    x["donchian_high"] = (
        x["high"].rolling(20).max().shift(1)
    )

    x["donchian_low"] = (
        x["low"].rolling(20).min().shift(1)
    )

    x["body"] = (
        x["close"] - x["open"]
    ).abs()

    x["body_atr"] = (

        x["body"]
        / x["atr"].replace(0, np.nan)

    ).fillna(0)

    x["bullish"] = (
        x["close"] > x["open"]
    )

    x["bearish"] = (
        x["close"] < x["open"]
    )

    return x


# ============================================================
# H1 SIGNAL ENGINE
# ============================================================

def evaluate_signal(df):

    if not valid_dataframe(
        df, MIN_H1_CANDLES
    ):

        return (
            None,
            0,
            0,
            "DATA",
            "Insufficient H1 candles",
            "NONE"
        )

    x = indicators(df)

    # Completed H1 candles only.
    row = x.iloc[-2]

    previous = x.iloc[-3]

    atr = number(row["atr"])

    close = number(row["close"])

    if atr <= 0 or close <= 0:

        return (
            None,
            0,
            atr,
            "INVALID",
            "Invalid H1 ATR or close",
            "NONE"
        )

    adx = number(row["adx"])

    rsi = number(row["rsi"])

    ema9 = number(row["ema9"])

    ema21 = number(row["ema21"])

    ema50 = number(row["ema50"])

    ema200 = number(row["ema200"])

    plus_di = number(row["plus_di"])

    minus_di = number(row["minus_di"])

    buy = 0

    sell = 0

    buy_reasons = []

    sell_reasons = []

    buy_type = "NONE"

    sell_type = "NONE"

    trending = (
        adx >= ADX_TREND_THRESHOLD
    )

    ranging = (
        adx <= ADX_RANGE_THRESHOLD
    )

    regime = (

        "TREND" if trending

        else "RANGE" if ranging

        else "TRANSITION"

    )

    # --------------------------------------------------------
    # TREND CONTINUATION
    # --------------------------------------------------------

    if trending:

        if (

            ema9 > ema21 > ema50

            and close > ema50

            and plus_di > minus_di

        ):

            buy += 3

            buy_type = "TREND"

            buy_reasons.append(
                "EMA trend + DI"
            )

        if (

            ema9 < ema21 < ema50

            and close < ema50

            and minus_di > plus_di

        ):

            sell += 3

            sell_type = "TREND"

            sell_reasons.append(
                "EMA trend + DI"
            )

        if (
            close > ema200
            and ema50 > ema200
        ):

            buy += 1

            buy_reasons.append(
                "EMA200 bullish alignment"
            )

        if (
            close < ema200
            and ema50 < ema200
        ):

            sell += 1

            sell_reasons.append(
                "EMA200 bearish alignment"
            )

        if 50 < rsi < 72:

            buy += 1

            buy_reasons.append(
                "RSI bullish momentum"
            )

        if 28 < rsi < 50:

            sell += 1

            sell_reasons.append(
                "RSI bearish momentum"
            )

        # Pullback recovery.
        if (

            ema9 > ema21

            and previous["close"] <= previous["ema9"]

            and close > ema9

            and row["bullish"]

        ):

            buy += 2

            buy_type = "PULLBACK"

            buy_reasons.append(
                "Pullback recovery"
            )

        if (

            ema9 < ema21

            and previous["close"] >= previous["ema9"]

            and close < ema9

            and row["bearish"]

        ):

            sell += 2

            sell_type = "PULLBACK"

            sell_reasons.append(
                "Pullback recovery"
            )

    # --------------------------------------------------------
    # RANGE REVERSAL
    # --------------------------------------------------------

    if ranging:

        lower = number(
            row["bb_lower"]
        )

        upper = number(
            row["bb_upper"]
        )

        if (

            lower > 0

            and close > lower

            and previous["close"] <= previous["bb_lower"]

            and rsi < 45

            and row["bullish"]

        ):

            buy += 3

            buy_type = "RANGE"

            buy_reasons.append(
                "Lower band recovery"
            )

        if (

            upper > 0

            and close < upper

            and previous["close"] >= previous["bb_upper"]

            and rsi > 55

            and row["bearish"]

        ):

            sell += 3

            sell_type = "RANGE"

            sell_reasons.append(
                "Upper band recovery"
            )

    # --------------------------------------------------------
    # DONCHIAN BREAKOUT
    # --------------------------------------------------------

    if adx >= 15:

        high = number(
            row["donchian_high"]
        )

        low = number(
            row["donchian_low"]
        )

        body_atr = number(
            row["body_atr"]
        )

        if (

            high > 0

            and close > high

            and row["bullish"]

            and body_atr >= 0.25

            and plus_di > minus_di

        ):

            buy += 2

            buy_type = "BREAKOUT"

            buy_reasons.append(
                "Donchian breakout"
            )

        if (

            low > 0

            and close < low

            and row["bearish"]

            and body_atr >= 0.25

            and minus_di > plus_di

        ):

            sell += 2

            sell_type = "BREAKOUT"

            sell_reasons.append(
                "Donchian breakout"
            )

    # --------------------------------------------------------
    # EXTENSION FILTER
    # --------------------------------------------------------

    extension = (
        abs(close - ema21) / atr
    )

    if extension > 2.8:

        buy = max(0, buy - 2)

        sell = max(0, sell - 2)

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if (
        buy >= MIN_SIGNAL_SCORE
        and buy > sell
    ):

        return (

            "BUY",

            buy,

            atr,

            regime,

            ", ".join(buy_reasons),

            buy_type

        )

    if (
        sell >= MIN_SIGNAL_SCORE
        and sell > buy
    ):

        return (

            "SELL",

            sell,

            atr,

            regime,

            ", ".join(sell_reasons),

            sell_type

        )

    return (

        None,

        max(buy, sell),

        atr,

        regime,

        (
            f"No confirmed edge | "
            f"BUY={buy} SELL={sell} "
            f"RSI={rsi:.1f} ADX={adx:.1f}"
        ),

        "NONE"

    )


# ============================================================
# H4 CONFIRMATION ENGINE
# ============================================================

def confirm_h4(df, side):

    if not valid_dataframe(
        df, MIN_H4_CANDLES
    ):

        return (
            False,
            "Insufficient H4 history"
        )

    x = indicators(df)

    # Last completed H4 candle.
    row = x.iloc[-2]

    previous = x.iloc[-6]

    close = number(row["close"])

    ema50 = number(row["ema50"])

    ema200 = number(row["ema200"])

    previous_ema50 = number(
        previous["ema50"]
    )

    if min(
        close,
        ema50,
        ema200,
        previous_ema50
    ) <= 0:

        return (
            False,
            "Invalid H4 indicators"
        )

    adx = number(row["adx"])

    bullish = (

        close > ema50

        and ema50 > ema200

        and ema50 > previous_ema50

    )

    bearish = (

        close < ema50

        and ema50 < ema200

        and ema50 < previous_ema50

    )

    if side == "BUY":

        if bullish:

            return (
                True,
                (
                    f"H4 bullish alignment "
                    f"ADX={adx:.1f}"
                )
            )

        return (
            False,
            (
                f"H4 BUY rejected | "
                f"close={close:.5f} "
                f"EMA50={ema50:.5f} "
                f"EMA200={ema200:.5f}"
            )
        )

    if side == "SELL":

        if bearish:

            return (
                True,
                (
                    f"H4 bearish alignment "
                    f"ADX={adx:.1f}"
                )
            )

        return (
            False,
            (
                f"H4 SELL rejected | "
                f"close={close:.5f} "
                f"EMA50={ema50:.5f} "
                f"EMA200={ema200:.5f}"
            )
        )

    return (
        False,
        "Unknown signal direction"
    )


# ============================================================
# TRADING BOT
# ============================================================

class TradingBot:

    def __init__(self):

        self.api = TickerAllAPI()

        self.risk = RiskManager()

        self.symbols = []

        self.specs = {}

        self.last_trade = {}

        self.cycle = 0

        self.scan_semaphore = asyncio.Semaphore(
            max(1, MAX_CONCURRENT_SCANS)
        )

    # --------------------------------------------------------
    # SYMBOL DISCOVERY
    # --------------------------------------------------------

    async def discover_symbols(self):

        available = await self.api.symbols()

        self.specs = await self.api.specs()

        selected = []

        for desired in TARGET_SYMBOLS:

            matches = [

                s for s in available

                if symbol_matches(
                    s,
                    desired
                )

            ]

            if matches:

                matches.sort(

                    key=lambda s: (

                        normalize_symbol(s)
                        != normalize_symbol(desired),

                        len(s)

                    )

                )

                selected.append(
                    matches[0]
                )

            else:

                log.warning(
                    "SYMBOL NOT FOUND: %s",
                    desired
                )

        # One broker-native symbol per metal group.
        for group in METAL_GROUPS:

            matches = [

                s for s in available

                if any(

                    symbol_matches(s, alias)

                    for alias in group

                )

            ]

            if matches:

                matches.sort(

                    key=lambda s: (

                        normalize_symbol(s) not in [

                            normalize_symbol(a)

                            for a in group

                        ],

                        len(s)

                    )

                )

                selected.append(
                    matches[0]
                )

            else:

                log.warning(
                    "METAL NOT FOUND: %s",
                    group[0]
                )

        self.symbols = list(
            dict.fromkeys(selected)
        )

        log.info(
            "DISCOVERY | available=%d matched=%d",
            len(available),
            len(self.symbols)
        )

        log.info(
            "SYMBOL LIST: %s",
            ", ".join(self.symbols)
        )

        if not self.symbols:

            raise RuntimeError(
                "No broker symbols matched"
            )

    def get_spec(self, symbol):

        if symbol in self.specs:
            return self.specs[symbol]

        for name, spec in self.specs.items():

            if name.lower() == symbol.lower():
                return spec

        return None

    # --------------------------------------------------------
    # ACCOUNT SNAPSHOT
    # --------------------------------------------------------

    async def account_snapshot(self):

        snapshot = await self.api.account()

        account = snapshot.get("account")

        if not isinstance(account, dict):
            account = snapshot

        equity = number(
            account.get("equity")
        )

        balance = number(
            account.get("balance")
        )

        if equity <= 0:
            equity = balance

        positions = snapshot.get(
            "positions", []
        )

        if not isinstance(positions, list):
            positions = []

        return (
            equity,
            balance,
            positions,
            snapshot
        )

    # --------------------------------------------------------
    # ANALYZE ONE SYMBOL
    #
    # This function ONLY scans and generates signals.
    # It does not place orders.
    # --------------------------------------------------------

    async def analyze_symbol(self, symbol):

        async with self.scan_semaphore:

            try:

                # Fetch H1 and H4 history.
                h1_task = self.api.candles(
                    symbol,
                    TIMEFRAME
                )

                h4_task = self.api.candles(
                    symbol,
                    CONFIRMATION_TIMEFRAME
                )

                h1, h4 = await asyncio.gather(
                    h1_task,
                    h4_task
                )

                if not valid_dataframe(
                    h1,
                    MIN_H1_CANDLES
                ):

                    log.warning(
                        "DATA INSUFFICIENT | %s | "
                        "H1 candles=%d need=%d",
                        symbol,
                        len(h1),
                        MIN_H1_CANDLES
                    )

                    return {
                        "symbol": symbol,
                        "signal": None,
                        "status": "H1_DATA"
                    }

                if not valid_dataframe(
                    h4,
                    MIN_H4_CANDLES
                ):

                    log.warning(
                        "DATA INSUFFICIENT | %s | "
                        "H4 candles=%d need=%d",
                        symbol,
                        len(h4),
                        MIN_H4_CANDLES
                    )

                    return {
                        "symbol": symbol,
                        "signal": None,
                        "status": "H4_DATA"
                    }

                (
                    side,
                    score,
                    atr,
                    regime,
                    reason,
                    signal_type

                ) = evaluate_signal(h1)

                if not side:

                    log.info(
                        "SCAN %s | H1=%s | H4=CHECKED | "
                        "NO SIGNAL | %s",
                        symbol,
                        regime,
                        reason
                    )

                    return {
                        "symbol": symbol,
                        "signal": None,
                        "status": "NO_SIGNAL"
                    }

                log.info(
                    "H1 CANDIDATE | %s %s | "
                    "score=%d | regime=%s | type=%s | %s",
                    side,
                    symbol,
                    score,
                    regime,
                    signal_type,
                    reason
                )

                # H4 confirmation.
                if USE_H4_CONFIRMATION:

                    confirmed, confirmation_reason = (
                        confirm_h4(
                            h4,
                            side
                        )
                    )

                    if not confirmed:

                        log.info(
                            "H4 REJECTED | %s %s | %s",
                            side,
                            symbol,
                            confirmation_reason
                        )

                        return {
                            "symbol": symbol,
                            "signal": None,
                            "status": "H4_REJECTED"
                        }

                    log.info(
                        "H4 CONFIRMED | %s %s | %s",
                        side,
                        symbol,
                        confirmation_reason
                    )

                return {

                    "symbol": symbol,

                    "signal": side,

                    "score": score,

                    "atr": atr,

                    "regime": regime,

                    "reason": reason,

                    "signal_type": signal_type,

                    "status": "CONFIRMED"

                }

            except Exception as exc:

                log.exception(
                    "SCAN ERROR | %s | %s",
                    symbol,
                    exc
                )

                return {
                    "symbol": symbol,
                    "signal": None,
                    "status": "ERROR",
                    "error": str(exc)
                }

    # --------------------------------------------------------
    # EXECUTE A CONFIRMED SIGNAL
    # --------------------------------------------------------

    async def execute_signal(
        self,
        candidate,
        equity,
        positions,
        risk_allowed
    ):

        symbol = candidate["symbol"]

        side = candidate["signal"]

        score = candidate["score"]

        atr = candidate["atr"]

        regime = candidate["regime"]

        reason = candidate["reason"]

        signal_type = candidate["signal_type"]

        # Scanning has already happened.
        # The following restrictions apply only to orders.

        if not (
            ENABLE_TRADING
            and ALLOW_DEMO_TRADING
            and self.api.is_demo
        ):

            log.info(
                "ORDER BLOCKED | %s %s | SIGNAL-ONLY MODE",
                side,
                symbol
            )

            return False

        if not risk_allowed:

            log.warning(
                "ORDER BLOCKED | %s | risk protection active",
                symbol
            )

            return False

        now = time.time()

        last = self.last_trade.get(
            symbol, 0
        )

        if (
            now - last
            < SYMBOL_COOLDOWN_SECONDS
        ):

            log.info(
                "ORDER BLOCKED | %s | cooldown",
                symbol
            )

            return False

        same_symbol = [

            p for p in positions

            if str(
                p.get("symbol", "")
            ).lower() == symbol.lower()

        ]

        if (
            len(same_symbol)
            >= MAX_POSITIONS_PER_SYMBOL
        ):

            log.info(
                "ORDER BLOCKED | %s | existing position",
                symbol
            )

            return False

        if len(positions) >= MAX_POSITIONS:

            log.info(
                "ORDER BLOCKED | %s | position limit",
                symbol
            )

            return False

        # Contract specifications.
        spec = self.get_spec(symbol)

        if not spec:

            log.warning(
                "ORDER BLOCKED | %s | missing specifications",
                symbol
            )

            return False

        tick_size = number(
            spec.get("tickSize")
        )

        point = number(
            spec.get("point")
        )

        stop_level = number(
            spec.get("stopLevel")
        )

        digits = int(
            number(
                spec.get("digits"),
                5
            )
        )

        if tick_size <= 0 or point <= 0:

            log.warning(
                "ORDER BLOCKED | %s | invalid tick size/point",
                symbol
            )

            return False

        # Current quote.
        quote = await self.api.quote(
            symbol
        )

        bid = number(
            quote.get("bid")
        )

        ask = number(
            quote.get("ask")
        )

        if (
            bid <= 0
            or ask <= 0
            or ask < bid
        ):

            log.info(
                "ORDER BLOCKED | %s | invalid quote",
                symbol
            )

            return False

        spread = ask - bid

        if spread > atr * MAX_SPREAD_ATR:

            log.info(
                "ORDER BLOCKED | %s | "
                "spread %.8f > limit %.8f",
                symbol,
                spread,
                atr * MAX_SPREAD_ATR
            )

            return False

        entry = (
            ask if side == "BUY"
            else bid
        )

        # Stops.
        minimum_distance = max(

            stop_level * point + 2 * point,

            2 * point

        )

        stop_dist = max(

            ATR_SL_MULT * atr,

            minimum_distance

        )

        tp_dist = max(

            ATR_TP_MULT * atr,

            stop_dist * MIN_RR

        )

        if side == "BUY":

            sl = entry - stop_dist

            tp = entry + tp_dist

        else:

            sl = entry + stop_dist

            tp = entry - tp_dist

        sl = round_price(
            sl,
            digits,
            tick_size
        )

        tp = round_price(
            tp,
            digits,
            tick_size
        )

        # Broker stop geometry.
        if side == "BUY":

            if sl >= bid - minimum_distance:

                log.info(
                    "ORDER BLOCKED | %s | invalid BUY SL",
                    symbol
                )

                return False

            if tp <= ask + minimum_distance:

                log.info(
                    "ORDER BLOCKED | %s | invalid BUY TP",
                    symbol
                )

                return False

        else:

            if sl <= ask + minimum_distance:

                log.info(
                    "ORDER BLOCKED | %s | invalid SELL SL",
                    symbol
                )

                return False

            if tp >= bid - minimum_distance:

                log.info(
                    "ORDER BLOCKED | %s | invalid SELL TP",
                    symbol
                )

                return False

        # Position sizing.
        lots = self.risk.calculate_lot(

            equity,

            entry,

            sl,

            spec

        )

        if lots <= 0:

            log.warning(
                "ORDER BLOCKED | %s | "
                "safe volume unavailable",
                symbol
            )

            return False

        # Final account verification.
        latest = await self.api.account()

        if latest.get("isDemo") is not True:

            raise RuntimeError(
                "Final demo verification failed"
            )

        latest_positions = latest.get(
            "positions", []
        )

        if not isinstance(
            latest_positions, list
        ):

            latest_positions = []

        if len(latest_positions) >= MAX_POSITIONS:

            log.info(
                "ORDER BLOCKED | %s | final position cap",
                symbol
            )

            return False

        symbol_count = sum(

            1 for p in latest_positions

            if str(
                p.get("symbol", "")
            ).lower() == symbol.lower()

        )

        if (
            symbol_count
            >= MAX_POSITIONS_PER_SYMBOL
        ):

            log.info(
                "ORDER BLOCKED | %s | final symbol cap",
                symbol
            )

            return False

        # Submit order.
        log.info(

            "SENDING ORDER | %s %s | "
            "lots=%s | entry=%s | SL=%s | TP=%s",

            side,
            symbol,
            lots,
            entry,
            sl,
            tp

        )

        result = await self.api.place_order(

            symbol,

            side,

            lots,

            sl,

            tp

        )

        status = str(
            result.get("status", "")
        ).lower()

        ticket = result.get("ticket")

        if (
            status != "open"
            and not ticket
        ):

            log.error(
                "ORDER NOT CONFIRMED | %s",
                result
            )

            return False

        self.last_trade[symbol] = time.time()

        log.info(

            "ORDER OPENED | %s %s | "
            "lots=%s | ticket=%s",

            side,
            symbol,
            lots,
            ticket

        )

        await telegram(

            "TICKERALL V10 DEMO ORDER\n"

            f"{side} {symbol}\n"

            f"Timeframe: H1 / H4\n"

            f"Type: {signal_type}\n"

            f"Regime: {regime}\n"

            f"Score: {score}\n"

            f"Volume: {lots}\n"

            f"Entry: {entry}\n"

            f"SL: {sl}\n"

            f"TP: {tp}\n"

            f"Reason: {reason}\n"

            f"Ticket: {ticket}"

        )

        return True

    # --------------------------------------------------------
    # COMPLETE SCAN CYCLE
    # --------------------------------------------------------

    async def run_cycle(self):

        self.cycle += 1

        started = time.time()

        log.info("=" * 70)

        log.info(
            "STARTING SCAN CYCLE %d",
            self.cycle
        )

        log.info(
            "TIMEFRAMES | ENTRY=%s | CONFIRMATION=%s",
            TIMEFRAME,
            CONFIRMATION_TIMEFRAME
        )

        # Account snapshot.
        equity, balance, positions, snapshot = (
            await self.account_snapshot()
        )

        risk_allowed, risk_message = (
            self.risk.update(equity)
        )

        log.info(

            "ACCOUNT | DEMO=%s | equity=%.2f | "
            "balance=%.2f | positions=%d/%d | %s",

            snapshot.get("isDemo"),

            equity,

            balance,

            len(positions),

            MAX_POSITIONS,

            risk_message

        )

        # IMPORTANT:
        # Never return here because of disabled trading,
        # position limits, cooldowns or risk halts.
        # Every symbol is still analyzed.

        scan_list = self.symbols

        if MAX_SYMBOLS_PER_CYCLE > 0:

            scan_list = scan_list[
                :MAX_SYMBOLS_PER_CYCLE
            ]

        log.info(
            "SCANNING ALL MATCHED SYMBOLS | count=%d",
            len(scan_list)
        )

        # Analyze all symbols concurrently with a limit.
        tasks = [

            self.analyze_symbol(symbol)

            for symbol in scan_list

        ]

        results = await asyncio.gather(
            *tasks
        )

        # Cycle statistics.
        confirmed = [

            item for item in results

            if item.get("status") == "CONFIRMED"

        ]

        buy_signals = [

            item for item in confirmed

            if item.get("signal") == "BUY"

        ]

        sell_signals = [

            item for item in confirmed

            if item.get("signal") == "SELL"

        ]

        no_signals = sum(

            item.get("status") == "NO_SIGNAL"

            for item in results

        )

        rejected = sum(

            item.get("status") == "H4_REJECTED"

            for item in results

        )

        data_errors = sum(

            item.get("status") in (
                "H1_DATA",
                "H4_DATA"
            )

            for item in results

        )

        errors = sum(

            item.get("status") == "ERROR"

            for item in results

        )

        log.info("=" * 70)

        log.info(
            "SCAN SUMMARY | scanned=%d | "
            "confirmed=%d | BUY=%d | SELL=%d | "
            "no-signal=%d | H4-rejected=%d | "
            "data-shortage=%d | errors=%d",
            len(results),
            len(confirmed),
            len(buy_signals),
            len(sell_signals),
            no_signals,
            rejected,
            data_errors,
            errors
        )

        # Display every confirmed signal.
        for item in confirmed:

            log.info(

                "CONFIRMED SIGNAL | %s %s | "
                "score=%d | type=%s | %s",

                item["signal"],

                item["symbol"],

                item["score"],

                item["signal_type"],

                item["reason"]

            )

        # Telegram summary.
        if confirmed:

            lines = [

                "TICKERALL V10 H1/H4 SCAN",

                f"Scanned: {len(results)}",

                f"Confirmed: {len(confirmed)}",

                f"BUY: {len(buy_signals)}",

                f"SELL: {len(sell_signals)}",

                ""

            ]

            for item in confirmed[:20]:

                lines.append(

                    f'{item["signal"]} '
                    f'{item["symbol"]} '
                    f'Score={item["score"]} '
                    f'{item["signal_type"]}'

                )

            if len(confirmed) > 20:

                lines.append(
                    f"... and {len(confirmed) - 20} more"
                )

            await telegram(
                "\n".join(lines)
            )

        else:

            log.info(
                "NO H1/H4 CONFIRMED SIGNALS THIS CYCLE"
            )

        # Signal-only mode still scans everything.
        if not (

            ENABLE_TRADING

            and ALLOW_DEMO_TRADING

            and self.api.is_demo

        ):

            log.warning(

                "SIGNAL-ONLY MODE | "
                "Scanning active; order placement disabled"

            )

        elif not risk_allowed:

            log.warning(

                "RISK HALT | "
                "Scanning active; new orders blocked: %s",

                risk_message

            )

        else:

            # Process confirmed candidates one by one.
            # Refresh positions after every successful order.
            opened = 0

            for candidate in confirmed:

                try:

                    current_positions = (
                        await self.api.positions()
                    )

                    success = await self.execute_signal(

                        candidate,

                        equity,

                        current_positions,

                        risk_allowed

                    )

                    if success:

                        opened += 1

                except Exception as exc:

                    log.exception(

                        "EXECUTION ERROR | %s | %s",

                        candidate["symbol"],

                        exc

                    )

            log.info(
                "ORDER SUMMARY | opened=%d",
                opened
            )

        elapsed = time.time() - started

        log.info(

            "CYCLE COMPLETE | cycle=%d | "
            "symbols=%d | confirmed=%d | "
            "elapsed=%.1fs",

            self.cycle,

            len(results),

            len(confirmed),

            elapsed

        )

        log.info("=" * 70)

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    async def run(self):

        log.info("=" * 70)

        log.info(
            "TICKERALL ADAPTIVE ACTIVE SCALPER V10.0"
        )

        log.info(
            "ENTRY TIMEFRAME: H1"
        )

        log.info(
            "CONFIRMATION TIMEFRAME: H4"
        )

        log.info(
            "H4 CONFIRMATION: %s",
            USE_H4_CONFIRMATION
        )

        log.info(
            "TRADING FLAGS: %s / %s",
            ENABLE_TRADING,
            ALLOW_DEMO_TRADING
        )

        log.info(
            "RISK PER TRADE: %.3f%%",
            RISK_PERCENT
        )

        log.info(
            "CANDLE HISTORY HOURS: %d",
            MAX_CANDLE_HOURS
        )

        log.info(
            "MIN H1/H4 CANDLES: %d / %d",
            MIN_H1_CANDLES,
            MIN_H4_CANDLES
        )

        log.info(
            "SCAN INTERVAL: %d seconds",
            SCAN_INTERVAL
        )

        log.info("=" * 70)

        # Connect and verify demo.
        await self.api.connect()

        # Initial symbol discovery.
        await self.discover_symbols()

        await telegram(

            "TickerAll V10 connected.\n"

            "DEMO=True\n"

            "Entry: H1\n"

            "Confirmation: H4\n"

            f"Matched symbols: {len(self.symbols)}\n"

            f"Trading flags: "
            f"{ENABLE_TRADING}/{ALLOW_DEMO_TRADING}"

        )

        while True:

            started = time.time()

            try:

                await self.run_cycle()

            except Exception as exc:

                log.exception(
                    "CYCLE ERROR: %s",
                    exc
                )

                await telegram(

                    "TickerAll V10 cycle error:\n"

                    f"{str(exc)[:700]}"

                )

                message = str(exc).lower()

                if (

                    "broker_account_not_hot" in message

                    or "offline" in message

                ):

                    try:

                        log.warning(
                            "Reconnecting session..."
                        )

                        await self.api.connect()

                        await self.discover_symbols()

                    except Exception as reconnect_error:

                        log.error(

                            "Reconnect failed: %s",

                            reconnect_error

                        )

            elapsed = time.time() - started

            sleep_time = max(
                5,
                SCAN_INTERVAL - elapsed
            )

            log.info(
                "NEXT FULL SCAN IN %.1f SECONDS",
                sleep_time
            )

            await asyncio.sleep(
                sleep_time
            )


# ============================================================
# ENTRY POINT
# ============================================================

async def main():

    bot = TradingBot()

    try:

        await bot.run()

    finally:

        await bot.api.close()


if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        log.info(
            "Bot stopped."
        )

    except Exception:

        log.exception(
            "FATAL STARTUP ERROR"
        )

        raise
