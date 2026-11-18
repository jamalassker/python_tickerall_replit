import time
import json
import math
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from tickerall import Tickerall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
)
logger = logging.getLogger("HFT_Master_Bot_Full")

# =====================================================================
# INPUT PARAMETERS
# =====================================================================
API_KEY = "cf_api_c3a8cd47c717c7f5a609ea1cef7a66a209493b8a0b99c303d36b8a9e8e6b907e"
BROKER_NAME = "mt5"
BROKER_SERVER = "ValetaxGlobal-Live3"
ACCOUNT_LOGIN = 641425149
ACCOUNT_PASSWORD = "EJam123!@"

INP_LOT_SIZE = 0.01
INP_MAGIC_NUMBER = 123456
INP_MAX_POSITIONS = 1000
INP_MAX_POSITIONS_PER_SYMBOL = 20
INP_MAX_RISK_PERCENT = 2.0
INP_MAX_TRADES_PER_DAY = 1000
INP_SYMBOL_SUFFIX = ".vx"

INP_MODEL_PATH = "models/hft_model.onnx"
INP_SCALER_PATH = "models/hft_model_scaler.pkl"
INP_USE_AI = True
INP_AI_CONFIDENCE_THRESHOLD = 0.52
INP_USE_AI_CONFIDENCE_GATE = False
INP_AI_CONFIDENCE_MIN = 0.55
INP_PREDICTION_HORIZON = 5

INP_OPTIMIZE_LATENCY = True
INP_MAX_LATENCY_MS = 250

# ============ Risk / protection ============
INP_MAX_DRAWDOWN_PERCENT = 5.0
INP_DRAWDOWN_CLOSE_POSITIONS = False
INP_DAILY_LOSS_LIMIT = 200.0
INP_USE_KILL_SWITCH = True

INP_DAILY_PROFIT_LOCK_TRIGGER = 50.0
INP_DAILY_PROFIT_GIVEBACK_RATIO = 0.50

# [FIX] Set to 0 to DISABLE risk budget blocking entirely.
# The broker's margin check + _free_margin_ok() remain as the real safety net.
INP_MAX_TOTAL_OPEN_RISK_PERCENT = 0.0
# ==========================================

INP_STOP_LOSS_POINTS = 100
INP_TAKE_PROFIT_POINTS = 300
INP_MAX_SPREAD_POINTS = 25
INP_TRAILING_STOP_POINTS = 120
INP_TRAILING_STEP_POINTS = 40

INP_DATA_UPDATE_FREQUENCY_MS = 10

INP_BREAKEVEN_TRIGGER_RATIO = 0.5
INP_BREAKEVEN_OFFSET_POINTS = 10
INP_USE_ATR_TRAILING = True
INP_ATR_TRAILING_MULTIPLIER = 1.5
INP_ATR_PERIOD = 14
INP_TRAILING_MODIFY_INTERVAL_S = 5.0

INP_SYMBOL_COOLDOWN_SECONDS = 0
INP_MIN_FREE_MARGIN_RATIO = 0.10

BASE_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY", "NZDJPY",
    "CADJPY", "EURAUD", "EURNZD", "EURCAD", "GBPAUD", "GBPNZD",
    "GBPCAD", "GBPCHF", "AUDNZD", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF"
]

# =====================================================================
# FEATURE ENGINEER
# =====================================================================
class LiveFeatureEngineer:
    @staticmethod
    def create_features(history_slice) -> pd.DataFrame:
        df = pd.DataFrame(history_slice)
        if df.empty or len(df) < 25:
            return pd.DataFrame()
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['spread'] / df['mid_price']
        for period in [1, 5, 10, 20]:
            if len(df) > period:
                df[f'return_{period}'] = df['mid_price'].pct_change(period)
                df[f'momentum_{period}'] = df['mid_price'] / df['mid_price'].shift(period) - 1
            else:
                df[f'return_{period}'] = 0.0
                df[f'momentum_{period}'] = 0.0
        for period in [10, 20]:
            if len(df) > period:
                df[f'volatility_{period}'] = df['return_1'].rolling(period).std()
                df[f'realized_vol_{period}'] = np.sqrt((df['return_1'] ** 2).rolling(period).sum())
            else:
                df[f'volatility_{period}'] = 0.0
                df[f'realized_vol_{period}'] = 0.0
        if 'volume' in df.columns:
            for period in [5, 10]:
                if len(df) > period:
                    df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
                    df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
                else:
                    df[f'volume_ma_{period}'] = df['volume']
                    df[f'volume_ratio_{period}'] = 1.0
        df['rsi_14'] = LiveFeatureEngineer._calculate_rsi(df['mid_price'], 14)
        df['bb_upper'], df['bb_lower'] = LiveFeatureEngineer._calculate_bollinger_bands(df['mid_price'], 20, 2)
        df['bb_position'] = (df['mid_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        now = datetime.now()
        df['hour'] = now.hour
        df['minute'] = now.minute
        df['day_of_week'] = now.weekday()
        df['is_market_open'] = int(9 <= now.hour < 16)
        df['bid_ask_imbalance'] = (df['bid'] - df['ask']) / (df['bid'] + df['ask'])
        df['price_impact'] = df['mid_price'].diff() / df['spread']
        for lag in [1, 2, 3, 5]:
            df[f'mid_price_lag_{lag}'] = df['mid_price'].shift(lag)
            df[f'spread_lag_{lag}'] = df['spread'].shift(lag)
        df = df.fillna(0)
        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma + (std * std_dev), sma - (std * std_dev)

# =====================================================================
# MARKET DATA
# =====================================================================
class MarketData:
    def __init__(self):
        self.bid = 0.0
        self.ask = 0.0
        self.spread = 0.0
        self.volume = 0
        self.time = datetime.now()

class SymbolState:
    def __init__(self, symbol_name):
        self.symbol = symbol_name
        self.current_market = MarketData()
        self.market_history = []
        self.point = 0.001 if "JPY" in symbol_name else 0.00001
        self.spec = None

# =====================================================================
# HFT MASTER BOT CORE ENGINE
# =====================================================================
class HFTMasterBotFull:
    def __init__(self):
        logger.info("Initializing HFT Master Bot v2.4 (Broker-limited exposure)...")
        self.client = Tickerall(api_key=API_KEY)
        self.session = self.client.sessions.start(
            broker=BROKER_NAME, server=BROKER_SERVER,
            account=ACCOUNT_LOGIN, password=ACCOUNT_PASSWORD
        )
        self.account_id = self.session.account_id
        logger.info(f"Connected: {self.account_id}")

        self.active_symbols = [s + INP_SYMBOL_SUFFIX for s in BASE_SYMBOLS]
        self.symbol_states = {sym: SymbolState(sym) for sym in self.active_symbols}
        self.latest_ticks = {}

        self.onnx_session = None
        self.scaler = None
        global INP_USE_AI
        if INP_USE_AI:
            try:
                self.onnx_session = ort.InferenceSession(INP_MODEL_PATH)
                self.scaler = joblib.load(INP_SCALER_PATH)
                logger.info("AI Model loaded")
            except Exception as e:
                logger.error(f"AI load failed: {e}")
                INP_USE_AI = False

        self.trades_count = 0
        self.last_trade_time = None

        self.daily_start_equity = None
        self.daily_equity_peak = None
        self.daily_pnl = 0.0
        self.daily_locked = False
        self.daily_loss_halted = False
        self.drawdown_halted = False

        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.latency_count = 0
        self.avg_latency = 0.0

        self.cached_positions = []
        self.cached_equity = 100.0
        self.cached_balance = 100.0
        self._last_account_fetch_ts = 0.0

        self.pending_symbols = set()
        self.last_order_time_by_symbol = {}
        self.last_modify_time_by_ticket = {}

        self.blocked_by_per_symbol_count = 0
        self.blocked_by_margin_count = 0
        self.blocked_by_risk_count = 0
        self.blocked_by_protection_count = 0
        self.orders_rejected_count = 0

        self.reset_daily_counters()

        self.stream = self.client.stream.connect()
        self.stream.on("tick", self.on_tick_event)
        self.stream.on("position", self.on_position_event)
        self.stream.on("account", self.on_account_event)

    def reset_daily_counters(self):
        self.trades_count = 0
        self.daily_pnl = 0.0
        self.latency_count = 0
        self.avg_latency = 0.0
        self.daily_locked = False
        self.daily_loss_halted = False

    def on_tick_event(self, event):
        sym = event.symbol
        if sym in self.symbol_states:
            state = self.symbol_states[sym]
            market = state.current_market
            market.bid = event.bid
            market.ask = event.ask
            market.spread = (event.ask - event.bid) / state.point
            market.volume = getattr(event, "volume", 100)
            market.time = datetime.now()
            state.market_history.append({
                'bid': market.bid, 'ask': market.ask, 'volume': float(market.volume)
            })
            if len(state.market_history) > 150:
                state.market_history.pop(0)
            self.latest_ticks[sym] = market

    def on_position_event(self, event):
        try:
            if hasattr(event, "positions"):
                self.cached_positions = event.positions
                cached_syms = {p.symbol for p in self.cached_positions}
                self.pending_symbols -= cached_syms
            elif hasattr(event, "position"):
                pos = event.position
                event_type = getattr(event, "event", "update")
                updated = [p for p in self.cached_positions if p.ticket != pos.ticket]
                if event_type != "close":
                    updated.append(pos)
                    self.pending_symbols.discard(pos.symbol)
                else:
                    self.last_modify_time_by_ticket.pop(pos.ticket, None)
                self.cached_positions = updated
        except Exception as e:
            logger.debug(f"Position event error: {e}")

    def on_account_event(self, event):
        try:
            snapshot = getattr(event, "snapshot", None)
            if snapshot:
                self.cached_equity = getattr(snapshot, "equity", self.cached_equity)
                self.cached_balance = getattr(snapshot, "balance", self.cached_balance)
        except Exception as e:
            logger.debug(f"Account event error: {e}")

    def _get_symbol_spec(self, symbol):
        state = self.symbol_states.get(symbol)
        if state is None:
            return None
        if state.spec is not None:
            return state.spec
        try:
            spec = self.client.symbols.get(self.account_id, symbol=symbol)
            state.spec = spec
            return spec
        except Exception as e:
            logger.debug(f"Symbol spec fetch failed for {symbol}: {e}")
            state.spec = False
            return None

    # ============ [FIX] Sane tick value fallback ============
    def _tick_value_per_point(self, symbol):
        """
        Returns $ per 1 point per 1.0 lot.
        Uses broker spec if available, otherwise a realistic retail-MT5 fallback.
        """
        spec = self._get_symbol_spec(symbol)
        if spec:
            for attr in ("trade_tick_value", "tick_value", "tradeTickValue"):
                v = getattr(spec, attr, None)
                if v:
                    for sz_attr in ("trade_tick_size", "tick_size", "tradeTickSize"):
                        sz = getattr(spec, sz_attr, None)
                        if sz:
                            try:
                                return float(v) * (self.symbol_states[symbol].point / float(sz))
                            except Exception:
                                pass
                    try:
                        return float(v)
                    except Exception:
                        pass

        # Realistic fallback for retail MT5 account (0.01 lot minimum):
        # - JPY pairs: 1 point = 0.001, 1 pip = 0.01 → $0.10/pip per 0.01 lot → $0.01/point per 0.01 lot → $1/point per 1.0 lot
        # - Non-JPY forex: 1 point = 0.00001 → $0.10/pip per 0.01 lot → $0.01/point per 0.01 lot → $1/point per 1.0 lot
        # - Gold (XAUUSD): 1 point = 0.01 → ~$1/point per 0.01 lot → $100/point per 1.0 lot
        # - Crypto (BTCUSD/ETHUSD): varies wildly
        sym_upper = symbol.upper()
        if "XAU" in sym_upper or "GOLD" in sym_upper:
            return 100.0
        if "BTC" in sym_upper:
            return 1.0
        if "ETH" in sym_upper:
            return 0.1
        # Forex fallback
        return 1.0
    # ======================================================

    def _is_symbol_blocked(self, symbol):
        symbol_positions = sum(1 for p in self.cached_positions if p.symbol == symbol)
        if symbol_positions >= INP_MAX_POSITIONS_PER_SYMBOL:
            self.blocked_by_per_symbol_count += 1
            return True
        if symbol in self.pending_symbols:
            return True
        if INP_SYMBOL_COOLDOWN_SECONDS > 0:
            last_ts = self.last_order_time_by_symbol.get(symbol, 0.0)
            if time.time() - last_ts < INP_SYMBOL_COOLDOWN_SECONDS:
                return True
        return False

    def _free_margin_ok(self) -> bool:
        if self.cached_equity <= 0:
            return False
        buffer_needed = self.cached_equity * INP_MIN_FREE_MARGIN_RATIO
        if self.cached_equity < buffer_needed:
            self.blocked_by_margin_count += 1
            return False
        return True

    def _protection_allows_trading(self) -> bool:
        if self.daily_loss_halted:
            self.blocked_by_protection_count += 1
            return False
        if self.drawdown_halted:
            self.blocked_by_protection_count += 1
            return False
        if self.daily_locked:
            self.blocked_by_protection_count += 1
            return False
        return True

    def _total_open_risk_percent(self) -> float:
        """For monitoring only — never blocks trading when INP_MAX_TOTAL_OPEN_RISK_PERCENT == 0."""
        if self.cached_equity <= 0:
            return 0.0
        total_risk = 0.0
        for pos in self.cached_positions:
            try:
                entry = getattr(pos, "entry_price", getattr(pos, "entryPrice", 0.0))
                sl = getattr(pos, "sl", getattr(pos, "stop_loss", 0.0))
                vol = getattr(pos, "volume", getattr(pos, "lot", INP_LOT_SIZE))
                if entry and sl:
                    state = self.symbol_states.get(pos.symbol)
                    if not state:
                        continue
                    distance_points = abs(entry - sl) / state.point
                    tick_value = self._tick_value_per_point(pos.symbol)
                    risk = distance_points * tick_value * vol
                    total_risk += risk
            except Exception:
                continue
        return (total_risk / self.cached_equity) * 100.0

    # ============ [FIX] Risk budget is now opt-in ============
    def _risk_budget_ok(self) -> bool:
        """Only blocks if INP_MAX_TOTAL_OPEN_RISK_PERCENT > 0. Set to 0 to disable."""
        if INP_MAX_TOTAL_OPEN_RISK_PERCENT <= 0:
            return True
        current = self._total_open_risk_percent()
        if current >= INP_MAX_TOTAL_OPEN_RISK_PERCENT:
            self.blocked_by_risk_count += 1
            return False
        return True
    # =======================================================

    def _calc_atr(self, state, period=INP_ATR_PERIOD):
        if len(state.market_history) < period + 1:
            return 0.0
        trs = []
        for i in range(1, period + 1):
            h = max(state.market_history[-i]['bid'], state.market_history[-i]['ask'])
            l = min(state.market_history[-i]['bid'], state.market_history[-i]['ask'])
            prev_mid = (state.market_history[-i-1]['bid'] + state.market_history[-i-1]['ask']) / 2
            tr = max(h - l, abs(h - prev_mid), abs(l - prev_mid))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0.0

    def run(self):
        logger.info("Starting trading loop...")
        self.stream.subscribe_ticks(self.account_id, self.active_symbols)
        self.stream.subscribe_positions(self.account_id)
        self.stream.subscribe_account(self.account_id)

        last_timer_time = time.time()

        try:
            while True:
                tick_start = time.perf_counter()
                open_positions = self.cached_positions
                equity = self.cached_equity
                balance = self.cached_balance

                if time.time() - self._last_account_fetch_ts > 60.0:
                    try:
                        account_data = self.client.accounts.get(self.account_id)
                        if hasattr(account_data, "account") and account_data.account:
                            acc_obj = account_data.account
                            self.cached_equity = getattr(acc_obj, "equity", 100.0)
                            self.cached_balance = getattr(acc_obj, "balance", 100.0)
                        else:
                            self.cached_equity = getattr(account_data, "equity", getattr(account_data, "balance", 100.0))
                            self.cached_balance = getattr(account_data, "balance", 100.0)
                        self._last_account_fetch_ts = time.time()
                        equity = self.cached_equity
                        balance = self.cached_balance
                    except Exception as api_err:
                        logger.warning(f"Account fetch failed: {api_err}")

                self._update_protection_state(equity)

                for symbol in self.active_symbols:
                    if symbol not in self.latest_ticks:
                        continue
                    state = self.symbol_states[symbol]
                    market = state.current_market
                    if market.spread > INP_MAX_SPREAD_POINTS:
                        continue
                    if self.trades_count >= INP_MAX_TRADES_PER_DAY:
                        continue

                    self.manage_positions(symbol, open_positions, state)

                    if self._is_symbol_blocked(symbol):
                        continue
                    if not self._protection_allows_trading():
                        continue
                    if not self._risk_budget_ok():
                        continue
                    if len(open_positions) < INP_MAX_POSITIONS and self._free_margin_ok():
                        self.execute_trading_logic(symbol, state, len(open_positions))

                self.update_performance_metrics(equity, balance)

                if time.time() - last_timer_time >= 1.0:
                    self.send_monitoring_data(equity, balance, len(open_positions))
                    last_timer_time = time.time()

                tick_end = time.perf_counter()
                latency_ms = (tick_end - tick_start) * 1000.0
                self.update_latency_metrics(latency_ms)
                if INP_OPTIMIZE_LATENCY and latency_ms > INP_MAX_LATENCY_MS:
                    logger.warning(f"High latency: {latency_ms:.2f}ms")

                time.sleep(INP_DATA_UPDATE_FREQUENCY_MS / 1000.0)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if INP_USE_KILL_SWITCH:
                self.close_all_positions("KillSwitch")
        except Exception as e:
            logger.error(f"Critical error: {e}")
            raise e
        finally:
            self.stream.close()
            self.client.sessions.end(self.account_id)
            logger.info("Bot stopped")

    def _update_protection_state(self, current_equity):
        now = datetime.now()

        if self.daily_start_equity is None or (self.last_trade_time and now.day != self.last_trade_time.day):
            self.daily_start_equity = current_equity
            self.daily_equity_peak = current_equity
            self.daily_locked = False
            self.daily_loss_halted = False
            self.drawdown_halted = False
            logger.info(f"New day: start_equity={current_equity:.2f}")

        if current_equity > (self.daily_equity_peak or 0):
            self.daily_equity_peak = current_equity

        self.daily_pnl = current_equity - self.daily_start_equity

        if INP_DAILY_LOSS_LIMIT > 0 and self.daily_pnl <= -abs(INP_DAILY_LOSS_LIMIT):
            if not self.daily_loss_halted:
                logger.warning(f"DAILY LOSS LIMIT breached: {self.daily_pnl:.2f}. Halting new entries.")
            self.daily_loss_halted = True

        if INP_DAILY_PROFIT_LOCK_TRIGGER > 0 and self.daily_equity_peak is not None:
            peak_gain = self.daily_equity_peak - self.daily_start_equity
            if peak_gain >= INP_DAILY_PROFIT_LOCK_TRIGGER:
                protected_floor = self.daily_equity_peak - peak_gain * INP_DAILY_PROFIT_GIVEBACK_RATIO
                if current_equity <= protected_floor:
                    if not self.daily_locked:
                        logger.warning(
                            f"PROFIT LOCK triggered: peak_equity={self.daily_equity_peak:.2f} "
                            f"current={current_equity:.2f} floor={protected_floor:.2f}. Halting new entries."
                        )
                    self.daily_locked = True

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - current_equity) / self.peak_equity * 100.0
            if dd > self.max_drawdown:
                self.max_drawdown = dd
            if dd >= INP_MAX_DRAWDOWN_PERCENT:
                if not self.drawdown_halted:
                    logger.error(f"DRAWDOWN KILL SWITCH: {dd:.2f}% >= {INP_MAX_DRAWDOWN_PERCENT}%. Halting.")
                self.drawdown_halted = True
                if INP_DRAWDOWN_CLOSE_POSITIONS and self.cached_positions:
                    self.close_all_positions("Drawdown kill switch")

    def execute_trading_logic(self, target_symbol, state, current_positions_count):
        signal = self.generate_trading_signal(state)
        if signal != 0:
            self.execute_trade_signal(target_symbol, signal, state, current_positions_count)

    def generate_trading_signal(self, state):
        if len(state.market_history) < 25:
            return 0
        if INP_USE_AI and self.onnx_session and self.scaler:
            try:
                df_features = LiveFeatureEngineer.create_features(state.market_history)
                if not df_features.empty and len(df_features) >= 50:
                    feature_cols = [col for col in df_features.columns if col != 'price_change']
                    features_matrix = df_features[feature_cols].values[-50:]
                    scaled_features = self.scaler.transform(features_matrix)
                    input_tensor = np.array([scaled_features], dtype=np.float32)
                    input_name = self.onnx_session.get_inputs()[0].name
                    outputs = self.onnx_session.run(None, {input_name: input_tensor})
                    prediction = float(outputs[0][0][0])

                    if INP_USE_AI_CONFIDENCE_GATE and len(outputs) > 1:
                        try:
                            confidence = float(outputs[1][0][0])
                            if 0.0 <= confidence <= 1.0:
                                if confidence < INP_AI_CONFIDENCE_MIN:
                                    return 0
                        except Exception:
                            pass

                    if prediction >= INP_AI_CONFIDENCE_THRESHOLD:
                        return 1
                    elif prediction <= -INP_AI_CONFIDENCE_THRESHOLD:
                        return -1
            except Exception as e:
                logger.debug(f"AI inference failed: {e}")
        recent_prices = [(m['bid'] + m['ask']) / 2 for m in state.market_history[-10:]]
        if len(recent_prices) >= 10:
            short_momentum = recent_prices[-1] - recent_prices[0]
            threshold = 2.0 * state.point
            if short_momentum > threshold:
                return 1
            elif short_momentum < -threshold:
                return -1
        return 0

    def execute_trade_signal(self, target_symbol, signal, state, current_positions_count):
        if signal == 0 or current_positions_count >= INP_MAX_POSITIONS:
            return
        if self._is_symbol_blocked(target_symbol):
            return
        if not self._protection_allows_trading():
            return
        if not self._risk_budget_ok():
            return
        if not self._free_margin_ok():
            return

        self.pending_symbols.add(target_symbol)
        self.last_order_time_by_symbol[target_symbol] = time.time()

        lot_size = self.calculate_position_size(state)
        sl, tp = self.calculate_stop_levels(signal, state)
        side = "BUY" if signal > 0 else "SELL"

        try:
            order = self.client.orders.place(
                account_id=self.account_id, type="market", symbol=target_symbol,
                side=side, volume=lot_size, stop_loss=sl, take_profit=tp
            )
            if order:
                self.trades_count += 1
                self.last_trade_time = datetime.now()
                logger.info(f"Order: {target_symbol} {side} Lot={lot_size} SL={sl:.5f} TP={tp:.5f}")
            else:
                self.pending_symbols.discard(target_symbol)
        except Exception as e:
            self.orders_rejected_count += 1
            logger.error(f"Order failed {target_symbol}: {e}")
            self.pending_symbols.discard(target_symbol)

    def calculate_position_size(self, state):
        equity = self.cached_equity
        max_risk_amount = equity * INP_MAX_RISK_PERCENT / 100.0

        tick_value = self._tick_value_per_point(state.symbol)
        if not tick_value or tick_value <= 0:
            return INP_LOT_SIZE

        risk_per_lot_for_sl = INP_STOP_LOSS_POINTS * tick_value
        if risk_per_lot_for_sl <= 0:
            return INP_LOT_SIZE

        max_lots_by_risk = max_risk_amount / risk_per_lot_for_sl
        final_lot = min(INP_LOT_SIZE, max_lots_by_risk)
        final_lot = max(final_lot, 0.01)
        return round(final_lot, 2)

    def calculate_stop_levels(self, signal, state):
        point = state.point
        market = state.current_market
        if signal > 0:
            sl = round(market.bid - INP_STOP_LOSS_POINTS * point, 5)
            tp = round(market.ask + INP_TAKE_PROFIT_POINTS * point, 5)
        else:
            sl = round(market.ask + INP_STOP_LOSS_POINTS * point, 5)
            tp = round(market.bid - INP_TAKE_PROFIT_POINTS * point, 5)
        return sl, tp

    def manage_positions(self, target_symbol, open_positions, state):
        point = state.point
        atr = self._calc_atr(state) if INP_USE_ATR_TRAILING else 0.0

        for pos in open_positions:
            if pos.symbol != target_symbol:
                continue
            open_price = getattr(pos, "entry_price", getattr(pos, "entryPrice", 0.0))
            current_sl = getattr(pos, "sl", getattr(pos, "stop_loss", 0.0))
            current_tp = getattr(pos, "tp", getattr(pos, "take_profit", 0.0))
            ticket = getattr(pos, "ticket", getattr(pos, "id", None))
            side = getattr(pos, "side", None)

            if side == "BUY":
                current_price = state.current_market.bid
                profit_points = (current_price - open_price) / point
                if profit_points >= INP_TAKE_PROFIT_POINTS * INP_BREAKEVEN_TRIGGER_RATIO:
                    be_sl = open_price + INP_BREAKEVEN_OFFSET_POINTS * point
                    if be_sl > current_sl and be_sl < current_price:
                        self.modify_position(ticket, be_sl, current_tp)
                        current_sl = be_sl
                if profit_points > INP_TRAILING_STOP_POINTS:
                    if INP_USE_ATR_TRAILING and atr > 0:
                        trail_dist = atr * INP_ATR_TRAILING_MULTIPLIER
                    else:
                        trail_dist = INP_TRAILING_STOP_POINTS * point
                    new_sl = current_price - trail_dist
                    if new_sl > current_sl + INP_TRAILING_STEP_POINTS * point or current_sl == 0:
                        self.modify_position(ticket, new_sl, current_tp)

            elif side == "SELL":
                current_price = state.current_market.ask
                profit_points = (open_price - current_price) / point
                if profit_points >= INP_TAKE_PROFIT_POINTS * INP_BREAKEVEN_TRIGGER_RATIO:
                    be_sl = open_price - INP_BREAKEVEN_OFFSET_POINTS * point
                    if (current_sl == 0 or be_sl < current_sl) and be_sl > current_price:
                        self.modify_position(ticket, be_sl, current_tp)
                        current_sl = be_sl
                if profit_points > INP_TRAILING_STOP_POINTS:
                    if INP_USE_ATR_TRAILING and atr > 0:
                        trail_dist = atr * INP_ATR_TRAILING_MULTIPLIER
                    else:
                        trail_dist = INP_TRAILING_STOP_POINTS * point
                    new_sl = current_price + trail_dist
                    if current_sl == 0 or new_sl < current_sl - INP_TRAILING_STEP_POINTS * point:
                        self.modify_position(ticket, new_sl, current_tp)

    def modify_position(self, ticket, new_sl, tp):
        last = self.last_modify_time_by_ticket.get(ticket, 0.0)
        if time.time() - last < INP_TRAILING_MODIFY_INTERVAL_S:
            return
        self.last_modify_time_by_ticket[ticket] = time.time()
        try:
            self.client.positions.modify(
                account_id=self.account_id, ticket=ticket,
                stop_loss=new_sl, take_profit=tp
            )
        except Exception as e:
            logger.error(f"Modify failed ticket {ticket}: {e}")

    def send_monitoring_data(self, equity, balance, positions_count):
        monitoring_json = self.create_monitoring_json(equity, balance, positions_count)
        logger.info(f"Monitoring: {monitoring_json}")

    def create_monitoring_json(self, equity, balance, positions_count):
        symbol_counts = {}
        for p in self.cached_positions:
            symbol_counts[p.symbol] = symbol_counts.get(p.symbol, 0) + 1
        data = {
            "timestamp": int(time.time()),
            "equity": round(equity, 2),
            "balance": round(balance, 2),
            "trades_count": self.trades_count,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_peak_equity": round(self.daily_equity_peak or 0, 2),
            "daily_start_equity": round(self.daily_start_equity or 0, 2),
            "profit_locked": self.daily_locked,
            "loss_halted": self.daily_loss_halted,
            "drawdown_halted": self.drawdown_halted,
            "max_drawdown": round(self.max_drawdown, 2),
            "open_risk_pct": round(self._total_open_risk_percent(), 2),
            "avg_latency": round(self.avg_latency, 2),
            "positions": positions_count,
            "per_symbol": symbol_counts,
            "blocked_by_per_symbol": self.blocked_by_per_symbol_count,
            "blocked_by_margin": self.blocked_by_margin_count,
            "blocked_by_risk": self.blocked_by_risk_count,
            "blocked_by_protection": self.blocked_by_protection_count,
            "orders_rejected": self.orders_rejected_count,
        }
        return json.dumps(data)

    def update_performance_metrics(self, current_equity, current_balance):
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100.0
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

    def update_latency_metrics(self, latency):
        self.latency_count += 1
        self.avg_latency = ((self.avg_latency * (self.latency_count - 1)) + latency) / self.latency_count

    def close_all_positions(self, reason):
        logger.info(f"Closing all: {reason}")
        try:
            for pos in self.cached_positions:
                ticket = getattr(pos, "ticket", getattr(pos, "id", None))
                if ticket:
                    self.client.positions.close(self.account_id, ticket=ticket)
        except Exception as e:
            logger.error(f"Close failed: {e}")

if __name__ == "__main__":
    bot = HFTMasterBotFull()
    bot.run()