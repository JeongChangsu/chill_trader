# strategy.py
import logging
import holidays

from config import KST, SESSION_GUIDES
from datetime import datetime


def determine_market_regime(multi_tf_data):
    """
    강화된 알고리즘을 사용하여 시장 상황(장세)을 결정합니다.
    """
    if not multi_tf_data:
        return "undefined"

    # 1. 변동성 판단 (Volatility)
    volatility = "normal"
    atr_1h = multi_tf_data.get("1h", {}).get("atr", 0)
    price_1h = multi_tf_data.get("1h", {}).get("current_price", 1)

    if atr_1h / price_1h > 0.025:
        volatility = "high"
    elif atr_1h / price_1h < 0.005:
        volatility = "low"

    # Donchian Channel Width (1시간봉 기준)
    donchian_width_1h = multi_tf_data.get("1h", {}).get('donchian_upper', 0) - multi_tf_data.get("1h", {}).get(
        'donchian_lower', 0)
    donchian_width_percent_1h = (donchian_width_1h / price_1h) * 100 if price_1h else 0

    # 2. 추세 판단 (Trend)
    trend = "sideways"  # 기본값: 횡보
    ema200_1d = multi_tf_data.get("1d", {}).get("ema200", None)
    price_1d = multi_tf_data.get("1d", {}).get("current_price", None)

    if price_1d is not None and ema200_1d is not None:
        if price_1d > ema200_1d:
            trend = "bull"  # 가격이 200일 EMA 위에 있으면 상승 추세
        elif price_1d < ema200_1d:
            trend = "bear"  # 가격이 200일 EMA 아래에 있으면 하락 추세

    # 3. 추세 강도 (Trend Strength) - ADX, DMI 사용 (1시간봉 기준)
    adx_1h = multi_tf_data.get("1h", {}).get("adx", 0)
    plus_di_1h = multi_tf_data.get("1h", {}).get("plus_di", 0)
    minus_di_1h = multi_tf_data.get("1h", {}).get("minus_di", 0)

    if "bull" in trend:
        if adx_1h > 25 and plus_di_1h > minus_di_1h:
            trend = "strong_bull"
        elif adx_1h < 20:
            trend = "weak_bull"
    elif "bear" in trend:
        if adx_1h > 25 and minus_di_1h > plus_di_1h:
            trend = "strong_bear"
        elif adx_1h < 20:
            trend = "weak_bear"

    # 4. 캔들 패턴 (Candle Patterns) - 1시간봉 기준
    candle_pattern = "neutral"
    if multi_tf_data.get("1h", {}).get("engulfing_bullish"):
        candle_pattern = "bullish"
    elif multi_tf_data.get("1h", {}).get("engulfing_bearish"):
        candle_pattern = "bearish"
    elif multi_tf_data.get("1h", {}).get("morning_star"):
        candle_pattern = "bullish"
    elif multi_tf_data.get("1h", {}).get("evening_star"):
        candle_pattern = "bearish"
    elif multi_tf_data.get("1h", {}).get("hammer"):
        candle_pattern = "bullish"
    elif multi_tf_data.get("1h", {}).get("hanging_man"):
        candle_pattern = "bearish"

    # 5. 거래량 분석 (Volume Analysis)
    volume_analysis = "neutral"
    volume_change_1h = multi_tf_data.get("1h", {}).get("volume_change", 0)
    bearish_div_1h = multi_tf_data.get("1h", {}).get("bearish_divergence", 0)
    bullish_div_1h = multi_tf_data.get("1h", {}).get("bullish_divergence", 0)

    if "bull" in trend and volume_change_1h > 50:
        volume_analysis = "confirming"
    elif "bear" in trend and volume_change_1h > 50:
        volume_analysis = "confirming"
    elif bullish_div_1h > 5:
        volume_analysis = "bullish_divergence"
    elif bearish_div_1h > 5:
        volume_analysis = "bearish_divergence"

    # 종합적인 시장 상황 판단
    if trend == "strong_bull" and volume_analysis == "confirming":
        market_regime = "strong_bull_trend"
    elif trend == "weak_bull":
        market_regime = "weak_bull_trend"
    elif trend == "strong_bear" and volume_analysis == "confirming":
        market_regime = "strong_bear_trend"
    elif trend == "weak_bear":
        market_regime = "weak_bear_trend"
    elif trend == "sideways":
        if donchian_width_percent_1h < 3:
            market_regime = "tight_sideways"
        elif donchian_width_percent_1h > 7:
            market_regime = "wide_sideways"
        else:
            market_regime = "normal_sideways"
    else:
        market_regime = "undefined_trend"

    market_regime = f"{volatility}_volatility_{market_regime}"

    if candle_pattern != "neutral":
        market_regime += f"_{candle_pattern}_candle"
    if volume_analysis != "neutral":
        market_regime += f"_{volume_analysis}"

    logging.info(f"Market regime determined: {market_regime.upper()}")
    return market_regime


def adjust_indicator_thresholds(market_regime):
    """시장 상황에 따라 RSI, MACD, Donchian Channel 등의 임계값을 동적으로 조정"""
    thresholds = {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "donchian_window": 20,
        "indicator_weights": {
            "rsi": 0.2,
            "macd": 0.2,
            "ema": 0.3,
            "donchian": 0.2,
            "volume": 0.1,
        }
    }

    if "strong_bull_trend" in market_regime:
        thresholds["rsi_oversold"] = 40
        thresholds["rsi_overbought"] = 80
        thresholds["donchian_window"] = 25
        thresholds["indicator_weights"]["ema"] = 0.4
        thresholds["indicator_weights"]["volume"] = 0.2
        thresholds["indicator_weights"]["rsi"] = 0.15
        thresholds["indicator_weights"]["macd"] = 0.15
        thresholds["indicator_weights"]["donchian"] = 0.1

    elif "weak_bull_trend" in market_regime:
        thresholds["rsi_oversold"] = 35
        thresholds["rsi_overbought"] = 75
        thresholds["indicator_weights"]["ema"] = 0.35
        thresholds["indicator_weights"]["volume"] = 0.15
        thresholds["indicator_weights"]["rsi"] = 0.25

    elif "strong_bear_trend" in market_regime:
        thresholds["rsi_oversold"] = 20
        thresholds["rsi_overbought"] = 60
        thresholds["donchian_window"] = 25
        thresholds["indicator_weights"]["ema"] = 0.4
        thresholds["indicator_weights"]["volume"] = 0.2
        thresholds["indicator_weights"]["rsi"] = 0.15
        thresholds["indicator_weights"]["macd"] = 0.15
        thresholds["indicator_weights"]["donchian"] = 0.1

    elif "weak_bear_trend" in market_regime:
        thresholds["rsi_oversold"] = 25
        thresholds["rsi_overbought"] = 65
        thresholds["indicator_weights"]["ema"] = 0.35
        thresholds["indicator_weights"]["volume"] = 0.15
        thresholds["indicator_weights"]["rsi"] = 0.25

    elif "tight_sideways" in market_regime:
        thresholds["rsi_oversold"] = 25
        thresholds["rsi_overbought"] = 75
        thresholds["donchian_window"] = 15
        thresholds["indicator_weights"]["donchian"] = 0.4
        thresholds["indicator_weights"]["rsi"] = 0.3
        thresholds["indicator_weights"]["macd"] = 0.2
        thresholds["indicator_weights"]["ema"] = 0.05
        thresholds["indicator_weights"]["volume"] = 0.05

    elif "wide_sideways" in market_regime:
        thresholds["rsi_oversold"] = 35
        thresholds["rsi_overbought"] = 65
        thresholds["donchian_window"] = 25
        thresholds["indicator_weights"]["donchian"] = 0.4
        thresholds["indicator_weights"]["rsi"] = 0.3
        thresholds["indicator_weights"]["macd"] = 0.2
        thresholds["indicator_weights"]["ema"] = 0.05
        thresholds["indicator_weights"]["volume"] = 0.05

    elif "normal_sideways" in market_regime:
        thresholds["indicator_weights"]["donchian"] = 0.35
        thresholds["indicator_weights"]["rsi"] = 0.3
        thresholds["indicator_weights"]["macd"] = 0.25
        thresholds["indicator_weights"]["ema"] = 0.05
        thresholds["indicator_weights"]["volume"] = 0.05

    if "high_volatility" in market_regime:
        thresholds["indicator_weights"]["volume"] += 0.1
        thresholds["indicator_weights"]["atr"] = 0.2

    elif "low_volatility" in market_regime:
        thresholds["indicator_weights"]["ema"] += 0.1
        thresholds["indicator_weights"]["atr"] = 0.05

    return thresholds


def select_strategy(market_regime):
    """
    결정된 시장 상황(장세)에 따라 가장 적합한 전략을 선택합니다.
    """
    strategy_templates = {  # 이 부분을 strategy.py 안으로 옮김
        "strong_bull_trend_follow": {
            "name": "Strong Bull Trend Following (Momentum)",
            "description": "Follow the trend in a strong uptrend.",
            "primary_timeframe": "1d",
            "indicators": {
                "ema": {"weight": 0.4, "params": [20, 50, 200]},
                "rsi": {"weight": 0.2, "params": [14, 40, 80]},
                "macd": {"weight": 0.1, "params": []},
                "volume": {"weight": 0.3, "params": []},
            },
            "entry_rules": {
                "long": [
                    "price > ema20",
                    "price > ema50",
                    "price > ema200",
                    "rsi > 50",
                    "volume_change > 20",
                ],
            },
            "exit_rules": {
                "tp": "atr_multiplier * 3",
                "sl": "atr_multiplier * 2",
            },
            "trade_term": "1d ~ 3d",
            "leverage": "3x ~ 5x"
        },
        "weak_bull_trend_pullback": {
            "name": "Weak Bull Trend Pullback (Dip Buying)",
            "description": "Buy the dip in a weak uptrend.",
            "primary_timeframe": "4h",
            "indicators": {
                "ema": {"weight": 0.35, "params": [20, 50]},
                "rsi": {"weight": 0.35, "params": [14, 35, 75]},
                "macd": {"weight": 0.1, "params": []},
                "volume": {"weight": 0.2, "params": []},
            },
            "entry_rules": {
                "long": [
                    "price > ema50",
                    "rsi < 35",
                    "bullish_divergence",
                ],
            },
            "exit_rules": {
                "tp": "atr_multiplier * 2.5",
                "sl": "atr_multiplier * 1.5",
            },
            "trade_term": "6h ~ 1d",
            "leverage": "3x ~ 4x"
        }, "strong_bear_trend_follow": {
            "name": "Strong Bear Trend Following (Momentum)",
            "description": "Follow the trend in a strong downtrend.",
            "primary_timeframe": "1d",
            "indicators": {
                "ema": {"weight": 0.4, "params": [20, 50, 200]},
                "rsi": {"weight": 0.2, "params": [14, 20, 60]},
                "macd": {"weight": 0.1, "params": []},
                "volume": {"weight": 0.3, "params": []},
            },
            "entry_rules": {
                "short": [
                    "price < ema20",
                    "price < ema50",
                    "price < ema200",
                    "rsi < 50",
                    "volume_change > 20",
                ],
            },
            "exit_rules": {
                "tp": "atr_multiplier * 3",
                "sl": "atr_multiplier * 2",
            },
            "trade_term": "1d ~ 3d",
            "leverage": "3x ~ 5x"
        },
        "weak_bear_trend_bounce": {
            "name": "Weak Bear Trend Bounce (Short Selling)",
            "description": "Sell the rally in a weak downtrend.",
            "primary_timeframe": "4h",
            "indicators": {
                "ema": {"weight": 0.35, "params": [20, 50]},
                "rsi": {"weight": 0.35, "params": [14, 25, 65]},
                "macd": {"weight": 0.1, "params": []},
                "volume": {"weight": 0.2, "params": []},
            },
            "entry_rules": {
                "short": [
                    "price < ema50",
                    "rsi > 65",
                    "bearish_divergence",
                ],
            },
            "exit_rules": {
                "tp": "atr_multiplier * 2.5",
                "sl": "atr_multiplier * 1.5",
            },
            "trade_term": "6h ~ 1d",
            "leverage": "2x ~ 4x"
        },
        "tight_sideways_range": {
            "name": "Tight Sideways Range (Scalping)",
            "description": "Trade within a narrow range using Donchian Channel.",
            "primary_timeframe": "5m",
            "indicators": {
                "donchian": {"weight": 0.4, "params": [15]},
                "rsi": {"weight": 0.3, "params": [14, 25, 75]},
                "macd": {"weight": 0.2, "params": []},
                "volume": {"weight": 0.1, "params": []},
                "ema": {"weight": 0.0, "params": []}
            },
            "entry_rules": {
                "long": [
                    "price <= donchian_lower",
                    "rsi < 25",
                ],
                "short": [
                    "price >= donchian_upper",
                    "rsi > 75",
                ],
            },
            "exit_rules": {
                "tp": "donchian_middle",
                "sl": "donchian_lower - atr * 1",
            },
            "trade_term": "5m ~ 15m",
            "leverage": "5x ~ 10x"
        },
        "wide_sideways_range": {
            "name": "Wide Sideways Range (Range Trading)",
            "description": "Trade within a wide range using Donchian Channel.",
            "primary_timeframe": "1h",
            "indicators": {
                "donchian": {"weight": 0.4, "params": [25]},
                "rsi": {"weight": 0.3, "params": [14, 35, 65]},
                "macd": {"weight": 0.2, "params": []},
                "volume": {"weight": 0.1, "params": []},
                "ema": {"weight": 0.0, "params": []}
            },
            "entry_rules": {
                "long": [
                    "price <= donchian_lower",
                    "rsi < 35",
                ],
                "short": [
                    "price >= donchian_upper",
                    "rsi > 65",
                ],
            },
            "exit_rules": {
                "tp": "donchian_middle",
                "sl": "donchian_lower - atr * 1.5",
            },
            "trade_term": "1h ~ 4h",
            "leverage": "3x ~ 5x"
        },
        "normal_sideways_range": {
            "name": "Normal Sideways Range (Range Trading)",
            "description": "Trade within a normal range using Donchian Channel.",
            "primary_timeframe": "15m",
            "indicators": {
                "donchian": {"weight": 0.35, "params": [20]},
                "rsi": {"weight": 0.3, "params": [14, 30, 70]},
                "macd": {"weight": 0.25, "params": []},
                "volume": {"weight": 0.05, "params": []},
                "ema": {"weight": 0.05, "params": []}
            },
            "entry_rules": {
                "long": [
                    "price <= donchian_lower",
                    "rsi < 30",
                ],
                "short": [
                    "price >= donchian_upper",
                    "rsi > 70",
                ],
            },
            "exit_rules": {
                "tp": "donchian_middle",
                "sl": "donchian_lower - atr * 1.2",
            },
            "trade_term": "15m ~ 1h",
            "leverage": "3x ~ 5x"
        }
    }

    if "strong_bull_trend" in market_regime:
        return strategy_templates["strong_bull_trend_follow"]
    elif "weak_bull_trend" in market_regime:
        return strategy_templates["weak_bull_trend_pullback"]
    elif "strong_bear_trend" in market_regime:
        return strategy_templates["strong_bear_trend_follow"]
    elif "weak_bear_trend" in market_regime:
        return strategy_templates["weak_bear_trend_bounce"]
    elif "tight_sideways" in market_regime:
        return strategy_templates["tight_sideways_range"]
    elif "wide_sideways" in market_regime:
        return strategy_templates["wide_sideways_range"]
    elif "normal_sideways" in market_regime:
        return strategy_templates["normal_sideways_range"]
    else:
        return None


def get_current_session_kst():
    """
    KST 기준 현재 시간을 확인하여 트레이딩 세션, 주말, 미국 공휴일 여부 결정
    """
    now = datetime.now(KST)
    hour = now.hour

    # 주말 여부 확인
    is_weekend = now.weekday() >= 5

    # 미국 공휴일 여부 확인
    us_holidays = holidays.US()
    is_us_holiday = now.date() in us_holidays

    if 0 <= hour < 8:
        session = "OVERNIGHT"
    elif 8 <= hour < 16:
        session = "ASIAN"
    elif 16 <= hour < 22:
        session = "LONDON"
    elif 22 <= hour < 24 or 0 <= hour < 6:
        session = "US"
    elif 6 <= hour < 8:
        session = "TRANSITION"
    else:
        session = "UNDEFINED"

    if is_weekend:
        session += "_WEEKEND"
    if is_us_holiday:
        session += "_US_HOLIDAY"

    return session
