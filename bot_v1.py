import os
import re
import ta
import csv
import time
import ccxt
import requests

import numpy as np
import pandas as pd
import undetected_chromedriver as uc  # 무료 데이터 크롤링을 위한 stealth driver

from openai import OpenAI
from datetime import datetime
from selenium.webdriver.common.by import By

# Initialize GPT client (OPENAI_API_KEY must be set in your environment)
openai_api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=openai_api_key)

# Global settings
SYMBOL = "BTC/USDT"
DECISIONS_LOG_FILE = "trading_decisions.csv"
OPEN_POSITIONS_FILE = "open_positions.csv"
CLOSED_POSITIONS_FILE = "closed_positions.csv"
TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

##############################################
# 0. Persistent Driver Setup
##############################################
# 전역 딕셔너리를 사용해 session_id별로 driver를 저장합니다.
drivers = {}


def create_driver():
    """새로운 undetected_chromedriver 인스턴스를 생성합니다."""
    options = uc.ChromeOptions()
    # 필요한 옵션을 추가할 수 있습니다. 예: headless 모드 설정 등.
    driver = uc.Chrome(options=options)
    return driver


def get_driver(session_id='default_session'):
    """
    session_id를 기준으로 이미 생성된 드라이버가 있으면 재사용하고,
    없으면 새 드라이버를 생성하여 반환합니다.
    이 함수는 driver.quit()을 호출하지 않으므로, 브라우저는 계속 실행 상태를 유지합니다.
    """
    global drivers
    if session_id in drivers and drivers[session_id] is not None:
        return drivers[session_id]
    else:
        driver = create_driver()
        drivers[session_id] = driver
        return driver


##############################################
# 1. Data Collection & Technical Indicator Calculation
##############################################

def fetch_ohlcv(symbol, timeframe, limit=300):
    """
    Fetch OHLCV data from Binance using ccxt and return as a pandas DataFrame.
    """
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching OHLCV for {symbol} on {timeframe}: {e}")
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def compute_technical_indicators(df):
    """
    Compute various technical indicators and append as new columns.
    Indicators include: RSI, SMA20, SMA50, SMA200, MACD, ATR, OBV, MFI, Bollinger Bands,
    and moving average percentage differences.
    """
    # Momentum and trend indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['sma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Volatility & volume-based indicators
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()

    # Bollinger Bands (20-period, 2 standard deviations)
    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()

    # Moving Average differences (in percentage)
    df['ma50_diff'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['ma200_diff'] = (df['close'] - df['sma200']) / df['sma200'] * 100

    return df


def fetch_order_book(symbol):
    """
    Retrieve order book data (best bid, ask, and spread) from Binance.
    """
    exchange = ccxt.binance()
    try:
        order_book = exchange.fetch_order_book(symbol)
        bid = order_book['bids'][0][0] if order_book['bids'] else None
        ask = order_book['asks'][0][0] if order_book['asks'] else None
        spread = round(ask - bid, 2) if bid and ask else None
    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")
        bid = ask = spread = None
    return {"bid": bid, "ask": ask, "spread": spread}


##############################################
# 1-2. Extended Data Collection via undetected_chromedriver (무료 크롤링)
##############################################

def fetch_exchange_inflows():
    """
    무료 소스(예: CryptoQuant)에서 exchange inflow 및 outflow 데이터를 크롤링하여
    net inflow (유입량 - 유출량)을 계산하여 반환.
    (실제 URL 및 셀렉터는 대상 사이트에 맞게 수정 필요)
    """
    url = f"https://cryptoquant.com/asset/btc/chart/exchange-flows"  # 예시 URL
    try:
        driver = get_driver('chill_trader')
        driver.get(url)
        time.sleep(3)
        netflow_all_text = driver.find_element(By.XPATH, '//tbody[@class="ant-table-tbody"]/tr[6]').text
        driver.quit()

        # 문자열을 숫자로 변환 (K, M 단위 처리)
        def parse_value(val_str):
            val_str = val_str.replace(",", "").strip()
            if "K" in val_str:
                return float(val_str.replace("K", "")) * 1000
            elif "M" in val_str:
                return float(val_str.replace("M", "")) * 1e6
            else:
                try:
                    return float(val_str)
                except:
                    return 0.0

        if '거래소 순입출금량' in netflow_all_text:
            netflow_text = netflow_all_text.split('\n')[-2]
            net_inflow = parse_value(netflow_text)
            return net_inflow
    except Exception as e:
        print(f"Error crawling net exchange inflows from CryptoQuant: {e}")
        return "N/A"


def fetch_funding_rate(symbol):
    """
    기존 Binance Futures 데이터를 사용.
    """
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        funding_info = futures_exchange.fetch_funding_rate(symbol=symbol)
        latest_funding = funding_info['info']['lastFundingRate'] if 'info' in funding_info else None
        return latest_funding
    except Exception as e:
        print(f"Error fetching funding rate for {symbol}: {e}")
        return "N/A"


def fetch_open_interest(symbol):
    """
    기존 Binance Futures 데이터를 사용.
    """
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        symbol_futures = symbol.replace("/", "")
        oi_response = futures_exchange.fetch_open_interest(symbol=symbol_futures)
        open_interest = oi_response['openInterest'] if oi_response and 'openInterest' in oi_response else None
        return open_interest
    except Exception as e:
        print(f"Error fetching open interest for {symbol}: {e}")
        return "N/A"


def fetch_fear_and_greed_index():
    """
    Retrieve the current Fear & Greed Index from Alternative.me.
    """
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        data = response.json()
        value = data['data'][0]['value'] if 'data' in data and len(data['data']) > 0 else None
        classification = data['data'][0]['value_classification'] if 'data' in data and len(data['data']) > 0 else None
        return classification, value
    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return None, None


def fetch_multi_tf_data(symbol, timeframes=None, limit=300):
    """
    Fetch OHLCV data and compute technical indicators for multiple timeframes.
    Returns a dictionary keyed by timeframe.
    """
    if timeframes is None:
        timeframes = ["5m", "15m", "1h", "4h", "1d"]
    multi_tf_data = {}
    for tf in timeframes:
        df = fetch_ohlcv(symbol, tf, limit)
        if df is None:
            continue
        df = compute_technical_indicators(df)
        latest = df.iloc[-1]
        multi_tf_data[tf] = {
            "current_price": round(latest['close'], 2),
            "rsi": round(latest['rsi'], 2) if not np.isnan(latest['rsi']) else None,
            "sma20": round(latest['sma20'], 2) if not np.isnan(latest['sma20']) else None,
            "sma50": round(latest['sma50'], 2) if not np.isnan(latest['sma50']) else None,
            "sma200": round(latest['sma200'], 2) if not np.isnan(latest['sma200']) else None,
            "ma50_diff": round(latest['ma50_diff'], 2) if not np.isnan(latest['ma50_diff']) else None,
            "ma200_diff": round(latest['ma200_diff'], 2) if not np.isnan(latest['ma200_diff']) else None,
            "bb_upper": round(latest['bb_upper'], 2) if not np.isnan(latest['bb_upper']) else None,
            "macd": round(latest['macd'], 2) if not np.isnan(latest['macd']) else None,
            "macd_signal": round(latest['macd_signal'], 2) if not np.isnan(latest['macd_signal']) else None,
            "atr": round(latest['atr'], 2) if not np.isnan(latest['atr']) else None,
            "obv": round(latest['obv'], 2) if not np.isnan(latest['obv']) else None,
            "mfi": round(latest['mfi'], 2) if not np.isnan(latest['mfi']) else None,
            "timestamp": latest['timestamp']
        }
    return multi_tf_data


##############################################
# 2. Market Regime & Dynamic Indicator Thresholds
##############################################

def determine_market_regime(multi_tf_data):
    """
    단순 기준: 1시간봉 데이터를 기준으로 가격이 SMA200, SMA50 보다 높으면 bull,
    둘 다 낮으면 bear, 아니면 sideways.
    """
    data = multi_tf_data.get("1h")
    if data is None:
        return "sideways"
    current_price = data["current_price"]
    sma200 = data["sma200"]
    sma50 = data["sma50"]
    if sma200 is None or sma50 is None:
        return "sideways"
    if current_price > sma200 and current_price > sma50:
        return "bull"
    elif current_price < sma200 and current_price < sma50:
        return "bear"
    else:
        return "sideways"


def adjust_indicator_thresholds(market_regime):
    """
    시장 장세에 따라 RSI, MACD, MA 관련 기준을 동적으로 설정.
    """
    if market_regime == "bull":
        thresholds = {
            "rsi_oversold": 45,  # bull 시장에서는 RSI 30 대신 45 정도에서 반등
            "rsi_overbought": 80,  # 강세장에서 RSI가 70에 머무르므로 80으로 상향 조정
            "macd_comment": "Ignore minor bearish MACD crosses; wait for a deep pullback for entries.",
            "ma_comment": "Buy pullbacks when price touches the 50MA (acting as dynamic support)."
        }
    elif market_regime == "bear":
        thresholds = {
            "rsi_oversold": 20,  # 약세장에서는 RSI가 더 낮은 값까지 내려감
            "rsi_overbought": 55,  # 약세장에서 RSI 55 이상이면 단기 반등 가능
            "macd_comment": "Ignore minor bullish MACD crosses; wait for confirmation of momentum continuation.",
            "ma_comment": "Sell rallies when price touches the 50MA (acting as resistance)."
        }
    else:
        thresholds = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_comment": "Use MACD crosses as confirmation only.",
            "ma_comment": "Apply mean reversion strategies around the 50MA."
        }
    return thresholds


##############################################
# 3. GPT Prompt Generation & Trading Decision (동적 기준 포함)
##############################################

def generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data, multi_tf_data, market_regime,
                        thresholds):
    """
    Generate a detailed XML prompt in English for GPT.
    XML 내에 동적 기준(시장 장세별 RSI, MACD, MA 기준)과 청산맵 정보를 포함.
    """
    # Create a multi-timeframe summary string
    multi_tf_summary = ""
    for tf, data in multi_tf_data.items():
        multi_tf_summary += (
            f"{tf} - Price: {data['current_price']}, RSI: {data['rsi']}, "
            f"SMA20: {data['sma20']}, SMA50: {data['sma50']} (Diff: {data['ma50_diff']}%), "
            f"SMA200: {data['sma200']} (Diff: {data['ma200_diff']}%), "
            f"Bollinger Upper: {data['bb_upper']}, "
            f"MACD: {data['macd']} (Signal: {data['macd_signal']}), "
            f"ATR: {data['atr']}, OBV: {data['obv']}, MFI: {data['mfi']}\n"
        )

    prompt = f"""
<TradeBotPrompt>
    <Account>
        <WalletBalance>{wallet_balance}</WalletBalance>
        <CurrentPosition>{position_info}</CurrentPosition>
    </Account>
    <MarketContext>
        <MarketRegime>{market_regime.upper()}</MarketRegime>
        <MultiTimeframeSummary>
{multi_tf_summary}
        </MultiTimeframeSummary>
        <AdditionalData>
            <FundingRate>{extended_data.get('funding_rate', 'N/A')}</FundingRate>
            <OpenInterest>{extended_data.get('open_interest', 'N/A')}</OpenInterest>
            <OrderBook>
                <Bid>{extended_data.get('order_book_bid', 'N/A')}</Bid>
                <Ask>{extended_data.get('order_book_ask', 'N/A')}</Ask>
                <Spread>{extended_data.get('order_book_spread', 'N/A')}</Spread>
            </OrderBook>
            <ExchangeNetInflow>{extended_data.get('exchange_inflows', 'N/A')}</ExchangeNetInflow>
            <FearAndGreedIndex>{extended_data.get('fear_and_greed_index', 'N/A')}</FearAndGreedIndex>
        </AdditionalData>
    </MarketContext>
    <Indicators>
        <RSI>
            <Guide>
                In bullish conditions, consider RSI oversold around {thresholds['rsi_oversold']} rather than the classic 30.
                Evaluate RSI divergence and failure swing patterns. In bearish markets, adjust thresholds to 20 (oversold) and 55 (overbought),
                watching for persistent low RSI with bearish divergence.
            </Guide>
        </RSI>
        <MovingAverages>
            <SMA20>
                <Guide>
                    If the current price is above the SMA20, add bullish weight; if below, add bearish weight.
                    SMA20 can be used as a short-term dynamic support/resistance level.
                </Guide>
            </SMA20>
            <SMA50>
                <Guide>{thresholds['ma_comment']}</Guide>
            </SMA50>
            <SMA200>
                <Guide>
                    A significant premium (e.g. 23% or more) above the SMA200 may indicate an overextended market,
                    suggesting caution. Conversely, prices below SMA200 typically indicate a bearish bias.
                </Guide>
            </SMA200>
        </MovingAverages>
        <BollingerBands>
            <Guide>
                Use Bollinger Bands to assess volatility. In trending markets, "band walking" near the upper band suggests continuation,
                while a return to the middle band may signal a pullback. In range-bound conditions, consider mean reversion strategies,
                especially during band squeezes.
            </Guide>
        </BollingerBands>
        <MACD>
            <Guide>
                {thresholds['macd_comment']}
                Analyze MACD histogram shapes, divergences, and zero-line crossovers. In bullish conditions, only consider bullish signals;
                in bearish conditions, confirm signals with volume and trend divergence.
            </Guide>
        </MACD>
        <ATR>
            <Guide>
                If ATR/Price < 0.005, the market is quiet (allowing higher leverage up to 10x for longs).
                If ATR/Price > 0.02, the market is volatile (use lower leverage, minimum 1x).
            </Guide>
        </ATR>
        <OBV>
            <Guide>
                Rising OBV over the last 5 candles adds bullish weight; falling OBV indicates bearish pressure.
            </Guide>
        </OBV>
        <MFI>
            <Guide>
                MFI below 20 indicates oversold conditions and potential bullish reversals, while above 80 signals overbought conditions.
                However, always cross-check with other indicators.
            </Guide>
        </MFI>
        <FundingRate>
            <Guide>
                High positive funding rates imply an overheated long market, suggesting short position opportunities.
            </Guide>
        </FundingRate>
        <OpenInterest>
            <Guide>
                Record high or rapidly increasing open interest indicates high market leverage. Exercise caution as it may signal trend continuation.
            </Guide>
        </OpenInterest>
        <ExchangeNetInflow>
            <Guide>
                A positive net inflow suggests accumulation on exchanges; a negative net inflow implies selling pressure.
            </Guide>
        </ExchangeNetInflow>
        <FearAndGreedIndex>
            <Guide>
                Use the provided Fear & Greed Index value directly to gauge market sentiment. Cross-check with other indicators instead of relying solely on this value.
            </Guide>
        </FearAndGreedIndex>
    </Indicators>
    <DecisionRules>
        <MultiTimeframe>
            <Guide>
                Analyze 5m, 15m, 1h, 4h, and 1d timeframes. A trade is recommended only if at least 4 out of these 5 timeframes indicate the same directional bias (bullish or bearish).
            </Guide>
            <TradeTerm>
                <Guide>
                    If lower timeframes (5m, 15m) dominate, set trade term as intraday/6h; if mid timeframes (1h, 4h) dominate, set term as 1d;
                    if the higher timeframe (1d) is dominant, set term as 1w.
                </Guide>
            </TradeTerm>
        </MultiTimeframe>
        <RiskReward>
            <Guide>
                Only open a new position if the potential reward is at least twice the risk (2:1 ratio).
            </Guide>
        </RiskReward>
    </DecisionRules>
    <Task>
        Analyze the provided indicators, market context, and dynamic thresholds according to the guidelines above.
        Decide whether to GO LONG, GO SHORT, HOLD LONG, HOLD SHORT, or NO TRADE.
        Then, recommend a leverage multiplier (e.g. 3x, 5x, etc.), a trade term, a take profit price (in absolute value),
        a stop loss price (in absolute value), and a specific limit order price.
        Output your decision as a single comma-separated line with the following fields:
        Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Brief Rationale.
    </Task>
    <OutputExample>
        GO LONG, 5x, 6h, 11400, 10800, 11300, Majority of timeframes show bullish momentum with low volatility and adjusted RSI indicating a dip entry.
    </OutputExample>
</TradeBotPrompt>
    """
    return prompt


def generate_trading_decision(wallet_balance, position_info, aggregated_data, extended_data, multi_tf_data,
                              market_regime, thresholds):
    prompt = generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data, multi_tf_data,
                                 market_regime, thresholds)
    developer_prompt = (
        "You are an automated BTC/USDT trading assistant that follows detailed quantitative indicators and strict guidelines "
        "as defined in the provided XML. Do not deviate from the specified output format."
    )
    response = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def parse_trading_decision(response_text):
    """
    Parse GPT response.
    Expected output format (comma-separated):
    Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Brief Rationale
    """
    response_text = response_text.strip()
    parts = [part.strip() for part in re.split(r'\s*,\s*', response_text)]
    if len(parts) < 7:
        raise ValueError("Incomplete response. Expected at least 7 comma-separated fields.")
    decision = {
        "final_action": parts[0],
        "leverage": parts[1],
        "trade_term": parts[2],
        "tp_price": parts[3],
        "sl_price": parts[4],
        "limit_order_price": parts[5],
        "rationale": ", ".join(parts[6:])
    }
    return decision


##############################################
# 4. Position Logging Functions
##############################################

def log_decision(decision, symbol):
    file_exists = os.path.isfile(DECISIONS_LOG_FILE)
    with open(DECISIONS_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "final_action", "leverage", "trade_term", "tp_price", "sl_price",
                             "limit_order_price", "rationale"])
        writer.writerow([datetime.utcnow(), symbol, decision["final_action"], decision["leverage"],
                         decision["trade_term"], decision["tp_price"], decision["sl_price"],
                         decision["limit_order_price"], decision["rationale"]])


def log_open_position(symbol, decision, entry_price):
    file_exists = os.path.isfile(OPEN_POSITIONS_FILE)
    with open(OPEN_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["timestamp", "symbol", "action", "entry_price", "leverage", "trade_term", "tp_price", "sl_price",
                 "limit_order_price", "rationale"])
        writer.writerow([datetime.utcnow(), symbol, decision["final_action"], entry_price, decision["leverage"],
                         decision["trade_term"], decision["tp_price"], decision["sl_price"],
                         decision["limit_order_price"], decision["rationale"]])


def log_closed_position(symbol, entry_price, exit_price, trade_side):
    profit = (exit_price - entry_price) if trade_side.upper() == "LONG" else (entry_price - exit_price)
    file_exists = os.path.isfile(CLOSED_POSITIONS_FILE)
    with open(CLOSED_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "trade_side", "entry_price", "exit_price", "profit"])
        writer.writerow([datetime.utcnow(), symbol, trade_side, entry_price, exit_price, profit])


##############################################
# 5. Position Management & Main Trading Logic
##############################################

def compute_risk_reward(decision, entry_price):
    """
    Compute the risk/reward ratio based on absolute price levels.
    For long positions: (TP Price - entry_price) / (entry_price - SL Price)
    For short positions: (entry_price - TP Price) / (SL Price - entry_price)
    """
    try:
        tp_price = float(decision["tp_price"])
        sl_price = float(decision["sl_price"])
        if decision["final_action"].upper() == "GO LONG":
            reward = tp_price - entry_price
            risk = entry_price - sl_price
        elif decision["final_action"].upper() == "GO SHORT":
            reward = entry_price - tp_price
            risk = sl_price - entry_price
        else:
            return None
        if risk <= 0:
            return None
        return reward / risk
    except Exception as e:
        print("Error computing risk-reward:", e)
        return None


def fetch_additional_data(symbol):
    """
    Merge basic additional data with extended market data.
    """
    base_data = fetch_order_book(symbol)
    extended_data = {
        "funding_rate": fetch_funding_rate(symbol),
        "open_interest": fetch_open_interest(symbol),
        "order_book_bid": base_data["bid"],
        "order_book_ask": base_data["ask"],
        "order_book_spread": base_data["spread"],
        "exchange_inflows": fetch_exchange_inflows(),
        "fear_and_greed_index": fetch_fear_and_greed_index(),
    }
    return extended_data


def main():
    wallet_balance = "1000 USDT"
    # Current position info: "NONE" initially.
    position_info = "NONE"
    in_position = False
    current_position_side = None
    current_position_entry_price = 0.0

    # Fetch multi-timeframe data (5m, 15m, 1h, 4h, 1d)
    multi_tf_data = fetch_multi_tf_data(SYMBOL, timeframes=["5m", "15m", "1h", "4h", "1d"], limit=300)
    if not multi_tf_data or "1h" not in multi_tf_data:
        print("Not enough data fetched.")
        return

    # Use 1h data as baseline for current price
    current_price = multi_tf_data["1h"]["current_price"]

    # Fetch extended market data (includes crawled net inflow, high leverage ratio, and liquidation map info)
    extended_data = fetch_additional_data(SYMBOL)

    # Determine aggregated trend based on whether current price is above SMA200 (for each timeframe)
    bullish_count = 0
    bearish_count = 0
    tf_directions = {}
    for tf, data in multi_tf_data.items():
        if data["sma200"] is not None and data["current_price"] > data["sma200"]:
            bullish_count += 1
            tf_directions[tf] = "bullish"
        else:
            bearish_count += 1
            tf_directions[tf] = "bearish"
    aggregated_trend = "BULL" if bullish_count >= bearish_count else "BEAR"
    aggregated_data = {"trend": aggregated_trend, "timeframe_directions": tf_directions}

    # Require strong multi-timeframe consensus: at least 4 out of 5 timeframes must agree
    consensus = (bullish_count >= 4 or bearish_count >= 4)
    if not consensus:
        print("No strong multi-timeframe consensus. No trade recommended.")
        return

    # Determine market regime and adjust indicator thresholds accordingly
    market_regime = determine_market_regime(multi_tf_data)  # "bull", "bear", or "sideways"
    thresholds = adjust_indicator_thresholds(market_regime)

    # Generate GPT-based trading decision using the detailed XML prompt (with dynamic criteria)
    try:
        gpt_response = generate_trading_decision(wallet_balance, position_info, aggregated_data,
                                                 extended_data, multi_tf_data, market_regime, thresholds)
        print("Raw GPT Response:")
        print(gpt_response)
        decision = parse_trading_decision(gpt_response)
        print("Parsed Decision:")
        print(decision)
        log_decision(decision, SYMBOL)
    except Exception as e:
        print("Error generating/parsing GPT decision:", e)
        return

    # Position Management: Open new position only if risk/reward ratio is acceptable and decision is GO LONG/GO SHORT.
    rr = compute_risk_reward(decision, current_price)
    if not in_position:
        if rr is not None and rr >= 2 and decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
            print(f"Opening new position: {decision['final_action']} at {current_price}")
            log_open_position(SYMBOL, decision, current_price)
            in_position = True
            current_position_side = decision["final_action"].split()[-1]  # "LONG" or "SHORT"
            current_position_entry_price = current_price
        else:
            print("No favorable trade setup found based on risk/reward criteria.")
    else:
        # If already in position, check if GPT recommends to exit (if not HOLD LONG/HOLD SHORT)
        if decision["final_action"].upper() not in ["HOLD LONG", "HOLD SHORT"]:
            print(f"Closing position for {SYMBOL} at price {current_price}")
            log_closed_position(SYMBOL, current_position_entry_price, current_price, current_position_side)
            in_position = False
        else:
            print("Holding current position.")


if __name__ == "__main__":
    main()
