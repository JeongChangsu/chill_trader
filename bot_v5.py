import os
import re
import ta
import csv
import time
import ccxt
import base64
import logging
import requests

import numpy as np
import pandas as pd
import undetected_chromedriver as uc  # Free data crawling using a stealth driver

from openai import OpenAI
from datetime import datetime
from selenium.webdriver.common.by import By

# ===========================================
# Logging configuration (logs in Korean for overall system messages)
# ===========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ===========================================
# Global constants and API initialization
# ===========================================
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

# OPENAI_API_KEY must be set in the environment
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# Telegram variables (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')


# ===========================================
# Telegram message sending function
# ===========================================
def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        logging.error("Telegram bot token or chat ID is not set.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            logging.info("Telegram message sent successfully")
        else:
            logging.error(f"Telegram message sending failed (status code: {response.status_code})")
    except Exception as e:
        logging.error(f"Error during Telegram message sending: {e}")


# ===========================================
# Persistent Driver Setup for Data Crawling
# ===========================================
drivers = {}


def create_driver():
    """Create a new undetected_chromedriver instance."""
    options = uc.ChromeOptions()
    # Configure options if needed (e.g., headless mode)
    driver = uc.Chrome(options=options)
    return driver


def get_driver(session_id='default_session'):
    global drivers
    if session_id in drivers and drivers[session_id] is not None:
        return drivers[session_id]
    else:
        driver = create_driver()
        drivers[session_id] = driver
        return driver


# ===========================================
# 1. Data Collection and Technical Indicator Calculation
# ===========================================
def fetch_ohlcv(symbol, timeframe, limit=300):
    """
    Fetch OHLCV data from Binance via ccxt and return as a pandas DataFrame.
    """
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        logging.info(f"{symbol} / {timeframe} OHLCV data fetched successfully")
    except Exception as e:
        logging.error(f"Error fetching {symbol} / {timeframe} OHLCV data: {e}")
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def compute_technical_indicators(df):
    """
    Compute various technical indicators and add them to the DataFrame.
    Included: RSI, SMA20, SMA50, SMA200, MACD, ATR, OBV, MFI, Bollinger Bands, MA difference percentages.
    """
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['sma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()

    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()

    df['ma50_diff'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['ma200_diff'] = (df['close'] - df['sma200']) / df['sma200'] * 100

    logging.info("Technical indicators calculated")
    return df


def fetch_order_book(symbol):
    """
    Fetch order book data from Binance and return the top bid, ask, and spread.
    """
    exchange = ccxt.binance()
    try:
        order_book = exchange.fetch_order_book(symbol)
        bid = order_book['bids'][0][0] if order_book['bids'] else None
        ask = order_book['asks'][0][0] if order_book['asks'] else None
        spread = round(ask - bid, 2) if bid and ask else None
        logging.info(f"{symbol} order book data fetched")
    except Exception as e:
        logging.error(f"Error fetching order book for {symbol}: {e}")
        bid = ask = spread = None
    return {"bid": bid, "ask": ask, "spread": spread}


# ===========================================
# 1-2. Extended Data Collection (Crawling, On-chain Data, etc.)
# ===========================================
def fetch_exchange_inflows():
    """
    Crawl exchange inflow/outflow data from a free source (e.g., CryptoQuant).
    Adjust XPath as needed for the actual site structure.
    """
    url = "https://cryptoquant.com/asset/btc/chart/exchange-flows"  # Example URL
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(3)
        netflow_all_text = driver.find_element(By.XPATH, '//tbody[@class="ant-table-tbody"]/tr[6]').text

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
            logging.info("Exchange inflow/outflow data crawled successfully")
            return net_inflow
    except Exception as e:
        logging.error(f"Error crawling exchange inflow/outflow data: {e}")
        return "N/A"


def fetch_funding_rate(symbol):
    """
    Retrieve funding rate using Binance Futures data.
    """
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        funding_info = futures_exchange.fetch_funding_rate(symbol=symbol)
        latest_funding = funding_info['info']['lastFundingRate'] if 'info' in funding_info else None
        logging.info(f"{symbol} funding rate fetched successfully")
        return latest_funding
    except Exception as e:
        logging.error(f"Error fetching funding rate for {symbol}: {e}")
        return "N/A"


def fetch_open_interest(symbol):
    """
    Retrieve open interest data using Binance Futures data.
    """
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        symbol_futures = symbol.replace("/", "")
        oi_response = futures_exchange.fetch_open_interest(symbol=symbol_futures)
        open_interest = oi_response['openInterest'] if oi_response and 'openInterest' in oi_response else None
        logging.info(f"{symbol} open interest fetched successfully")
        return open_interest
    except Exception as e:
        logging.error(f"Error fetching open interest for {symbol}: {e}")
        return "N/A"


def fetch_fear_and_greed_index():
    """
    Retrieve the Fear & Greed Index from Alternative.me.
    """
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        data = response.json()
        value = data['data'][0]['value'] if 'data' in data and len(data['data']) > 0 else None
        classification = data['data'][0]['value_classification'] if 'data' in data and len(data['data']) > 0 else None
        logging.info("Fear & Greed Index fetched successfully")
        return classification, value
    except Exception as e:
        logging.error(f"Error fetching Fear & Greed Index: {e}")
        return None, None


def fetch_onchain_data(symbol):
    """
    Retrieve on-chain metrics (e.g., MVRV, SOPR).
    In production, use appropriate APIs; here dummy data is used for illustration.
    """
    try:
        driver = get_driver()
        driver.get('https://kr.tradingview.com/chart/?symbol=INTOTHEBLOCK%3ABTC_MVRV')
        time.sleep(3)
        mvrv_text = driver.find_element(By.XPATH, '//span[contains(@class, "priceWrapper")]/span').text
        mvrv = float(mvrv_text.replace(",", ""))

        driver.get('https://kr.tradingview.com/chart/?symbol=GLASSNODE%3ABTC_SOPR')
        time.sleep(3)
        sopr_text = driver.find_element(By.XPATH, '//span[contains(@class, "priceWrapper")]/span').text
        sopr = float(sopr_text.replace(",", ""))

        logging.info("On-chain data (MVRV, SOPR) fetched successfully")
        return {"mvrv": mvrv, "sopr": sopr}
    except Exception as e:
        logging.error(f"Error fetching on-chain data: {e}")
        return {"mvrv": "N/A", "sopr": "N/A"}


def fetch_multi_tf_data(symbol, timeframes=None, limit=300):
    """
    Retrieve OHLCV data for multiple timeframes, compute technical indicators, and return a summary dict.
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
    logging.info("Multi-timeframe data and indicators calculated")
    return multi_tf_data


# ===========================================
# 1-3. Liquidation Heatmap Data Fetching and Analysis
# ===========================================
def fetch_liquidation_heatmap():
    """
    Retrieve liquidation heatmap data.
    In production, parse the data from a source such as https://coinank.com/liqHeatMapChart.
    Here we return dummy data for illustration.
    """
    url = "https://coinank.com/liqHeatMapChart"  # Example URL
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(3)

        if driver.find_element(By.XPATH, '//span[@class="anticon anticon-camera"]').is_displayed():
            driver.find_element(By.XPATH, '//span[@class="anticon anticon-camera"]').click()
            time.sleep(2)

            driver.quit()

        logging.info("Liquidation heatmap data fetched successfully")
    except Exception as e:
        logging.error(f"Error fetching liquidation heatmap data: {e}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_liquidation_heatmap():
    """
    Analyze the liquidation heatmap using a downloaded image.
    The image file path is '/Users/changpt/Downloads/Liquidation Heat Map.png'.
    After sending the image to GPT-4o and receiving the output, delete the image file.

    GPT-4o is instructed to output the analysis in a single line using the following format:
    "Long Liquidation Zone: <value>; Short Liquidation Zone: <value>; Impact: <analysis text>; Risk: <which side is at higher risk>."
    """
    image_path = "/Users/changpt/Downloads/Liquidation Heat Map.png"

    # Encode the image to base64
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        logging.error("Error encoding image: " + str(e))
        return "N/A"

    # Prepare the messages for GPT-4o:
    system_message = "You are a specialized analyst in crypto liquidations."
    user_message = [
        {
            "type": "text",
            "text": (
                "Please analyze the attached liquidation heatmap image for BTC futures. "
                "Identify the key liquidation zones, explain their potential impact on future price movements, "
                "and indicate which side (longs or shorts) is at higher risk. "
                "Output your analysis in a single line using the following format: "
                "\"Long Liquidation Zone: <value>; Short Liquidation Zone: <value>; Impact: <analysis text>; Risk: <which side is at higher risk>.\""
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
        )
        analysis_result = response.choices[0].message.content
    except Exception as e:
        logging.error("Error during GPT analysis of liquidation heatmap: " + str(e))
        analysis_result = "N/A"

    # Delete the image file after processing
    try:
        os.remove(image_path)
        logging.info("Deleted the liquidation heatmap image file after processing.")
    except Exception as e:
        logging.error("Error deleting the image file: " + str(e))

    logging.info("Liquidation heatmap analysis result:")
    logging.info(analysis_result)
    return analysis_result


# ===========================================
# 2. Market Regime Determination and Dynamic Indicator Threshold Adjustment
# ===========================================
def is_high_volatility(multi_tf_data):
    """
    Determine high volatility if the 1-hour ATR/price ratio exceeds 0.02 (2%).
    """
    data = multi_tf_data.get("1h")
    if data and data.get("atr") is not None and data.get("current_price") is not None:
        atr_ratio = data["atr"] / data["current_price"]
        return atr_ratio > 0.02
    return False


def determine_market_regime(multi_tf_data, onchain_data):
    """
    Determine the market regime (bull, bear, or sideways) based on 1-hour data using SMA50 and SMA200 relationships.
    If price is within ±1% of these MAs, consider it sideways.
    Also, if on-chain metrics (MVRV, SOPR) are extremely low, adjust the regime to sideways.
    Append '_high_vol' if 1h ATR/price ratio indicates high volatility.
    """
    data = multi_tf_data.get("1h")
    if data is None:
        logging.warning("1h data not available; defaulting to sideways")
        regime = "sideways"
    else:
        current_price = data["current_price"]
        sma50 = data["sma50"]
        sma200 = data["sma200"]
        if sma50 is None or sma200 is None:
            regime = "sideways"
        else:
            if abs(current_price - sma50) / sma50 < 0.01 and abs(current_price - sma200) / sma200 < 0.01:
                regime = "sideways"
            elif current_price > sma50 and current_price > sma200:
                regime = "bull"
            elif current_price < sma50 and current_price < sma200:
                regime = "bear"
            else:
                regime = "sideways"
    # Adjust regime based on on-chain data: if both MVRV and SOPR are below 1, set to sideways.
    if onchain_data["mvrv"] != "N/A" and onchain_data["sopr"] != "N/A":
        if onchain_data["mvrv"] < 1 and onchain_data["sopr"] < 1:
            logging.info("On-chain metrics are extreme; adjusting regime to sideways")
            regime = "sideways"
    if is_high_volatility(multi_tf_data):
        regime += "_high_vol"
    logging.info(f"Market regime determined: {regime.upper()}")
    return regime


def adjust_indicator_thresholds(market_regime):
    """
    Dynamically adjust RSI, MACD, and MA thresholds based on market regime.
    In high volatility conditions, thresholds are modified slightly.
    """
    if "bull" in market_regime:
        if "high_vol" in market_regime:
            thresholds = {
                "rsi_oversold": 40,
                "rsi_overbought": 85,
                "macd_comment": "In strong bullish conditions with high volatility, minor bearish MACD crossovers may be less significant.",
                "ma_comment": "Consider entering near a rebound at the 50MA support, but use wider stop-losses due to high volatility."
            }
        else:
            thresholds = {
                "rsi_oversold": 45,
                "rsi_overbought": 80,
                "macd_comment": "In strong bullish trends, ignore minor bearish MACD crossovers and consider entering during deep pullbacks.",
                "ma_comment": "Consider buying when the price rebounds at the 50MA support."
            }
    elif "bear" in market_regime:
        if "high_vol" in market_regime:
            thresholds = {
                "rsi_oversold": 15,
                "rsi_overbought": 60,
                "macd_comment": "In high volatility bear markets, note brief bullish MACD crossovers but prioritize momentum loss for entry.",
                "ma_comment": "Consider selling when the price rebounds at the 50MA resistance, and set wider stops due to high volatility."
            }
        else:
            thresholds = {
                "rsi_oversold": 20,
                "rsi_overbought": 55,
                "macd_comment": "In bear markets, ignore minor bullish MACD crossovers and enter when momentum confirms the move.",
                "ma_comment": "Consider selling when the price rebounds at the 50MA resistance."
            }
    else:  # sideways
        if "high_vol" in market_regime:
            thresholds = {
                "rsi_oversold": 35,
                "rsi_overbought": 75,
                "macd_comment": "In sideways markets with high volatility, allow more leeway for interpreting MACD signals.",
                "ma_comment": "Consider entering if the price repeatedly bounces around the 50MA, but exercise caution due to high volatility."
            }
        else:
            thresholds = {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "macd_comment": "In sideways markets, use MACD crossover signals as supplementary indicators.",
                "ma_comment": "Apply a mean reversion strategy around the 50MA."
            }
    logging.info(f"Indicator thresholds set for market regime {market_regime}")
    return thresholds


def choose_primary_timeframe(multi_tf_data):
    """
    Select the primary timeframe based on the maximum percentage difference between the current price and SMA200.
    """
    primary_tf = None
    max_diff = 0
    for tf, data in multi_tf_data.items():
        if data["sma200"] is not None:
            diff = abs(data["current_price"] - data["sma200"]) / data["sma200"]
            if diff > max_diff:
                max_diff = diff
                primary_tf = tf
    logging.info(f"Primary timeframe chosen: '{primary_tf}' (max deviation: {max_diff:.2%} from SMA200)")
    return primary_tf


# ===========================================
# 3. GPT Prompt Generation and Trading Decision
# ===========================================
def generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data, onchain_data, multi_tf_data,
                        market_regime, thresholds, heatmap_analysis):
    """
    Generate an XML-based detailed prompt for GPT.
    This prompt includes account info, multi-timeframe summary, technical/on-chain/sentiment data,
    liquidation heatmap analysis (with guidance), and dynamic decision rules.
    """
    # Generate multi-timeframe summary
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
    # Fear & Greed Index (classification, value)
    fng_class, fng_value = extended_data.get("fear_and_greed_index", ("N/A", "N/A"))

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
            <FearAndGreedIndex classification="{fng_class}">{fng_value}</FearAndGreedIndex>
            <OnChain>
                <MVRV>{onchain_data.get('mvrv', 'N/A')}</MVRV>
                <SOPR>{onchain_data.get('sopr', 'N/A')}</SOPR>
            </OnChain>
            <LiquidationHeatmap>
                <Guide>
                    Analyze the provided liquidation heatmap data using the following guidelines:
                    - Identify the key liquidation zones.
                    - Explain the potential impact of these zones on future price movements.
                    - Determine which side (longs or shorts) is at greater risk.
                    Output your analysis in a single line in the following format:
                    "Long Liquidation Zone: <value>; Short Liquidation Zone: <value>; Impact: <analysis text>; Risk: <which side is at higher risk>."
                </Guide>
                <Analysis>{heatmap_analysis}</Analysis>
            </LiquidationHeatmap>
        </AdditionalData>
    </MarketContext>
    <Indicators>
        <RSI>
            <Guide>
                In {market_regime.upper()} conditions, adjust RSI interpretation: consider oversold around {thresholds['rsi_oversold']} and overbought near {thresholds['rsi_overbought']}. Analyze divergence and failure swings.
            </Guide>
        </RSI>
        <MovingAverages>
            <SMA20>
                <Guide>
                    If price is above SMA20, add bullish weight; below, add bearish weight.
                </Guide>
            </SMA20>
            <SMA50>
                <Guide>{thresholds['ma_comment']}</Guide>
            </SMA50>
            <SMA200>
                <Guide>
                    A significant premium above SMA200 may indicate an overextended market; below SMA200 suggests bearish bias.
                </Guide>
            </SMA200>
        </MovingAverages>
        <BollingerBands>
            <Guide>
                Use Bollinger Bands to assess volatility. In trending markets, band walking near the upper band suggests continuation, while reversion to the middle may signal a pullback.
            </Guide>
        </BollingerBands>
        <MACD>
            <Guide>
                {thresholds['macd_comment']}
                Analyze histogram shapes, divergences, and zero-line crossovers.
            </Guide>
        </MACD>
        <ATR>
            <Guide>
                If ATR/Price < 0.005, the market is quiet (allowing higher leverage); if ATR/Price > 0.02, the market is volatile (use lower leverage).
            </Guide>
        </ATR>
        <OBV>
            <Guide>
                Rising OBV over the last 5 candles indicates bullish pressure; falling OBV indicates bearish pressure.
            </Guide>
        </OBV>
        <MFI>
            <Guide>
                MFI below 20 indicates oversold; above 80 indicates overbought conditions. Always cross-check with other indicators.
            </Guide>
        </MFI>
        <OnChainIndicators>
            <Guide>
                Consider on-chain metrics: MVRV below 1 may suggest undervaluation and accumulation; SOPR below 1 indicates coins are being sold at a loss.
            </Guide>
        </OnChainIndicators>
    </Indicators>
    <DecisionRules>
        <MultiTimeframe>
            <Guide>
                Analyze all available timeframes. A trade is recommended if either (a) at least 4 out of 5 timeframes agree on a directional bias, or (b) if the primary timeframe shows a strong signal (with at least 3 of 5 supporting it).
            </Guide>
            <TradeTerm>
                <Guide>
                    If lower timeframes (5m, 15m) dominate, set trade term as intraday/6h; if mid timeframes (1h, 4h) dominate, set as 1d; if 1d is dominant, set as 1w.
                </Guide>
            </TradeTerm>
        </MultiTimeframe>
        <RiskReward>
            <Guide>
                Only open a new position if the potential reward is at least 2x the risk.
            </Guide>
        </RiskReward>
        <StrategySwitch>
            <Guide>
                Based on the market regime, adopt a trend-following strategy in bullish conditions or a mean reversion approach in sideways or bearish conditions.
                Incorporate technical, on-chain, sentiment, and liquidation heatmap data into your decision.
            </Guide>
        </StrategySwitch>
    </DecisionRules>
    <Task>
        Based on the provided account information, market context, indicators (technical and on-chain), liquidation heatmap analysis, and dynamic decision rules, decide whether to GO LONG, GO SHORT, HOLD LONG, HOLD SHORT, or NO TRADE.
        Then, recommend a leverage multiplier (e.g., 3x, 5x, etc.), a trade term, a take profit price, a stop loss price, and a specific limit order price.
        Output your decision as a single comma-separated line with the following fields:
        Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Brief Rationale.
    </Task>
    <OutputExample>
        GO LONG, 5x, 6h, 11400, 10800, 11300, Majority of timeframes show bullish momentum with low volatility and adjusted RSI indicating a dip entry.
    </OutputExample>
</TradeBotPrompt>
    """
    return prompt


def generate_trading_decision(wallet_balance, position_info, aggregated_data, extended_data, onchain_data,
                              multi_tf_data, market_regime, thresholds, heatmap_analysis):
    prompt = generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data, onchain_data,
                                 multi_tf_data, market_regime, thresholds, heatmap_analysis)
    developer_prompt = (
        "You are an automated BTC/USDT trading assistant that follows detailed quantitative indicators and strict guidelines "
        "as defined in the provided XML. Do not deviate from the specified output format. "
        "Based on market regime and all provided data including liquidation heatmap analysis, adopt either a trend-following or mean reversion strategy accordingly."
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
    Parse GPT's response.
    Expected output format (comma-separated): Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Brief Rationale
    """
    response_text = response_text.strip()
    parts = [part.strip() for part in re.split(r'\s*,\s*', response_text)]
    if len(parts) < 7:
        raise ValueError("Incomplete response: at least 7 comma-separated fields are required.")
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


# ===========================================
# 4. Position Logging Functions
# ===========================================
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
    logging.info("Trading decision logged to file.")


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
    logging.info(f"{symbol} open position logged (entry price: {entry_price}).")


def log_closed_position(symbol, entry_price, exit_price, trade_side):
    profit = (exit_price - entry_price) if trade_side.upper() == "LONG" else (entry_price - exit_price)
    file_exists = os.path.isfile(CLOSED_POSITIONS_FILE)
    with open(CLOSED_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "trade_side", "entry_price", "exit_price", "profit"])
        writer.writerow([datetime.utcnow(), symbol, trade_side, entry_price, exit_price, profit])
    logging.info(f"{symbol} closed position logged (profit: {profit}).")


# ===========================================
# 5. Position Management and Main Trading Logic
# ===========================================
def compute_risk_reward(decision, entry_price):
    """
    Compute the risk/reward ratio based on absolute prices.
    For LONG: (TP Price - entry_price) / (entry_price - SL Price)
    For SHORT: (entry_price - TP Price) / (SL Price - entry_price)
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
        rr_ratio = reward / risk
        logging.info(f"Risk/Reward ratio computed: {rr_ratio:.2f}")
        return rr_ratio
    except Exception as e:
        logging.error(f"Error computing risk/reward ratio: {e}")
        return None


def fetch_additional_data(symbol):
    """
    Merge basic and extended data and return.
    """
    base_data = fetch_order_book(symbol)
    exchange_inflows = fetch_exchange_inflows()
    funding_rate = fetch_funding_rate(symbol)
    open_interest = fetch_open_interest(symbol)
    fear_and_greed = fetch_fear_and_greed_index()  # (classification, value)
    onchain = fetch_onchain_data(symbol)

    extended_data = {
        "order_book_bid": base_data.get("bid", "N/A"),
        "order_book_ask": base_data.get("ask", "N/A"),
        "order_book_spread": base_data.get("spread", "N/A"),
        "exchange_inflows": exchange_inflows,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "fear_and_greed_index": fear_and_greed  # Tuple (classification, value)
    }
    logging.info("Extended and on-chain data merged")
    return extended_data, onchain


def main():
    logging.info("Trading bot execution started")
    wallet_balance = "1000 USDT"
    position_info = "NONE"  # Initial position info
    in_position = False
    current_position_side = None
    current_position_entry_price = 0.0

    # Collect multi-timeframe data (5m, 15m, 1h, 4h, 1d)
    multi_tf_data = fetch_multi_tf_data(SYMBOL, timeframes=["5m", "15m", "1h", "4h", "1d"], limit=300)
    if not multi_tf_data or "1h" not in multi_tf_data:
        logging.error("Insufficient data collected.")
        return

    # Current price from 1h data
    current_price = multi_tf_data["1h"]["current_price"]
    logging.info(f"Current 1h price: {current_price}")

    # Extended data (order book, funding rate, open interest, exchange inflows, Fear & Greed Index, on-chain data)
    extended_data, onchain_data = fetch_additional_data(SYMBOL)

    # Liquidation heatmap: fetch and analyze via GPT-4o
    fetch_liquidation_heatmap()
    heatmap_analysis = analyze_liquidation_heatmap()

    # Count bullish/bearish signals across timeframes (using SMA200 as reference)
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
    logging.info(f"Timeframe signals: {tf_directions} (Aggregated: {aggregated_trend})")

    # Select primary timeframe (based on maximum deviation from SMA200)
    primary_tf = choose_primary_timeframe(multi_tf_data)
    primary_direction = tf_directions.get(primary_tf, None)
    logging.info(f"Primary timeframe: {primary_tf} (Direction: {primary_direction})")

    # Pre-filter: if clear signals are absent (e.g., not at least 4 out of 5 in agreement), do not call GPT
    if not (bullish_count >= 4 or bearish_count >= 4):
        if primary_direction is None or not ((primary_direction == "bullish" and bullish_count >= 3) or
                                             (primary_direction == "bearish" and bearish_count >= 3)):
            logging.info("No clear market signal detected; trading decision not initiated.")
            return

    # Determine market regime based on technical and on-chain data
    market_regime = determine_market_regime(multi_tf_data,
                                            onchain_data)  # e.g., bull, bear, sideways, bull_high_vol, etc.
    thresholds = adjust_indicator_thresholds(market_regime)

    # Generate trading decision using GPT prompt
    try:
        gpt_response = generate_trading_decision(wallet_balance, position_info, aggregated_data,
                                                 extended_data, onchain_data, multi_tf_data,
                                                 market_regime, thresholds, heatmap_analysis)
        logging.info("Raw GPT response:")
        logging.info(gpt_response)
        decision = parse_trading_decision(gpt_response)
        logging.info("Parsed trading decision:")
        logging.info(decision)
        log_decision(decision, SYMBOL)
    except Exception as e:
        logging.error(f"Error during GPT decision generation/parsing: {e}")
        return

    # Compute risk/reward ratio and send Telegram alert regardless of trade execution
    rr = compute_risk_reward(decision, current_price)
    rr_text = f"{rr:.2f}" if rr is not None else "N/A"

    # Send Telegram alert only if decision is to open a position (GO LONG or GO SHORT) with detailed info
    if decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
        action = "BUY" if decision["final_action"].upper() == "GO LONG" else "SELL"
        alert_msg = (f"{action} signal!\n"
                     f"Symbol: {SYMBOL}\n"
                     f"Entry Price: {current_price}\n"
                     f"Leverage: {decision['leverage']}\n"
                     f"Risk/Reward Ratio: {rr_text}\n"
                     f"Trade Term: {decision['trade_term']}\n"
                     f"Limit Order Price: {decision['limit_order_price']}\n"
                     f"TP: {decision['tp_price']}, SL: {decision['sl_price']}\n"
                     f"Rationale: {decision['rationale']}")

        send_telegram_message(alert_msg)
    else:
        logging.info("No position open signal; Telegram alert not sent.")

    # Position management: open new position only if risk/reward >= 2 and decision is GO LONG/GO SHORT
    if not in_position:
        if rr is not None and rr >= 2 and decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
            logging.info(
                f"Favorable trading setup detected. Entering position: {decision['final_action']} at {current_price}")
            log_open_position(SYMBOL, decision, current_price)
            in_position = True
            current_position_side = decision["final_action"].split()[-1]  # LONG or SHORT
            current_position_entry_price = current_price
        else:
            logging.info("No trading setup meeting risk/reward criteria found.")
    else:
        # If in position and decision is not HOLD LONG/HOLD SHORT, then exit the position.
        if decision["final_action"].upper() not in ["HOLD LONG", "HOLD SHORT"]:
            logging.info(f"Exiting position for {SYMBOL} at current price {current_price}")
            log_closed_position(SYMBOL, current_position_entry_price, current_price, current_position_side)
            in_position = False
        else:
            logging.info("Maintaining current position.")


if __name__ == "__main__":
    main()
