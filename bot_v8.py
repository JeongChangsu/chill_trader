# bot_v8.py
import os
import re
import ta
import csv
import time
import ccxt
import pytz
import logging
import requests

import numpy as np
import pandas as pd
import pandas_ta as pta
import undetected_chromedriver as uc

from PIL import Image
from google import genai
from openai import OpenAI
from bs4 import BeautifulSoup
from google.genai import types
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By

# =====================================================
# 1. 기본 설정 및 글로벌 변수
# =====================================================
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

# GOOGLE_API_KEY must be set in the environment
google_api_key = os.environ.get('GOOGLE_API_KEY')
gemini_client = genai.Client(api_key=google_api_key)

# Telegram variables (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# KST 타임존 설정
KST = pytz.timezone("Asia/Seoul")


# =====================================================
# 2. 텔레그램 메시지 전송 함수
# =====================================================
def send_telegram_message(message):
    """
    Telegram API를 사용하여 메시지를 전송한다.
    """
    if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        logging.error("Telegram bot token or chat ID is not set.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"  # Markdown 활성화
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            logging.info("Telegram message sent successfully")
        else:
            logging.error(f"Telegram message sending failed (status code: {response.status_code})")
    except Exception as e:
        logging.error(f"Error during Telegram message sending: {e}")


# =====================================================
# 3. Persistent Driver Setup (Data Crawling)
# =====================================================
drivers = {}


def create_driver():
    """
    undetected_chromedriver의 새로운 인스턴스를 생성하여 반환한다.
    """
    options = uc.ChromeOptions()
    # Configure options if needed (e.g., headless mode)
    # options.add_argument('--headless') # 필요에 따라 headless 모드 활성화
    driver = uc.Chrome(options=options)
    return driver


def get_driver(session_id='default_session'):
    """
    주어진 session_id에 해당하는 드라이버가 있으면 반환하고,
    없으면 새 드라이버를 생성한다.
    """
    global drivers
    if session_id in drivers and drivers[session_id] is not None:
        return drivers[session_id]
    else:
        driver = create_driver()
        drivers[session_id] = driver
        return driver


# =====================================================
# 4. 데이터 수집 및 기술적 지표 계산
# =====================================================
def fetch_ohlcv(symbol, timeframe, limit=300):
    """
    Binance의 ccxt 라이브러리를 이용하여 OHLCV 데이터를 가져오고 DataFrame으로 반환한다.
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
    입력받은 DataFrame에 RSI, EMA, MACD, ATR, Bollinger Bands, 캔들 패턴, 거래량 분석 지표를 계산하여 추가한다.
    """
    # 1. 추세 지표 (EMA, MACD)
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # 2. 모멘텀 지표 (RSI)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # 3. 변동성 지표 (ATR, Bollinger Bands)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()

    # 4. 캔들 패턴
    patterns_df = df.ta.cdl_pattern(name="all")  # 모든 캔들 패턴 계산

    # 캔들 패턴 컬럼명에 맞춰서 boolean 값 추출
    df['engulfing_bullish'] = patterns_df['CDL_ENGULFING'] > 0  # Bullish Engulfing (양수: Bullish)
    df['engulfing_bearish'] = patterns_df['CDL_ENGULFING'] < 0  # Bearish Engulfing (음수: Bearish)
    df['morning_star'] = patterns_df['CDL_MORNINGSTAR'] != 0  # Morning Star (0이 아니면 패턴)
    df['evening_star'] = patterns_df['CDL_EVENINGSTAR'] != 0  # Evening Star (0이 아니면 패턴)
    df['hammer'] = patterns_df['CDL_HAMMER'] != 0  # Hammer (0이 아니면 패턴)
    df['hanging_man'] = patterns_df['CDL_HANGINGMAN'] != 0  # Hanging Man (0이 아니면 패턴)
    df['doji'] = patterns_df['CDL_DOJI_10_0.1'] != 0  # Doji (0이 아니면 패턴, CDL_DOJI_10_0.1 컬럼 사용)

    # 5. 거래량 분석 (거래량 변화율, 거래량-가격 다이버전스, Volume Oscillator 추가) - ta library
    df['volume_change'] = df['volume'].pct_change() * 100  # Volume Change (%)

    vo_df = df.ta.kvo()  # KVO returns DataFrame
    df['volume_oscillator'] = vo_df['KVO_34_55_13']  # Select 'KVO_34_55_13' column

    # 가격 - EMA 차이 (percent diff) - ta library
    df['ema50_diff'] = (df['close'] - df['ema50']) / df['ema50'] * 100
    df['ema200_diff'] = (df['close'] - df['ema200']) / df['ema200'] * 100

    logging.info("Technical indicators calculated")
    return df


def calculate_volume_divergence(df, period=20):
    """
    거래량-가격 다이버전스 (Volume-Price Divergence) 를 분석하는 함수
    """
    # 1. 가격 상승 + 거래량 감소  => Bearish Divergence (weakening uptrend)
    price_up_volume_down = (df['close'].diff() > 0) & (df['volume'].diff() < 0)

    # 2. 가격 하락 + 거래량 감소  => Bullish Divergence (weakening downtrend)
    price_down_volume_down = (df['close'].diff() < 0) & (df['volume'].diff() < 0)

    df['bearish_divergence'] = price_up_volume_down.rolling(window=period).sum()
    df['bullish_divergence'] = price_down_volume_down.rolling(window=period).sum()

    logging.info("Volume divergence calculated.")
    return df


def fetch_order_book(symbol):
    """
    Binance에서 주문서 데이터를 가져와 최상위 bid, ask, spread 값을 반환한다.
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


def fetch_spot_future_price_diff(symbol):
    """
    현물-선물 가격 차이 (Spot-Future Price Difference) 를 계산
    """
    try:
        exchange = ccxt.binance()
        spot_ticker = exchange.fetch_ticker(symbol)
        future_ticker = exchange.fetch_ticker(symbol.replace('/', '') + ':USDT')  # 선물 심볼 가정

        spot_price = spot_ticker['last'] if spot_ticker else None
        future_price = future_ticker['last'] if future_ticker else None

        if spot_price and future_price:
            price_diff_percent = ((future_price - spot_price) / spot_price) * 100
            return round(price_diff_percent, 2)
        else:
            return None
    except Exception as e:
        logging.error(f"Error fetching spot-future price difference: {e}")
        return None


def fetch_bitcoin_dominance():
    """
    TradingView에서 비트코인 도미넌스(Bitcoin Dominance) 데이터를 크롤링
    """
    url = "https://kr.tradingview.com/symbols/BTC.D/"
    try:
        driver = get_driver()
        driver.get('https://kr.tradingview.com/chart/?symbol=CRYPTOCAP%3ABTC.D')
        time.sleep(3)
        dominance_text = driver.find_element(By.XPATH, '//span[contains(@class, "priceWrapper")]/span').text
        dominance_value = float(dominance_text.replace("%", "").replace(",", ""))  # % 문자, 쉼표 제거 후 float 변환
        logging.info(f"Bitcoin dominance fetched: {dominance_value}")
        return dominance_value
    except Exception as e:
        logging.error(f"Error fetching Bitcoin Dominance from TradingView: {e}")
        return None


# =====================================================
# 5. 확장 데이터 수집 (크롤링, 온체인 데이터 등)
# =====================================================
def fetch_exchange_inflows():
    """
    무료 소스(예: CryptoQuant)를 크롤링하여 거래소 순입출금 데이터를 반환한다.
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
    Binance Futures 데이터를 이용하여 funding rate를 가져온다.
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
    Binance Futures 데이터를 이용하여 open interest 데이터를 가져온다.
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
    Alternative.me API를 사용하여 Fear & Greed Index를 가져온다.
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
    온체인 지표(MVRV, SOPR 등)를 적절한 소스(API 또는 크롤링)에서 가져와 반환한다.
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
    여러 타임프레임(예: 5m, 15m, 1h, 4h, 1d)의 OHLCV 데이터를 가져오고
    기술적 지표를 계산하여 요약 정보를 반환한다.
    """
    if timeframes is None:
        timeframes = ["5m", "15m", "1h", "4h", "1d"]
    multi_tf_data = {}
    for tf in timeframes:
        df = fetch_ohlcv(symbol, tf, limit)
        if df is None:
            continue
        df = compute_technical_indicators(df)
        df = calculate_volume_divergence(df)  # 거래량 다이버전스 추가
        latest = df.iloc[-1]
        multi_tf_data[tf] = {
            "current_price": round(latest['close'], 2),
            "rsi": round(latest['rsi'], 2) if not np.isnan(latest['rsi']) else None,
            "ema20": round(latest['ema20'], 2) if not np.isnan(latest['ema20']) else None,  # EMA 추가
            "ema50": round(latest['ema50'], 2) if not np.isnan(latest['ema50']) else None,  # EMA 추가
            "ema200": round(latest['ema200'], 2) if not np.isnan(latest['ema200']) else None,  # EMA 추가
            "ema50_diff": round(latest['ema50_diff'], 2) if not np.isnan(latest['ema50_diff']) else None,  # EMA diff
            "ema200_diff": round(latest['ema200_diff'], 2) if not np.isnan(latest['ema200_diff']) else None,  # EMA diff
            "bb_upper": round(latest['bb_upper'], 2) if not np.isnan(latest['bb_upper']) else None,
            "macd": round(latest['macd'], 2) if not np.isnan(latest['macd']) else None,
            "macd_signal": round(latest['macd_signal'], 2) if not np.isnan(latest['macd_signal']) else None,
            "atr": round(latest['atr'], 2) if not np.isnan(latest['atr']) else None,
            "volume_change": round(latest['volume_change'], 2) if not np.isnan(latest['volume_change']) else None,
            "volume_oscillator": round(latest['volume_oscillator'], 2) if not np.isnan(
                latest['volume_oscillator']) else None,  # Volume Oscillator
            "bearish_divergence": latest['bearish_divergence'],  # 거래량-가격 Bearish Divergence
            "bullish_divergence": latest['bullish_divergence'],  # 거래량-가격 Bullish Divergence
            "engulfing_bullish": latest['engulfing_bullish'],  # Bullish Engulfing pattern
            "engulfing_bearish": latest['engulfing_bearish'],  # Bearish Engulfing pattern
            "morning_star": latest['morning_star'],  # Morning Star pattern
            "evening_star": latest['evening_star'],  # Evening Star pattern
            "hammer": latest['hammer'],  # Hammer pattern
            "hanging_man": latest['hanging_man'],  # Hanging Man pattern
            "doji": latest['doji'],  # Doji pattern
            "timestamp": latest['timestamp'],
            "df_full": df
        }
    logging.info("Multi-timeframe data and indicators calculated")
    return multi_tf_data


# =====================================================
# 6. 청산 히트맵 데이터 및 분석
# =====================================================
def fetch_liquidation_heatmap():
    """
    청산 히트맵 데이터를 CoinAnk 사이트에서 다운로드한다.
    """
    url = "https://coinank.com/liqHeatMapChart"
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(5)  # Increased wait time

        if driver.find_element(By.XPATH, '//span[@class="anticon anticon-camera"]').is_displayed():
            driver.find_element(By.XPATH, '//span[@class="anticon anticon-camera"]').click()
            time.sleep(3)
            driver.quit()

        logging.info("Liquidation heatmap data fetched successfully")
    except Exception as e:
        logging.error(f"Error fetching liquidation heatmap data: {e}")


def analyze_liquidation_heatmap_gemini(image_path):
    """
    다운로드된 청산 히트맵 이미지를 Gemini Pro Vision API를 통해 분석하고,
    지정된 포맷의 분석 결과를 반환한다. 이미지를 Gemini에게 직접 전달.
    """
    image = Image.open(image_path)

    try:
        prompt_parts = [
            image,
            """
Analyze this liquidation heatmap for [Asset, e.g., BTC] futures.

1.  **Identify Key Liquidation Levels:**
    *   Clearly define the price ranges for the most significant **long liquidation zones** (below the current price).
    *   Clearly define the price ranges for the most significant **short liquidation zones** (above the current price).
    *   Mention any secondary or less intense liquidation zones if notable.

2.  **Explain Potential Price Impact:**
    *   Describe how price might react when approaching or entering these liquidation zones.
    *   Explain the expected direction of price movement upon triggering long vs. short liquidations.
    *   Discuss the potential for increased volatility around these levels.

3.  **Determine Liquidation Risk Balance:**
    *   Compare the intensity and concentration of long vs. short liquidation zones.
    *   Explicitly state whether **longs or shorts face a higher liquidation risk** based on the heatmap.
    *   Explain your reasoning based on the heatmap's visual cues (color intensity, zone density).

Format your response concisely in English, like:
"**Long Liquidation Zone:** \$[Price Range]; **Short Liquidation Zone:** \$[Price Range]; **Impact:** [Explain potential price action]; **Risk:** [Longs or Shorts are at higher risk].
"""
        ]

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=prompt_parts)

        analysis_result = response.text
        logging.info("Gemini Liquidation Heatmap Analysis Raw Response:")
        logging.info(analysis_result)  # Raw response 출력
        return analysis_result

    except Exception as e:
        logging.error(f"Error during Gemini analysis of liquidation heatmap: {e}")
        formatted_analysis = "N/A"

    # Delete the image file after processing
    try:
        os.remove(image_path)
        logging.info("Deleted the liquidation heatmap image file after processing.")
    except Exception as e:
        logging.error("Error deleting the image file: {e}")

    logging.info("Liquidation heatmap analysis result:")
    logging.info(formatted_analysis)
    return formatted_analysis


# =====================================================
# 7. 시장 상태 결정 및 지표 임계값 조정
# =====================================================
def is_high_volatility(multi_tf_data):
    """
    1시간봉의 ATR/현재가 비율을 기준으로 고변동성 여부를 판단한다.
    """
    data = multi_tf_data.get("1h")
    if data and data.get("atr") is not None and data.get("current_price") is not None:
        atr_ratio = data["atr"] / data["current_price"]
        return atr_ratio > 0.02  # 임계값 조정 (optional)
    return False


def determine_market_regime(multi_tf_data, onchain_data):
    """
    EMA50, EMA200, RSI, 거래량, 캔들 패턴, 온체인 데이터 등을 활용하여
    더 세분화된 시장 상태를 결정한다.
    """
    tf_1h_data = multi_tf_data.get("1h")
    tf_4h_data = multi_tf_data.get("4h")
    tf_1d_data = multi_tf_data.get("1d")

    if not tf_1h_data or not tf_4h_data or not tf_1d_data:
        logging.warning("Missing timeframe data for regime determination.")
        return "sideways_normal_volatility"  # Default regime

    price_1h = tf_1h_data["current_price"]
    ema50_1h = tf_1h_data["ema50"]
    ema200_1h = tf_1h_data["ema200"]
    rsi_1h = tf_1h_data["rsi"]

    price_4h = tf_4h_data["current_price"]
    ema50_4h = tf_4h_data["ema50"]
    ema200_4h = tf_4h_data["ema200"]
    rsi_4h = tf_4h_data["rsi"]

    price_1d = tf_1d_data["current_price"]
    ema200_1d = tf_1d_data["ema200"]

    # 기본 Regime: sideways_normal_volatility
    regime = "sideways_normal_volatility"

    # 1. 추세 기반 Regime 판단 (EMA, Price 위치 관계)
    if price_1h > ema50_1h and price_1h > ema200_1h and price_4h > ema200_4h and price_1d > ema200_1d:
        regime = "bull_trend"  # 강세 추세
    elif price_1h < ema50_1h and price_1h < ema200_1h and price_4h < ema200_4h and price_1d < ema200_1d:
        regime = "bear_trend"  # 약세 추세
    else:
        regime = "sideways_normal_volatility"  # 횡보 (EMA 밀집 or 혼조세)

    # 2. 변동성 Regime (ATR)
    if is_high_volatility(multi_tf_data):
        regime = regime.replace("normal", "high")  # 변동성 증가

    # 3. RSI 활용한 추가 조건 (과매수/과매도 영역 진입) - 추세 반전 or 되돌림 가능성 감지
    if rsi_1h > 70 or rsi_4h > 70:
        regime += "_overbought"  # 과매수 영역
    elif rsi_1h < 30 or rsi_4h < 30:
        regime += "_oversold"  # 과매도 영역

    # 4. 온체인 데이터 활용 (extreme MVRV/SOPR conditions) - Extreme Fear/Uncertainty 시 Sideways
    if onchain_data["mvrv"] != "N/A" and onchain_data["sopr"] != "N/A":
        if onchain_data["mvrv"] < 1 and onchain_data["sopr"] < 1:
            regime = "sideways_extreme_fear"  # Extreme Fear 상태

    logging.info(f"Market regime determined: {regime.upper()}")
    return regime


def adjust_indicator_thresholds(market_regime):
    """
    시장 상태에 따라 RSI, MACD, MA 등의 임계값을 동적으로 조정한다.
    더 세분화된 Regime에 따라 Thresholds 설정.
    """
    thresholds = {}  # Thresholds dictionary 초기화

    if "bull_trend" in market_regime:
        if "high_volatility" in market_regime:
            thresholds = {
                "rsi_oversold": 40,
                "rsi_overbought": 85,
                "rsi_trend_follow": 50,  # New: 추세 추종 RSI 기준
                "macd_signal_cross_lookback": 5,  # MACD Signal Cross Lookback
                "ema50_correction_percent": 2.0,  # EMA50 Correction % 기준 (ex: 2% band)
                "atr_stop_loss_multiplier": 2.5,  # ATR Stop Loss multiplier for Bull Trend High Volatility
                "atr_take_profit_multiplier": 3.5  # ATR Take Profit multiplier for Bull Trend High Volatility
            }
        else:  # bull_trend + normal_volatility
            thresholds = {
                "rsi_oversold": 45,
                "rsi_overbought": 80,
                "rsi_trend_follow": 55,  # 추세 추종 RSI 기준 (normal vol)
                "macd_signal_cross_lookback": 3,  # MACD Signal Cross Lookback (normal vol)
                "ema50_correction_percent": 1.5,  # EMA50 Correction % 기준 (normal vol)
                "atr_stop_loss_multiplier": 2.0,  # ATR Stop Loss multiplier for Bull Trend Normal Volatility
                "atr_take_profit_multiplier": 3.0  # ATR Take Profit multiplier for Bull Trend Normal Volatility
            }
    elif "bear_trend" in market_regime:
        if "high_volatility" in market_regime:
            thresholds = {
                "rsi_oversold": 15,
                "rsi_overbought": 60,
                "rsi_trend_follow": 50,  # 추세 추종 RSI 기준 (bear market)
                "macd_signal_cross_lookback": 5,  # MACD Signal Cross Lookback (bear market)
                "ema50_bounce_percent": 2.0,  # EMA50 Bounce % 기준 (bear market)
                "atr_stop_loss_multiplier": 2.5,  # ATR Stop Loss multiplier for Bear Trend High Volatility
                "atr_take_profit_multiplier": 3.5  # ATR Take Profit multiplier for Bear Trend High Volatility
            }
        else:  # bear_trend + normal_volatility
            thresholds = {
                "rsi_oversold": 20,
                "rsi_overbought": 55,
                "rsi_trend_follow": 45,  # 추세 추종 RSI 기준 (bear market, normal vol)
                "macd_signal_cross_lookback": 3,  # MACD Signal Cross Lookback (bear market, normal vol)
                "ema50_bounce_percent": 1.5,  # EMA50 Bounce % 기준 (bear market, normal vol)
                "atr_stop_loss_multiplier": 2.0,  # ATR Stop Loss multiplier for Bear Trend Normal Volatility
                "atr_take_profit_multiplier": 3.0  # ATR Take Profit multiplier for Bear Trend Normal Volatility
            }
    elif "sideways" in market_regime:  # Sideways Regime
        if "high_volatility" in market_regime:
            thresholds = {
                "rsi_oversold": 35,
                "rsi_overbought": 75,
                "rsi_reversal": 40,  # New: 횡보장 RSI Reversal 기준
                "macd_histogram_divergence_lookback": 7,  # MACD Histogram Divergence Lookback
                "bb_band_bounce_percent": 1.0,  # 볼린저 밴드 Band Bounce % 기준
                "atr_stop_loss_multiplier": 1.5,  # ATR Stop Loss Multiplier for Sideways High Volatility
                "atr_take_profit_multiplier": 2.0  # ATR Take Profit Multiplier for Sideways High Volatility
            }
        else:  # sideways + normal_volatility
            thresholds = {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "rsi_reversal": 45,  # 횡보장 RSI Reversal 기준 (normal vol)
                "macd_histogram_divergence_lookback": 5,  # MACD Histogram Divergence Lookback (normal vol)
                "bb_band_bounce_percent": 0.8,  # 볼린저 밴드 Band Bounce % 기준 (normal vol)
                "atr_stop_loss_multiplier": 1.2,  # ATR Stop Loss Multiplier for Sideways Normal Volatility
                "atr_take_profit_multiplier": 1.8  # ATR Take Profit Multiplier for Sideways Normal Volatility
            }
    else:  # Default Thresholds (for unexpected regime)
        thresholds = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_stop_loss_multiplier": 1.5,
            "atr_take_profit_multiplier": 2.0
        }

    logging.info(f"Indicator thresholds set for market regime: {market_regime}")
    return thresholds


def choose_primary_timeframe(market_regime):
    """
    시장 Regime 및 전략에 따라 Primary Timeframe 선정 로직 변경.
    """
    if "trend" in market_regime:  # Trend following regime (Bull/Bear Trend)
        return "4h"  # 4시간봉 (or 1h) - 중장기 추세 추종
    elif "sideways" in market_regime:  # Sideways regime
        return "15m"  # 15분봉 (or 5m, 1h) - 단기 Mean Reversion
    else:  # Default timeframe
        return "1h"  # 1시간봉 - default


# =====================================================
# 8. Gemini 프롬프트 생성 및 거래 결정
# =====================================================
def generate_gemini_prompt(wallet_balance, position_info, extended_data,
                           onchain_data, multi_tf_data, market_regime, thresholds,
                           heatmap_analysis, econ_summary, primary_tf, current_session):
    """
    Gemini Pro 모델에 전달할 Prompt 생성. XML 대신 텍스트 기반, 상세 가이드 및 우선순위 명시.
    """

    # Multi-timeframe 분석 요약
    tf_summary_lines = []
    for tf, data in multi_tf_data.items():
        tf_summary_lines.append(
            f"**{tf} Timeframe Analysis:**\n"
            f"- Price: {data['current_price']:.2f}, RSI: {data['rsi']:.2f}\n"
            f"- EMA20: {data['ema20']:.2f}, EMA50: {data['ema50']:.2f} (Diff: {data['ema50_diff']:.2f}%), EMA200: {data['ema200']:.2f} (Diff: {data['ema200_diff']:.2f}%)\n"
            f"- Bollinger Bands (Upper): {data['bb_upper']:.2f}\n"
            f"- MACD: {data['macd']:.2f} (Signal: {data['macd_signal']:.2f})\n"
            f"- ATR: {data['atr']:.2f}, Volume Change: {data['volume_change']:.2f}%, Volume Oscillator: {data['volume_oscillator']:.2f}\n"  # Volume Oscillator 추가
            f"- Volume Divergence: Bearish={data['bearish_divergence']}, Bullish={data['bullish_divergence']}\n"
            f"- Candle Patterns: Engulfing Bullish={data['engulfing_bullish']}, Engulfing Bearish={data['engulfing_bearish']}, Morning Star={data['morning_star']}, Evening Star={data['evening_star']}, Hammer={data['hammer']}, Hanging Man={data['hanging_man']}, Doji={data['doji']}\n"
        )
    multi_tf_summary = "\n".join(tf_summary_lines)

    fng_class, fng_value = extended_data.get("fear_and_greed_index", ("N/A", "N/A"))
    order_book_data = extended_data.get('order_book', {})

    prompt_text = f"""
Your goal is to make optimal trading decisions based on a comprehensive analysis of market data.

**Current Trading Session (KST):** **{current_session}**

**Account Status:**
- Wallet Balance: {wallet_balance}
- Current Position: {position_info}

**Market Context:**
- Market Regime: **{market_regime.upper()}**
- Primary Timeframe for Decision: **{primary_tf}**

**Multi-Timeframe Technical Analysis Summary:**
{multi_tf_summary}

**Additional Market Data:**
- Funding Rate: {extended_data.get('funding_rate', 'N/A')}
- Open Interest: {extended_data.get('open_interest', 'N/A')}
- Order Book: Bid={order_book_data.get('bid', 'N/A')}, Ask={order_book_data.get('ask', 'N/A')}, Spread={order_book_data.get('spread', 'N/A')}
- Exchange Net Inflow: {extended_data.get('exchange_inflows', 'N/A')}
- Fear and Greed Index: Classification={fng_class}, Value={fng_value}
- On-Chain Data: MVRV={onchain_data.get('mvrv', 'N/A')}, SOPR={onchain_data.get('sopr', 'N/A')}
- Liquidation Heatmap Analysis: {heatmap_analysis}
- Economic Calendar Events: {econ_summary}
- Spot-Future Price Diff: {extended_data.get('spot_future_price_diff', 'N/A')}%
- Bitcoin Dominance: {extended_data.get('bitcoin_dominance', 'N/A')}%

**Indicator Guidelines & Priorities:**

1. **RSI (Priority: High):**
   - Oversold: <={thresholds.get('rsi_oversold', 30)}, Overbought: >={thresholds.get('rsi_overbought', 70)}.
   - In trend regimes (bull/bear): Use RSI to confirm momentum and identify potential continuation or pullback entries. RSI Trend Follow Level: {thresholds.get('rsi_trend_follow', 50)}.
   - In sideways regimes: Use RSI for reversal signals at extremes. RSI Reversal Level: {thresholds.get('rsi_reversal', 45)}.

2. **EMA (Priority: High):**
   - EMA20, EMA50, EMA200: Analyze price position relative to EMAs for trend direction.
   - EMA50 & EMA200 Crossovers: Confirm trend shifts.
   - EMA50 Correction/Bounce: In bull/bear trends, look for price corrections to EMA50 as entry points. Correction Percent: ±{thresholds.get('ema50_correction_percent', 1.5)}%. Bounce Percent (Bear): ±{thresholds.get('ema50_bounce_percent', 1.5)}%.

3. **MACD (Priority: Medium):**
   - Signal Line Crossovers: Use for potential entry/exit signals, especially in sideways or early trend regimes. Lookback period for signal cross confirmation: {thresholds.get('macd_signal_cross_lookback', 3)} candles.
   - Histogram Divergence: In sideways regimes, histogram divergence from price action can signal reversals. Lookback period for histogram divergence: {thresholds.get('macd_histogram_divergence_lookback', 5)} candles.

4. **Bollinger Bands (Priority: Medium):**
   - Band Expansion/Contraction: Assess volatility.
   - Band Bounce: In sideways markets, bounces off Bollinger Bands can signal mean reversion opportunities. Band Bounce Percent: ±{thresholds.get('bb_band_bounce_percent', 0.8)}%.
   - Band Walk: In trending markets, price "walking" along the upper/lower band indicates strong trend continuation.

5. **Candle Patterns (Priority: Low-Medium):**
   - Engulfing, Morning/Evening Star, Hammer/Hanging Man, Doji: Use as supplementary confirmation at key levels (support/resistance, EMA levels).

6. **Volume Analysis (Priority: Medium):**
   - Volume Surge: Confirm breakouts or breakdowns. Significant volume increase on price movement adds conviction.
   - Volume Divergence: Bearish divergence (price up, volume down) warns of uptrend weakness; bullish divergence (price down, volume down) suggests downtrend weakening.

7. **ATR (Priority: High for Risk Management):**
   - Volatility Assessment: ATR/Price ratio > 0.02 indicates high volatility, adjust leverage and stop-loss accordingly.
   - Stop Loss & Take Profit: Use ATR multipliers for dynamic stop-loss and take-profit levels. Stop Loss Multiplier: {thresholds.get('atr_stop_loss_multiplier', 1.5)}, Take Profit Multiplier: {thresholds.get('atr_take_profit_multiplier', 2.0)}.

8. **Order Book (Priority: Low):**
   - Bid/Ask Spread: Monitor spread for liquidity assessment. Wider spread in high volatility, tighter in normal conditions.

9. **On-Chain Data & Fear/Greed (Priority: Low-Medium for broad market context):**
   - MVRV & SOPR: Assess overall market valuation and sentiment. Extreme undervaluation (MVRV<1, SOPR<1) may suggest sideways or accumulation phases.
   - Fear & Greed Index: Sentiment context. Extreme fear/greed can be contrarian indicators.

10. **Economic Calendar (Priority: Low-Medium for short-term volatility spikes):**
    - High Impact Events: Be aware of major economic announcements that can cause volatility spikes. Consider tightening stops or reducing leverage before major events.

**Session-Based Trading Patterns (KST):**

- **OVERNIGHT_SESSION (00:00-08:00):** Low liquidity, watch for fake breakouts (Bart patterns). Strategy: Be cautious of low volume breakouts, consider betting against them, use tighter stop-losses.
- **ASIAN_SESSION (08:00-16:00):** Medium volatility, trend continuation or sideways. Session open (08:00 KST) may have volatility. Strategy: If trend persists after 09:00 KST, follow the trend. Otherwise, expect range-bound trading. Look for breakouts at 08:00 KST open.
- **LONDON_SESSION (16:00-22:00):** High liquidity, trend setting session, major breakouts. Session open (16:00 KST) has high volatility. Strategy: Identify direction after 16:00 KST open and trade the trend. Be ready for breakouts at 16:00 KST open.
- **US_SESSION (22:00-06:00):** Highest volume and volatility, strong trend continuation or reversals. US session often follows London session trend but reversals can occur. Economic news releases (22:30-23:30 KST) cause high volatility. Strategy: Follow London session trend if it continues. Watch for reversal signals. For news events, avoid entering right at release, wait for secondary reaction.
- **TRANSITION_SESSION (06:00-08:00):** Transition period between US close and Asian open. Liquidity may be recovering from overnight lows. Strategy: Be cautious, may see some trend establishment ahead of Asian session.

**Regime-Specific Strategy Guidelines (within Sessions):**

- **Bull Trend Regime:**
    - Strategy: Trend Following, Buy the Dip.
    - Entry Signals: RSI pullback to {thresholds.get('rsi_trend_follow', 50)} level, price correction to EMA50, Bullish candle patterns at EMA50 support.
    - Exit/Take Profit: ATR Take Profit Multiplier: {thresholds.get('atr_take_profit_multiplier', 3.0)}x ATR from entry, resistance levels.
    - Stop Loss: ATR Stop Loss Multiplier: {thresholds.get('atr_stop_loss_multiplier', 2.0)}x ATR from entry, below EMA50 support.

- **Bear Trend Regime:**
    - Strategy: Trend Following, Sell the Rally.
    - Entry Signals: RSI rebound to {thresholds.get('rsi_trend_follow', 50)} level, price bounce off EMA50 resistance, Bearish candle patterns at EMA50 resistance.
    - Exit/Take Profit: ATR Take Profit Multiplier: {thresholds.get('atr_take_profit_multiplier', 3.0)}x ATR from entry, support levels.
    - Stop Loss: ATR Stop Loss Multiplier: {thresholds.get('atr_stop_loss_multiplier', 2.0)}x ATR from entry, above EMA50 resistance.

- **Sideways Regime:**
    - Strategy: Mean Reversion, Range Trading.
    - Entry Signals: RSI Reversal from extremes (Overbought >={thresholds.get('rsi_overbought', 70)} or Oversold <={thresholds.get('rsi_oversold', 30)}), MACD Histogram Divergence, Bollinger Band bounces (±{thresholds.get('bb_band_bounce_percent', 0.8)}% from bands).
    - Exit/Take Profit: Bollinger Band Middle Band (EMA20), opposite Bollinger Band, ATR Take Profit Multiplier: {thresholds.get('atr_take_profit_multiplier', 2.0)}x ATR from entry.
    - Stop Loss: ATR Stop Loss Multiplier: {thresholds.get('atr_stop_loss_multiplier', 1.5)}x ATR from entry, beyond Bollinger Band extremes.

**Timeframe Agreement for Trade Confirmation:**

- **Primary Timeframe Dominance:** Focus on signals from the primary timeframe ({primary_tf}).
- **Confirmation from Multiple Timeframes:**  Look for confluence. Ideally, at least 3 out of 5 timeframes should support the signal from the primary timeframe for higher probability trades.

**Precise TP/SL and Limit Order Price Setting:**

- **Take Profit (TP):**
    - Dynamic TP: Use ATR Take Profit Multiplier (Regime-Adjusted) * ATR from entry price.
    - Resistance/Support Levels: Identify key resistance in Bull/Sideways regimes, support in Bear/Sideways regimes from higher timeframes.
    - Bollinger Middle Band (Sideways): For mean reversion in sideways markets.

- **Stop Loss (SL):**
    - Dynamic SL: Use ATR Stop Loss Multiplier (Regime-Adjusted) * ATR from entry price.
    - Support/Resistance Levels: Identify key support in Bull/Sideways regimes, resistance in Bear/Sideways regimes from lower timeframes, slightly beyond for stop placement.
    - EMA Levels: Below EMA50 in Bull trends, above EMA50 in Bear trends.

- **Limit Order Price:**
    - Slightly better than current price for quicker fills, but avoid being too aggressive.
    - For Long: Place limit order slightly below current price, near support levels or EMA levels if entering on pullback.
    - For Short: Place limit order slightly above current price, near resistance levels or EMA levels if entering on bounce.
    - Monitor Order Book: Check bid/ask spread and liquidity when setting limit orders.


**Task:**

Based on the comprehensive market analysis and guidelines above, decide the best course of action: **GO LONG, GO SHORT, HOLD LONG, HOLD SHORT, or NO TRADE.**

If GO LONG or GO SHORT, also determine:
- **Recommended Leverage:** (e.g., 3x, 5x - adjust based on volatility from ATR)
- **Trade Term:** (Intraday/6h, 1d, 1w - based on primary timeframe and market regime)
- **Take Profit Price:** (Based on guidelines above)
- **Stop Loss Price:** (Based on guidelines above)
- **Limit Order Price:** (Based on guidelines above)
- **Rationale:** (Briefly explain the decision based on indicators, market regime, and timeframe analysis.)

**Output Format (Comma-Separated):**

Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Rationale

**Example Output:**

* **For Trade Actions:** GO LONG, 5x, 6h, 11400, 10800, 11300, Bullish trend confirmed across timeframes, RSI oversold on 15m, EMA50 support holding, R/R > 2:1.
* **For No Trade:** NO TRADE, N/A, N/A, N/A, N/A, N/A, Market indecision due to conflicting signals and unclear liquidation levels. Waiting for stronger confirmation.

"""
    return prompt_text


def generate_trading_decision(wallet_balance, position_info, extended_data,
                              onchain_data, multi_tf_data, market_regime, thresholds,
                              heatmap_analysis, econ_summary, primary_tf, current_session):
    """
    Gemini Pro 모델을 통해 프롬프트를 전달하고, 거래 결정을 받아온다.
    """
    prompt = generate_gemini_prompt(wallet_balance, position_info, extended_data,
                                    onchain_data, multi_tf_data, market_regime, thresholds,
                                    heatmap_analysis, econ_summary, primary_tf, current_session)

    logging.info("------- Gemini Prompt -------")
    logging.info(prompt)
    logging.info("------- End Prompt -------")

    sys_instruct = "You are a world-class cryptocurrency trader specializing in BTC/USDT."
    response = gemini_client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05",
        config=types.GenerateContentConfig(system_instruction=sys_instruct),
        contents=[prompt]
    )
    return response.text


def parse_trading_decision(response_text):
    """
    Gemini 응답 텍스트를 파싱하여 거래 결정 dict 형태로 반환.
    정규 표현식(regex)을 사용하여 더 robust하게 파싱.
    **Always returns a dictionary, even if parsing fails.**
    """
    decision = {  # Initialize decision dictionary with default values
        "final_action": "NO TRADE",
        "leverage": "1x",
        "trade_term": "N/A",
        "tp_price": "N/A",
        "sl_price": "N/A",
        "limit_order_price": "N/A",
        "rationale": "N/A"
    }

    if not response_text:  # Handle empty or None response_text explicitly
        logging.warning("parse_trading_decision received empty response_text. Returning default NO TRADE decision.")
        return decision  # Return default decision dict if response is empty

    try:
        lines = response_text.strip().split('\n')
        first_line = lines[0] if lines else ""

        action_match = re.search(r"(GO LONG|GO SHORT|HOLD LONG|HOLD SHORT|NO TRADE)", first_line, re.IGNORECASE)
        if action_match:
            decision["final_action"] = action_match.group(1).upper()

        parts = response_text.split(',')
        if len(parts) >= 6:  # 쉼표로 분리된 값들 파싱 (예외 처리 강화)
            try:
                decision["leverage"] = parts[1].strip().replace("x", "").strip()
                decision["trade_term"] = parts[2].strip()
                decision["tp_price"] = parts[3].strip()
                decision["sl_price"] = parts[4].strip()
                decision["limit_order_price"] = parts[5].strip()
                decision["rationale"] = ", ".join([p.strip() for p in parts[6:]]) if len(
                    parts) > 6 else "N/A"  # Rationale
            except Exception as parse_err:
                logging.error(f"Error parsing decision details: {parse_err}")
                decision["rationale"] = response_text  # Raw response 전체를 rationale로 저장
        else:
            decision["rationale"] = response_text  # 쉼표 분리 실패 시, raw response 전체를 rationale로

    except Exception as e:
        logging.error(f"Error parsing Gemini response: {e}")
        decision["rationale"] = response_text  # parsing error시 raw response 전체를 rationale로

    logging.info("Parsed Trading Decision:")
    logging.info(decision)
    return decision  # Always return the decision dictionary


# =====================================================
# 9. 포지션 로깅 함수 (변동 없음)
# =====================================================
def log_decision(decision, symbol):
    """
    거래 결정 내용을 CSV 파일에 기록한다.
    """
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
    """
    신규 포지션의 오픈 내역을 CSV 파일에 기록한다.
    """
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
    """
    청산된 포지션의 내역 및 수익을 CSV 파일에 기록한다.
    """
    profit = (exit_price - entry_price) if trade_side.upper() == "LONG" else (entry_price - exit_price)
    file_exists = os.path.isfile(CLOSED_POSITIONS_FILE)
    with open(CLOSED_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "trade_side", "entry_price", "exit_price", "profit"])
        writer.writerow([datetime.utcnow(), symbol, trade_side, entry_price, exit_price, profit])
    logging.info(f"{symbol} closed position logged (profit: {profit}).")


# =====================================================
# 10. 포지션 관리 및 메인 트레이딩 로직
# =====================================================
def compute_risk_reward(decision, entry_price, atr_value, thresholds):
    """
    ATR 기반 Stop Loss와 Take Profit 사용하여 Risk/Reward Ratio 계산.
    """
    try:
        atr_multiplier_sl = thresholds.get('atr_stop_loss_multiplier', 1.5)  # default 1.5x ATR
        atr_multiplier_tp = thresholds.get('atr_take_profit_multiplier', 2.0)  # default 2.0x ATR

        stop_loss_atr = atr_multiplier_sl * atr_value
        take_profit_atr = atr_multiplier_tp * atr_value

        if decision["final_action"].upper() == "GO LONG":
            sl_price = entry_price - stop_loss_atr
            tp_price = entry_price + take_profit_atr
            reward = tp_price - entry_price
            risk = entry_price - sl_price

        elif decision["final_action"].upper() == "GO SHORT":
            sl_price = entry_price + stop_loss_atr
            tp_price = entry_price - take_profit_atr
            reward = entry_price - tp_price
            risk = sl_price - entry_price
        else:
            return None, None, None  # GO LONG, GO SHORT 아니면 None 반환

        if risk <= 0:
            return None, None, None

        rr_ratio = reward / risk
        tp_price_str = f"{tp_price:.2f}"  # format to 2 decimal places
        sl_price_str = f"{sl_price:.2f}"  # format to 2 decimal places

        return rr_ratio, tp_price_str, sl_price_str  # TP/SL 가격 string 값으로 반환

    except Exception as e:
        logging.error(f"Error computing risk/reward ratio: {e}")
        return None, None, None


def fetch_additional_data(symbol):
    """
    주문서, 거래소 유입량, funding rate, open interest, Fear & Greed Index, Spot-Future Price Diff, 비트코인 도미넌스 등 추가 데이터 취합.
    """
    base_data = fetch_order_book(symbol)
    exchange_inflows = fetch_exchange_inflows()
    funding_rate = fetch_funding_rate(symbol)
    open_interest = fetch_open_interest(symbol)
    fng = fetch_fear_and_greed_index()
    spot_future_price_diff = fetch_spot_future_price_diff(symbol)  # 선물-현물 가격 차이
    bitcoin_dominance = fetch_bitcoin_dominance()  # 비트코인 도미넌스

    extended_data = {
        "order_book": base_data,  # order_book 전체 데이터 (bid, ask, spread 포함)
        "exchange_inflows": exchange_inflows,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "fear_and_greed_index": fng,
        "spot_future_price_diff": spot_future_price_diff,  # 선물-현물 가격 차이 추가
        "bitcoin_dominance": bitcoin_dominance  # 비트코인 도미넌스 추가
    }
    return extended_data


def get_current_session_kst():
    """
    KST 기준 현재 시간을 확인하여 트레이딩 세션(Overnight, Asian, European, US)을 결정.
    세션 구분 기준 시간 조정 (UTC+9, KST 기준)
    """
    now = datetime.now(KST)
    hour = now.hour

    if 0 <= hour < 8:
        return "OVERNIGHT_SESSION"  # 00:00 ~ 08:00 (저유동성 세션)
    elif 8 <= hour < 16:  # 08:00 - 16:00 Asian Session
        return "ASIAN_SESSION"  # 08:00 ~ 16:00 (아시아 세션)
    elif 16 <= hour < 22:  # 16:00 - 22:00 London Session (including early overlap with US)
        return "LONDON_SESSION"  # 16:00 ~ 22:00 (런던 세션)
    elif 22 <= hour < 24 or 0 <= hour < 6:  # 22:00 - 06:00 US Session (including overlap with London and late US)
        return "US_SESSION"  # 22:00 ~ 06:00 (미국 세션)
    elif 6 <= hour < 8:  # 06:00 - 08:00 Overlap/Transition Session (between US close and Asian open)
        return "TRANSITION_SESSION"  # 06:00 - 08:00 (미국 마감 - 아시아 오픈 준비)
    else:  # 예외 처리
        return "UNDEFINED_SESSION"


def is_news_volatility_period(minutes_before=15, minutes_after=15):  # News 발표 전후 시간 확대
    """
    주요 경제 지표 발표 전후 기간인지 확인하여 변동성 상승 가능성 판단.
    """
    major_news_times = [
        datetime.now(KST).replace(hour=9, minute=30, second=0, microsecond=0),  # 한국시간 기준 주요 뉴스 발표 시간 예시 (조정 필요)
        datetime.now(KST).replace(hour=17, minute=00, second=0, microsecond=0),  # 유럽 주요 경제지표 발표 예상 시간
        datetime.now(KST).replace(hour=22, minute=30, second=0, microsecond=0)  # 미국 주요 경제지표 발표 예상 시간 (ex: 22:30 KST)
    ]
    now_kst = datetime.now(KST)

    for news_time in major_news_times:
        diff_in_minutes = abs((now_kst - news_time).total_seconds()) / 60.0
        if diff_in_minutes <= minutes_before or diff_in_minutes <= minutes_after:
            if diff_in_minutes < (minutes_before + minutes_after):
                return True
    return False


def fetch_economic_data():
    """
    Investing.com 등의 사이트에서 경제 캘린더 데이터를 가져온다.
    """
    url = 'https://www.investing.com/economic-calendar/Service/getCalendarFilteredData'
    headers = {
        'accept': '*/*',
        'accept-language': 'ko,en-US;q=0.9,en;q=0.8,ko-KR;q=0.7',
        'content-type': 'application/x-www-form-urlencoded',
        'origin': 'https://www.investing.com',
        'referer': 'https://www.investing.com/economic-calendar/',
        'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
    }
    data = 'country%5B%5D=5&importance%5B%5D=3&timeZone=8&timeFilter=timeOnly&currentTab=today&submitFilters=1&limit_from=0'

    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching economic data: {e}")
        return None


def parse_economic_data(json_data):
    """
    API response (JSON) 파싱 및 요약 정보 추출. HTML data 파싱으로 변경.
    """
    if not json_data or not json_data['data']:
        return "No significant economic events today."

    html_content = json_data['data']
    soup = BeautifulSoup(html_content, 'html.parser')
    event_rows = soup.find_all('tr', class_='js-event-item')  # 각 이벤트는 'js-event-item' 클래스의 tr 태그

    event_list = []
    now_kst = datetime.now(KST)

    for row in event_rows:
        try:
            time_cell = row.find('td', class_='time js-time')
            currency_cell = row.find('td', class_='left flagCur noWrap')
            impact_cell = row.find('td', class_='left textNum sentiment noWrap')
            event_cell = row.find('td', class_='left event')

            if not time_cell or not currency_cell or not impact_cell or not event_cell:
                logging.warning(f"Incomplete event data in row, skipping row.")
                continue

            event_time_str = time_cell.text.strip()  # 시간 추출
            event_currency = currency_cell.text.strip()  # 통화 추출
            event_impact_element = impact_cell.find('i')  # 중요도 아이콘 엘리먼트 찾기
            event_impact_str = event_impact_element.get('data-img_key', 'low')  # 속성에서 중요도 추출, default 'low'
            event_impact = event_impact_str.replace('bull', '').lower()  # 'bull3' -> '3' -> 'high' (mapping 필요)

            event_name = event_cell.text.strip()  # 이벤트 이름 추출

            # 중요도 텍스트로 매핑
            impact_mapping = {'3': 'high', '2': 'medium', '1': 'low', 'gray': 'low'}  # gray 도 low로 처리
            event_impact = impact_mapping.get(event_impact, 'low')  # 기본값 low

            # 시간 문자열을 datetime 객체로 변환 (KST 기준)
            event_hour, event_minute = map(int, event_time_str.split(':'))
            event_datetime_kst = now_kst.replace(hour=event_hour, minute=event_minute, second=0, microsecond=0)
            time_diff_minutes = (event_datetime_kst - now_kst).total_seconds() / 60

            if abs(time_diff_minutes) <= 60 * 24:  # 오늘 발표 예정 or 이미 발표된 지표만 포함 (24시간 이내)
                event_list.append({
                    'time': event_time_str,
                    'currency': event_currency,
                    'impact': event_impact,
                    'name': event_name,
                    'time_diff': time_diff_minutes
                })


        except (KeyError, ValueError, AttributeError) as e:
            logging.warning(f"Error parsing economic event row: {e}, row data: {row}")
            continue  # 파싱 에러 발생 시, 해당 이벤트(row) 건너뛰고 다음 이벤트 처리

    if not event_list:
        return "No significant economic events today."

    # 중요도 높은 이벤트 먼저, 시간순 정렬 (발표 임박 -> 과거)
    event_list.sort(key=lambda x: (-{'high': 3, 'medium': 2, 'low': 1}.get(x['impact'], 0), x['time_diff']))

    summary_lines = ["**Upcoming Economic Events (KST):**"]  # Markdown bold 적용
    for event in event_list:
        time_str = event['time']
        currency = event['currency']
        impact = event['impact']
        event_name = event['name']
        time_diff = event['time_diff']

        time_display = ""
        if time_diff > 0:
            time_display = f"in {int(time_diff)} min"  # 앞으로 남은 시간 (분)
        else:
            time_display = f"{abs(int(time_diff))} min ago"  # 발표 후 경과 시간 (분)

        summary_lines.append(
            f"- `{time_str}` ({currency}, {impact}): {event_name} ({time_display})")  # Markdown code block for time

    return "\n".join(summary_lines)


def detect_fake_breakout(df, lookback=20, volume_factor=1.5, rsi_threshold=(30, 70), pivot_dist=0.001):
    """
    df: 봉데이터 (기술적 지표 포함)
    가짜 돌파 (Fake Breakout) 여부 판단 알고리즘 (기존 로직과 동일)
    """
    if len(df) < lookback + 1:
        return False

    recent_slice = df.iloc[-lookback:]
    pivot_high = recent_slice['high'].max()
    pivot_low = recent_slice['low'].min()

    last_candle = df.iloc[-1]
    prev_20_vol_mean = df['volume'].iloc[-(lookback + 1):-1].mean()
    last_vol = last_candle['volume']
    last_rsi = last_candle['rsi']

    # 돌파 조건
    breakout_up = (last_candle['close'] >= pivot_high * (1.0 - pivot_dist))
    breakout_down = (last_candle['close'] <= pivot_low * (1.0 + pivot_dist))

    is_breakout = breakout_up or breakout_down
    if not is_breakout:
        return False

    # 체크 요소
    volume_check = (last_vol < prev_20_vol_mean * volume_factor)  # 거래량 부족
    rsi_check = ((breakout_up and last_rsi < rsi_threshold[1] - 5) or  # 위로 돌파인데 RSI 낮음
                 (breakout_down and last_rsi > rsi_threshold[0] + 5))  # 아래로 돌파인데 RSI 높음
    # 되돌림 체크: 종가-고가/저가 비율
    candle_range = last_candle['high'] - last_candle['low']
    if candle_range == 0:
        retrace_ratio = 0
    else:
        if breakout_up:
            retrace_ratio = (last_candle['high'] - last_candle['close']) / candle_range
        else:
            retrace_ratio = (last_candle['close'] - last_candle['low']) / candle_range
    retrace_check = (retrace_ratio > 0.6)  # 되돌림 비율

    # 가중치 합산 (2/3 이상 조건 만족 시 가짜 돌파)
    suspicion_score = 0
    if volume_check:
        suspicion_score += 1
    if rsi_check:
        suspicion_score += 1
    if retrace_check:
        suspicion_score += 1

    if suspicion_score >= 2:
        return True
    return False


def analyze_session_open_volatility(df, open_hour=8, open_minute=0, threshold_factor=1.5, window=5):
    """
    세션 시작 시점 (예: 아시아 세션 오픈 8시, 유럽 세션 16시)  변동성 분석 (기존 로직과 동일)
    """
    now_kst = datetime.now(KST)
    open_dt = now_kst.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)

    diff_minutes = (now_kst - open_dt).total_seconds() / 60.0
    if abs(diff_minutes) > 30:
        return False  # 세션 오픈 30분 이후 pass

    recent_time = open_dt + timedelta(minutes=window)
    subdf = df[(df['timestamp'] >= open_dt) & (df['timestamp'] <= recent_time)]
    if len(subdf) < 1:
        return False

    price_range = subdf['high'].max() - subdf['low'].min()
    last_atr = df['atr'].iloc[-20:].mean() if len(df) >= 20 else df['atr'].mean()

    if last_atr and last_atr > 0:
        if price_range / last_atr > threshold_factor:
            return True
    return False


def get_timeframe_agreement(multi_tf_data, market_regime):
    """
    Multi-Timeframe Agreement 기반으로 방향성 합의 점수 및 Confidence Level 계산.
    더 정교한 합의 알고리즘 적용 (장세, Primary TF 고려)
    """
    primary_tf = choose_primary_timeframe(market_regime)  # Primary TF 재선정 (Regime 반영)
    if primary_tf not in multi_tf_data:
        return {"agreement_score": 0, "confidence_level": "Low", "primary_tf": primary_tf,
                "details": "Primary timeframe data missing"}

    primary_tf_data = multi_tf_data[primary_tf]
    primary_signal = None  # Primary TF signal 초기화

    # Primary TF Signal - Regime 따라 기준 변경 (예: Bull 추세 시 EMA50 지지, Sideways 시 RSI Reversal 등)
    if "bull_trend" in market_regime:  # Bull 추세: EMA50 지지 & RSI 조건
        if primary_tf_data['current_price'] > primary_tf_data['ema50'] and primary_tf_data['rsi'] < 70:  # RSI 과매수 방지
            primary_signal = "bullish"
        else:
            primary_signal = "neutral"  # No clear signal
    elif "bear_trend" in market_regime:  # Bear 추세: EMA50 저항 & RSI 조건
        if primary_tf_data['current_price'] < primary_tf_data['ema50'] and primary_tf_data['rsi'] > 30:  # RSI 과매도 방지
            primary_signal = "bearish"
        else:
            primary_signal = "neutral"  # No clear signal
    elif "sideways" in market_regime:  # Sideways: RSI 과매수/과매도 Reversal
        if primary_tf_data['rsi'] < 35:  # Oversold
            primary_signal = "bullish_reversal"
        elif primary_tf_data['rsi'] > 65:  # Overbought
            primary_signal = "bearish_reversal"
        else:
            primary_signal = "neutral"  # No clear signal
    else:
        primary_signal = "neutral"  # Default signal

    # Secondary TF Agreement (가중치 적용)
    agreement_score = 0
    details = {}  # 상세 분석 결과 저장
    timeframes_to_check = [tf for tf in TIMEFRAMES if tf != primary_tf and tf in multi_tf_data]  # Primary TF 제외

    for tf in timeframes_to_check:
        tf_data = multi_tf_data[tf]
        tf_signal = "neutral"  # Default signal for secondary timeframes

        if "bull_trend" in market_regime:  # Bull 추세: EMA50 & EMA200 지지 여부
            if tf_data['current_price'] > tf_data['ema50'] and tf_data['current_price'] > tf_data['ema200']:
                tf_signal = "bullish"
            elif tf_data['current_price'] < tf_data['ema50'] and tf_data['current_price'] < tf_data['ema200']:
                tf_signal = "bearish"  # 반대 방향 신호도 체크 (weakness)
        elif "bear_trend" in market_regime:  # Bear 추세: EMA50 & EMA200 저항 여부
            if tf_data['current_price'] < tf_data['ema50'] and tf_data['current_price'] < tf_data['ema200']:
                tf_signal = "bearish"
            elif tf_data['current_price'] > tf_data['ema50'] and tf_data['current_price'] > tf_data['ema200']:
                tf_signal = "bullish"  # 반대 방향 신호도 체크 (weakness)
        elif "sideways" in market_regime:  # Sideways: RSI 기준 (Overbought/Oversold)
            if tf_data['rsi'] < 40:  # Oversold 기준 완화
                tf_signal = "bullish_reversal"
            elif tf_data['rsi'] > 60:  # Overbought 기준 완화
                tf_signal = "bearish_reversal"

        details[tf] = tf_signal  # TF별 시그널 기록

        if tf_signal == primary_signal:  # Primary TF 방향과 Secondary TF 방향 일치 시 가중치 부여
            agreement_score += 1
        elif tf_signal != "neutral" and primary_signal != "neutral" and tf_signal != primary_signal:
            agreement_score -= 0.5  # 반대 방향 시 감점 (optional)

    # Confidence Level (점수 -> 신뢰도 레벨)
    if agreement_score >= 3:  # Threshold 조정 (ex: 3/4 or 4/5)
        confidence_level = "High"
    elif agreement_score >= 1.5:  # Moderate 기준 완화
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    return {
        "agreement_score": agreement_score,
        "confidence_level": confidence_level,
        "primary_tf_signal": primary_signal,
        "primary_tf": primary_tf,
        "details": details  # TF별 상세 시그널 정보 포함
    }


def main():
    logging.info("Trading bot started.")
    wallet_balance = "1000 USDT"
    position_info = "NONE"
    in_position = False
    current_side = None
    entry_price = 0.0

    # 경제 지표 파싱 및 요약
    econ_data_raw = fetch_economic_data()
    econ_summary = parse_economic_data(econ_data_raw)

    # 멀티 타임프레임 데이터 및 지표 계산
    mtf = fetch_multi_tf_data(SYMBOL, TIMEFRAMES, limit=300)
    if not mtf or "1h" not in mtf:
        logging.error("Not enough TF data.")
        return

    cprice = mtf["1h"]["current_price"]
    ext_data = fetch_additional_data(SYMBOL)
    onchain_data = fetch_onchain_data(SYMBOL)

    # 시장 Regime 판단 및 Thresholds 설정
    regime = determine_market_regime(mtf, onchain_data)
    thresholds = adjust_indicator_thresholds(regime)
    primary_tf = choose_primary_timeframe(regime)  # Regime 기반 Primary TF 선정

    # 타임프레임 합의 점수 계산
    tf_agreement = get_timeframe_agreement(mtf, regime)
    agreement_score = tf_agreement["agreement_score"]
    confidence_level = tf_agreement["confidence_level"]
    primary_tf_signal = tf_agreement["primary_tf_signal"]
    tf_agreement_details = tf_agreement["details"]

    current_session = get_current_session_kst()
    logging.info(f"Current session: {current_session}")
    logging.info(
        f"Market Regime: {regime.upper()}, Primary TF: {primary_tf}, TF Agreement Score: {agreement_score}, Confidence: {confidence_level}")
    logging.info(f"Timeframe Agreement Details: {tf_agreement_details}")

    # 가짜 돌파 및 세션 변동성 감지 (Log only, 필요시 전략에 활용)
    if "5m" in mtf:
        df_5m = mtf["5m"]["df_full"]
        if detect_fake_breakout(df_5m):
            logging.info("Fake breakout suspicion on 5m TF. Proceed with caution.")
        if analyze_session_open_volatility(df_5m, 8, 0, threshold_factor=1.5, window=6):
            logging.info("High volatility detected at Asian session open.")
        if analyze_session_open_volatility(df_5m, 16, 0, threshold_factor=1.5, window=6):
            logging.info("High volatility detected at London session open.")

    # Final Trade Decision by Gemini Pro
    try:
        fetch_liquidation_heatmap()
        heatmap_analysis = analyze_liquidation_heatmap_gemini(
            "/Users/changpt/Downloads/Liquidation Heat Map.png")  # Gemini Pro Vision API 활용
        gemini_resp_text = generate_trading_decision(
            wallet_balance=wallet_balance,
            position_info=position_info,
            extended_data=ext_data,
            onchain_data=onchain_data,
            multi_tf_data=mtf,
            market_regime=regime,
            thresholds=thresholds,
            heatmap_analysis=heatmap_analysis,
            econ_summary=econ_summary,
            primary_tf=primary_tf,
            current_session=current_session
        )
        logging.info(f"Gemini Raw Response: {gemini_resp_text}")
        decision = parse_trading_decision(gemini_resp_text)  # Gemini response parsing
        log_decision(decision, SYMBOL)  # Trading decision logging

    except Exception as e:
        logging.error(f"Error in Gemini Pro interaction or decision parsing: {e}")
        return

    # Risk-Reward Ratio 계산 및 TP/SL 가격 설정 (ATR 기반)
    atr_value_1h = mtf["1h"]["atr"]  # 1시간봉 ATR 활용 (or Primary TF ATR)
    rr_ratio, tp_price_str, sl_price_str = compute_risk_reward(decision, cprice, atr_value_1h,
                                                               thresholds)  # thresholds 전달

    if rr_ratio is not None:  # compute_risk_reward 에서 반환값 None 체크
        decision["rr_ratio"] = f"{rr_ratio:.2f}"  # Risk-Reward Ratio decision 딕셔너리에 추가
        decision["tp_price"] = tp_price_str  # TP, SL 가격 decision 딕셔너리에 업데이트 (string)
        decision["sl_price"] = sl_price_str
    else:  # rr_ratio가 None이면 (계산 실패)
        logging.error("Failed to compute risk/reward. Skipping trade.")
        return  # 거래 중단

    rr_text = decision.get("rr_ratio", "N/A")  # decision 딕셔너리에서 R/R ratio text 가져오기

    # 텔레그램 메시지 전송 (더 상세하고, Markdown 포맷 적용)
    if decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
        side = "Buy" if decision["final_action"].upper() == "GO LONG" else "Sell"
        message = (
            f"*Trading Signal: {side} {SYMBOL}*\n\n"  # Bold 적용, Symbol 추가
            f"- **Market Regime:** {regime.upper()}\n"  # 장세 정보 추가
            f"- **Primary Timeframe:** {primary_tf}\n"  # Primary TF 정보 추가
            f"- **Confidence Level:** {confidence_level} ({agreement_score} agreement)\n"  # Confidence Level 추가
            f"- **R/R Ratio:** {rr_text}\n"
            f"- **Leverage:** {decision['leverage']}\n"
            f"- **Trade Term:** {decision['trade_term']}\n"
            f"- **Limit Order Price:** {decision['limit_order_price']}\n"
            f"- **Take Profit:** {decision['tp_price']}\n"
            f"- **Stop Loss:** {decision['sl_price']}\n\n"
            f"**Rationale:** {decision['rationale']}"  # Bold 적용
        )
        send_telegram_message(message)  # 텔레그램 메시지 전송 함수 호출

    if not in_position:
        if rr_ratio and rr_ratio >= 2 and decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
            logging.info(
                f"Opening {decision['final_action']} position @ {cprice}, TP={decision['tp_price']}, SL={decision['sl_price']}")
            log_open_position(SYMBOL, decision, cprice)
            in_position = True
            current_side = decision["final_action"].split()[-1]
            entry_price = cprice
        else:
            logging.info(f"No new position. R/R Ratio: {rr_text}, Action: {decision['final_action']}")
    else:
        if decision["final_action"].upper() not in ["HOLD LONG", "HOLD SHORT"]:
            logging.info(f"Exiting {current_side} position @ {cprice}")
            log_closed_position(SYMBOL, entry_price, cprice, current_side)
            in_position = False
        else:
            logging.info(f"Holding current {current_side} position.")

    logging.info("Trading bot cycle completed.\n" + "=" * 50)  # Cycle Log 구분선


if __name__ == "__main__":
    main()
