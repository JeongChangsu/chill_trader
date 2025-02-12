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
import undetected_chromedriver as uc

from PIL import Image
from google import genai
from openai import OpenAI
from bs4 import BeautifulSoup
from google.genai import types
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By

# =====================================================
# 1. 기본 설정 및 글로벌 변수 (v2.0)
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ===========================================
# Global constants and API initialization (v2.0)
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
CANDLE_PATTERNS = [
    "ENGULFING", "MORNING_STAR", "EVENING_STAR", "HAMMER", "HANGING_MAN",
    "SHOOTING_STAR", "DARK_CLOUD_COVER", "PIERCING_LINE", "DOJI", "MARUBOZU"
]

# OPENAI_API_KEY must be set in the environment
openai_api_key = os.environ.get('OPENAI_API_KEY')
gpt_client = OpenAI(api_key=openai_api_key)

# GOOGLE_API_KEY must be set in the environment
google_api_key = os.environ.get('GOOGLE_API_KEY')
gemini_client = genai.Client(api_key=google_api_key)

# Telegram variables (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# [새로 추가] KST 타임존 설정 (v2.0)
KST = pytz.timezone("Asia/Seoul")


# =====================================================
# 2. 텔레그램 메시지 전송 함수 (v2.0)
# =====================================================
def send_telegram_message(message):
    """Telegram API를 사용하여 메시지를 전송한다."""
    if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        logging.error("Telegram bot token or chat ID is not set.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"  # Markdown 활성화 (v2.0)
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
# 3. Persistent Driver Setup (Data Crawling) (v2.0)
# =====================================================
drivers = {}


def create_driver():
    """undetected_chromedriver의 새로운 인스턴스를 생성하여 반환한다."""
    options = uc.ChromeOptions()
    # 필요에 따라 옵션 설정 (예: headless 모드)
    # options.add_argument('--headless')  # 필요시 headless 모드 활성화
    driver = uc.Chrome(options=options)
    return driver


def get_driver(session_id='default_session'):
    """주어진 session_id에 해당하는 드라이버가 있으면 반환, 없으면 새로 생성."""
    global drivers
    if session_id in drivers and drivers[session_id] is not None:
        return drivers[session_id]
    else:
        driver = create_driver()
        drivers[session_id] = driver
        return driver


# =====================================================
# 4. 데이터 수집 및 기술적 지표 계산 (v2.0)
# =====================================================
def fetch_ohlcv(symbol, timeframe, limit=300):
    """Binance ccxt 라이브러리 이용하여 OHLCV 데이터 fetch 및 DataFrame 반환."""
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
    """DataFrame에 다양한 기술적 지표 계산 및 추가 (v2.0)."""
    # Volume 지표 (v2.0)
    df['volume_roc'] = df['volume'].pct_change() * 100  # 거래량 변화율 (%)
    df['volume_spike'] = df['volume'].rolling(window=20).mean() * 1.5 < df['volume']  # 거래량 급증 여부 (20MA * 1.5 기준)

    # Momentum 지표 (v2.0)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci() # CCI 추가 (v2.0)

    # Moving Average (MA) 지표 (v2.0)
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator() # EMA 추가 (v2.0)
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator() # EMA 추가 (v2.0)
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=20).vwap # VWAP 추가 (v2.0)

    # MACD (Moving Average Convergence Divergence) (v2.0)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff() # MACD Histogram 추가 (v2.0)

    # Volatility 지표
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower'] # Bollinger Bands width 추가 (v2.0)

    # 가격-MA Diff (v2.0)
    df['ma20_diff'] = (df['close'] - df['ema20']) / df['ema20'] * 100 # SMA -> EMA (v2.0)
    df['ma50_diff'] = (df['close'] - df['ema50']) / df['ema50'] * 100 # SMA -> EMA (v2.0)
    df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap'] * 100 # VWAP Diff 추가 (v2.0)

    logging.info("Technical indicators calculated (v2.0)")
    return df


def fetch_order_book(symbol):
    """Binance에서 주문서 데이터 fetch, 최상위 bid, ask, spread 값 반환."""
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


# =====================================================
# 5. 확장 데이터 수집 (크롤링, 온체인 데이터 등) (v2.0)
# =====================================================
def fetch_exchange_inflows():
    """CryptoQuant 크롤링하여 거래소 순입출금 데이터 반환."""
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
    """Binance Futures 데이터 이용하여 funding rate fetch."""
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
    """Binance Futures 데이터 이용하여 open interest 데이터 fetch."""
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


def fetch_open_interest_change(symbol, period="1h"): # 미결제 약정 변화 추가 (v2.0)
    """Binance Futures 통해 미결제 약정 변화 데이터 fetch."""
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        symbol_futures = symbol.replace("/", "")
        params = {'period': period} # 1시간 봉 기준 (v2.0)
        oi_response = futures_exchange.fetch_open_interest_history(symbol_futures, params=params, limit=2) # 최근 2개 데이터 (v2.0)
        if oi_response and len(oi_response) >= 2: # 데이터 존재 여부 확인 (v2.0)
            current_oi = oi_response[-1]['openInterestAmount']
            previous_oi = oi_response[-2]['openInterestAmount']
            oi_change_percent = (current_oi - previous_oi) / previous_oi * 100 if previous_oi != 0 else 0
            logging.info(f"{symbol} open interest change ({period}) fetched successfully")
            return oi_change_percent
        else:
            return "N/A"
    except Exception as e:
        logging.error(f"Error fetching open interest change for {symbol} ({period}): {e}")
        return "N/A"


def fetch_long_short_ratio():
    """
    CoinAnk 웹사이트에서 롱/숏 비율 데이터를 크롤링한다.
    """
    url = "https://coinank.com/longshort/realtime"
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(3)

        all_long_short = driver.find_elements(By.XPATH, '//div[@class="longshort-row"]')[
            1]  # 인덱스 1로 수정 (두 번째 row, BTC 롱숏 비율)
        long_ratio_text = all_long_short.find_element(By.XPATH,
                                                      './/div[@class="progress-value"]//div[contains(@class, "long-text")]').text
        short_ratio_text = all_long_short.find_element(By.XPATH,
                                                       './/div[@class="progress-value"]//div[contains(@class, "short-text")]').text

        # 퍼센트(%) 문자 제거 및 float 변환
        long_ratio = float(long_ratio_text.replace("%", "").strip()) / 100.0  # percentage -> ratio (0~1.0) 값으로 변환
        short_ratio = float(short_ratio_text.replace("%", "").strip()) / 100.0  # percentage -> ratio (0~1.0) 값으로 변환

        logging.info(
            f"CoinAnk Long/Short Ratio crawled successfully: Long Ratio: {long_ratio:.2f}, Short Ratio: {short_ratio:.2f}")
        return long_ratio

    except Exception as e:
        logging.error(f"Error crawling CoinAnk Long/Short Ratio: {e}")
        return "N/A"


def fetch_btc_dominance():
    """
    TradingView에서 BTC Dominance 데이터를 크롤링한다.
    """
    try:
        driver = get_driver()
        driver.get('https://kr.tradingview.com/chart/?symbol=CRYPTOCAP%3ABTC.D')
        time.sleep(3)
        dominance_text = driver.find_element(By.XPATH, '//span[contains(@class, "priceWrapper")]/span').text
        dominance = float(dominance_text.replace("%", "").replace(",", ""))  # % 문자, 쉼표 제거 후 float 변환
        logging.info(f"BTC dominance fetched successfully: {dominance:.2f}%")
        return dominance
    except Exception as e:
        logging.error(f"Error fetching BTC dominance: {e}")
        return "N/A"


def fetch_future_premium_rate(symbol): # 선물 vs 현물 가격 차이 추가 (v2.0)
    """Binance Futures 및 현물 데이터 이용하여 선물 프리미엄 비율 fetch."""
    try:
        exchange = ccxt.binance()
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})

        spot_ticker = exchange.fetch_ticker(symbol)
        future_ticker = futures_exchange.fetch_ticker(symbol)

        spot_price = spot_ticker['close']
        future_price = future_ticker['close']
        premium_rate = ((future_price - spot_price) / spot_price) * 100 if spot_price else None
        logging.info(f"{symbol} future premium rate fetched successfully")
        return premium_rate
    except Exception as e:
        logging.error(f"Error fetching future premium rate for {symbol}: {e}")
        return "N/A"


def fetch_fear_and_greed_index():
    """Alternative.me API 사용하여 Fear & Greed Index fetch."""
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
    """온체인 지표(MVRV, SOPR)를 TradingView에서 크롤링하여 반환."""
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
    """여러 타임프레임 OHLCV 데이터 fetch, 기술적 지표 계산, 요약 정보 반환."""
    if timeframes is None:
        timeframes = ["5m", "15m", "1h", "4h", "1d"]
    multi_tf_data = {}
    for tf in timeframes:
        df = fetch_ohlcv(symbol, tf, limit)
        if df is None:
            continue
        df = compute_technical_indicators(df)
        candle_pattern = detect_candle_patterns(df) # 캔들 패턴 분석 추가 (v2.0)
        latest = df.iloc[-1]
        multi_tf_data[tf] = {
            "current_price": round(latest['close'], 2),
            "volume_roc": round(latest['volume_roc'], 2) if not np.isnan(latest['volume_roc']) else None, # 거래량 변화율 (v2.0)
            "volume_spike": latest['volume_spike'], # 거래량 급증 (v2.0)
            "rsi": round(latest['rsi'], 2) if not np.isnan(latest['rsi']) else None,
            "cci": round(latest['cci'], 2) if not np.isnan(latest['cci']) else None, # CCI (v2.0)
            "ema20": round(latest['ema20'], 2) if not np.isnan(latest['ema20']) else None, # EMA (v2.0)
            "ema50": round(latest['ema50'], 2) if not np.isnan(latest['ema50']) else None, # EMA (v2.0)
            "vwap": round(latest['vwap'], 2) if not np.isnan(latest['vwap']) else None, # VWAP (v2.0)
            "ma20_diff": round(latest['ma20_diff'], 2) if not np.isnan(latest['ma20_diff']) else None, # MA Diff (v2.0)
            "ma50_diff": round(latest['ma50_diff'], 2) if not np.isnan(latest['ma50_diff']) else None, # MA Diff (v2.0)
            "vwap_diff": round(latest['vwap_diff'], 2) if not np.isnan(latest['vwap_diff']) else None, # VWAP Diff (v2.0)
            "bb_upper": round(latest['bb_upper'], 2) if not np.isnan(latest['bb_upper']) else None,
            "bb_lower": round(latest['bb_lower'], 2) if not np.isnan(latest['bb_lower']) else None,
            "bb_width": round(latest['bb_width'], 2) if not np.isnan(latest['bb_width']) else None, # BB Width (v2.0)
            "macd": round(latest['macd'], 2) if not np.isnan(latest['macd']) else None,
            "macd_signal": round(latest['macd_signal'], 2) if not np.isnan(latest['macd_signal']) else None,
            "macd_hist": round(latest['macd_hist'], 2) if not np.isnan(latest['macd_hist']) else None, # MACD Histogram (v2.0)
            "atr": round(latest['atr'], 2) if not np.isnan(latest['atr']) else None,
            "timestamp": latest['timestamp'],
            "candle_pattern": candle_pattern, # 캔들 패턴 (v2.0)
            "df_full": df
        }
    logging.info("Multi-timeframe data and indicators calculated (v2.0)")
    return multi_tf_data


# =====================================================
# 6. 캔들 패턴 분석 (v2.0)
# =====================================================
def detect_candle_patterns(df):
    """캔들 패턴 감지 및 패턴명 반환 (v2.0)."""
    patterns = {}

    # Engulfing Pattern
    engulfing = ta.momentum.CDLEngulfingIndicator(df['open'], df['high'], df['low'], df['close'])  # ta.trend -> ta.momentum
    if engulfing.cdl_engulfing().iloc[-1]:
        patterns['ENGULFING'] = 'Bullish' if df['open'].iloc[-1] > df['close'].iloc[-1] else 'Bearish'

    # Morning/Evening Star
    morning_star = ta.momentum.CDLMorningStar(df['open'], df['high'], df['low'], df['close']) # ta.trend -> ta.momentum
    if morning_star.cdl_morning_star().iloc[-1]:
        patterns['MORNING_STAR'] = 'Bullish'
    evening_star = ta.momentum.CDLEveningStar(df['open'], df['high'], df['low'], df['close']) # ta.trend -> ta.momentum
    if evening_star.cdl_evening_star().iloc[-1]:
        patterns['EVENING_STAR'] = 'Bearish'

    # Hammer/Hanging Man
    hammer = ta.momentum.CDLHammer(df['open'], df['high'], df['low'], df['close']) # ta.trend -> ta.momentum
    if hammer.cdl_hammer().iloc[-1]:
        patterns['HAMMER'] = 'Bullish'
    hanging_man = ta.momentum.CDLHangingMan(df['open'], df['high'], df['low'], df['close']) # ta.trend -> ta.momentum
    if hanging_man.cdl_hanging_man().iloc[-1]:
        patterns['HANGING_MAN'] = 'Bearish'

    # Shooting Star/Dark Cloud Cover/Piercing Line
    shooting_star = ta.momentum.CDLShootingStar(df['open'], df['high'], df['low'], df['close']) # ta.trend -> ta.momentum
    if shooting_star.cdl_shooting_star().iloc[-1]:
        patterns['SHOOTING_STAR'] = 'Bearish'
    dark_cloud_cover = ta.momentum.CDLDarkCloudCover(df['open'], df['high'], df['low'], df['close']) # ta.trend -> ta.momentum
    if dark_cloud_cover.cdl_dark_cloud_cover().iloc[-1]:
        patterns['DARK_CLOUD_COVER'] = 'Bearish'
    piercing_line = ta.momentum.CDLPiercingLine(df['open'], df['high'], df['low'], df['close'])  # ta.trend -> ta.momentum
    if piercing_line.cdl_piercing().iloc[-1]:
        patterns['PIERCING_LINE'] = 'Bullish'

    # Doji/Marubozu
    doji = ta.momentum.CDLDoji(df['open'], df['high'], df['low'], df['close']) # ta.trend -> ta.momentum
    if doji.cdl_doji().iloc[-1]:
        patterns['DOJI'] = 'Neutral'
    marubozu = ta.momentum.CDLMarubozu(df['open'], df['high'], df['low'], df['close']) # ta.trend -> ta.momentum
    if marubozu.cdl_marubozu().iloc[-1]:
        patterns['MARUBOZU'] = 'Bullish' if df['open'].iloc[-1] < df['close'].iloc[-1] else 'Bearish'

    detected_patterns = ", ".join([f"{pattern}({direction})" for pattern, direction in patterns.items()]) if patterns else "No Pattern" # 패턴 결과 문자열 (v2.0)
    logging.info(f"Detected candle patterns: {detected_patterns}")
    return detected_patterns


# =====================================================
# 7. 청산 히트맵 데이터 및 분석 (v2.0)
# =====================================================
def fetch_liquidation_heatmap():
    """CoinAnk 사이트에서 청산 히트맵 데이터 다운로드."""
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


def analyze_liquidation_heatmap():
    """다운로드된 청산 히트맵 이미지 Gemini 통해 분석, 지정 포맷 결과 반환."""
    image_path = "/Users/changpt/Downloads/Liquidation Heat Map.png"
    image = Image.open(image_path)

    try:
        sys_instruct = "You are a specialized analyst in crypto liquidations."
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=sys_instruct),
            contents=["Please analyze the attached liquidation heatmap image for BTC futures. "
                      "Identify the key liquidation zones, explain their potential impact on future price movements, "
                      "and indicate which side (longs or shorts) is at higher risk. "
                      "Output your analysis in a single line using the following format: "
                      "\"Long Liquidation Zone: <value>; Short Liquidation Zone: <value>; Impact: <analysis text>; Risk: <which side is at higher risk>.\"",
                      image])
        analysis_result = response.text
    except Exception as e:
        logging.error("Error during GPT analysis of liquidation heatmap: " + str(e))
        analysis_result = "N/A"

    # Delete image file after processing
    try:
        os.remove(image_path)
        logging.info("Deleted liquidation heatmap image file after processing.")
    except Exception as e:
        logging.error("Error deleting image file: " + str(e))

    logging.info("Liquidation heatmap analysis result:")
    logging.info(analysis_result)
    return analysis_result


# =====================================================
# 8. 시장 상태 결정 및 지표 임계값 조정 (v2.0)
# =====================================================
def is_high_volatility(multi_tf_data):
    """1시간봉 ATR/현재가 비율 기준으로 고변동성 여부 판단."""
    data = multi_tf_data.get("1h")
    if data and data.get("atr") is not None and data.get("current_price") is not None:
        atr_ratio = data["atr"] / data["current_price"]
        return atr_ratio > 0.02
    return False


def determine_market_regime(multi_tf_data, onchain_data):
    """1시간봉 EMA50, EMA200, VWAP, 온체인 데이터 활용, 시장 상태 결정 (v2.0)."""
    data = multi_tf_data.get("1h")
    if data is None:
        logging.warning("1h data not available; defaulting to sideways")
        regime = "sideways_rangebound" # 장세 세분화 (v2.0)
    else:
        current_price = data["current_price"]
        ema50 = data["ema50"] # EMA50으로 변경 (v2.0)
        ema200 = data["ema200"] # EMA200으로 변경 (v2.0)
        vwap = data["vwap"] # VWAP 추가 (v2.0)

        if ema50 is None or ema200 is None or vwap is None: # VWAP None 체크 추가 (v2.0)
            regime = "sideways_rangebound" # 장세 세분화 (v2.0)
        else:
            # 장세 유형 세분화 (v2.0)
            if abs(current_price - ema50) / ema50 < 0.01 and abs(current_price - ema200) / ema200 < 0.01 and abs(current_price - vwap) / vwap < 0.01:
                regime = "sideways_rangebound" # 레인지 횡보 (v2.0)
            elif current_price > ema50 and current_price > ema200 and current_price > vwap:
                regime = "bull_trend_following" # 추세 추종형 강세 (v2.0)
            elif current_price < ema50 and current_price < ema200 and current_price < vwap:
                regime = "bear_trend_following" # 추세 추종형 약세 (v2.0)
            elif abs(current_price - ema50) / ema50 > 0.03 or abs(current_price - ema200) / ema200 > 0.03 or abs(current_price - vwap) / vwap > 0.03:
                regime = "sideways_volatile" # 변동성 횡보 (v2.0)
            elif current_price > ema50 and current_price < ema200:
                regime = "bull_pullback" # 강세 속 풀백 (v2.0)
            elif current_price < ema50 and current_price > ema200:
                regime = "bear_pullback" # 약세 속 풀백 (v2.0)
            else:
                regime = "sideways_uncertain" # 불확실성 횡보 (v2.0)

    if onchain_data["mvrv"] != "N/A" and onchain_data["sopr"] != "N/A":
        if onchain_data["mvrv"] < 0: # MVRV 기준 강화 (v2.0)
            logging.info("On-chain metrics (MVRV) extreme undervaluation; adjusting regime to bull_reversal_signal") # 장세 세분화 (v2.0)
            regime = "bull_reversal_signal" # 강세 반전 시그널 (v2.0)
        elif onchain_data["sopr"] > 1.1: # SOPR 기준 강화 (v2.0)
            logging.info("On-chain metrics (SOPR) extreme overvaluation; adjusting regime to bear_reversal_signal") # 장세 세분화 (v2.0)
            regime = "bear_reversal_signal" # 약세 반전 시그널 (v2.0)

    if is_high_volatility(multi_tf_data):
        regime += "_high_volatility" # 고변동성 장세 추가 (v2.0)

    logging.info(f"Market regime determined: {regime.upper()}")
    return regime


def adjust_indicator_thresholds(market_regime):
    """시장 상태 따라 RSI, MACD, MA 등 임계값 동적 조정 (v2.0)."""
    thresholds = {} # thresholds 딕셔너리 초기화 (v2.0)

    if "bull" in market_regime: # 강세장 임계값 (v2.0)
        thresholds['rsi_oversold'] = 35 if "high_volatility" in market_regime else 40
        thresholds['rsi_overbought'] = 80 if "high_volatility" in market_regime else 75
        thresholds['macd_os_level'] = -5 if "high_volatility" in market_regime else -3 # MACD 과매도 레벨 (v2.0)
        thresholds['macd_ob_level'] = 10 if "high_volatility" in market_regime else 7 # MACD 과매수 레벨 (v2.0)
        thresholds['ma_band_ratio'] = 0.01 if "high_volatility" in market_regime else 0.005 # MA 밴드 비율 (v2.0)
        thresholds['vol_spike_factor'] = 2.0 if "high_volatility" in market_regime else 1.5 # 거래량 급등 계수 (v2.0)

    elif "bear" in market_regime: # 약세장 임계값 (v2.0)
        thresholds['rsi_oversold'] = 25 if "high_volatility" in market_regime else 30
        thresholds['rsi_overbought'] = 65 if "high_volatility" in market_regime else 60
        thresholds['macd_os_level'] = -7 if "high_volatility" in market_regime else -5 # MACD 과매도 레벨 (v2.0)
        thresholds['macd_ob_level'] = 5 if "high_volatility" in market_regime else 3 # MACD 과매수 레벨 (v2.0)
        thresholds['ma_band_ratio'] = 0.02 if "high_volatility" in market_regime else 0.01 # MA 밴드 비율 (v2.0)
        thresholds['vol_spike_factor'] = 2.5 if "high_volatility" in market_regime else 2.0 # 거래량 급등 계수 (v2.0)

    else: # 횡보장 임계값 (v2.0)
        thresholds['rsi_oversold'] = 30 if "high_volatility" in market_regime else 35
        thresholds['rsi_overbought'] = 70 if "high_volatility" in market_regime else 75
        thresholds['macd_os_level'] = -4 if "high_volatility" in market_regime else -2 # MACD 과매도 레벨 (v2.0)
        thresholds['macd_ob_level'] = 6 if "high_volatility" in market_regime else 4 # MACD 과매수 레벨 (v2.0)
        thresholds['ma_band_ratio'] = 0.015 if "high_volatility" in market_regime else 0.01 # MA 밴드 비율 (v2.0)
        thresholds['vol_spike_factor'] = 2.2 if "high_volatility" in market_regime else 1.8 # 거래량 급등 계수 (v2.0)

    logging.info(f"Indicator thresholds set for market regime {market_regime} (v2.0)")
    return thresholds


def choose_primary_timeframe(multi_tf_data):
    """거래량, 변동성, 추세 강도, SMA200 편차 종합 고려, Primary Timeframe 선정 (v2.0)."""
    primary_tf = None
    max_score = 0

    tf_weights = {"5m": 1, "15m": 1, "1h": 2, "4h": 3, "1d": 5} # TF별 가중치 (v2.0)

    for tf, data in multi_tf_data.items():
        if data["ema200"] is not None and data["atr"] is not None and data["volume_roc"] is not None: # EMA200, ATR, 거래량 변화율 None 체크 (v2.0)
            score = (
                tf_weights[tf] * abs(data["current_price"] - data["ema200"]) / data["ema200"] + # SMA200 -> EMA200 (v2.0)
                tf_weights[tf] * data["atr"] / data["current_price"] + # 변동성 반영 (v2.0)
                tf_weights[tf] * abs(data["volume_roc"]) # 거래량 변화율 반영 (v2.0)
            )
            if score > max_score:
                max_score = score
                primary_tf = tf

    logging.info(f"Primary timeframe chosen: '{primary_tf}' (score: {max_score:.2f}) (v2.0)")
    return primary_tf


# =====================================================
# 9. GPT 프롬프트 생성 및 거래 결정 (v2.0)
# =====================================================
def generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data,
                        onchain_data, multi_tf_data, market_regime, thresholds,
                        heatmap_analysis, econ_summary, primary_tf):
    """계좌, 시장, 지표, 확장 데이터 종합 XML 프롬프트 생성 (v2.0)."""
    # Multi-timeframe summary (v2.0)
    multi_tf_summary = ""
    for tf, data in multi_tf_data.items():
        multi_tf_summary += (
            f"<Timeframe tf='{tf}'>\n" # Timeframe XML 태그 추가 (v2.0)
            f"  <Price>{data['current_price']}</Price>\n"
            f"  <VolumeROC>{data['volume_roc']}</VolumeROC>\n" # 거래량 변화율 (v2.0)
            f"  <VolumeSpike>{data['volume_spike']}</VolumeSpike>\n" # 거래량 급증 (v2.0)
            f"  <RSI>{data['rsi']}</RSI>\n"
            f"  <CCI>{data['cci']}</CCI>\n" # CCI (v2.0)
            f"  <EMA20>{data['ema20']}</EMA20>\n" # EMA (v2.0)
            f"  <EMA50>{data['ema50']}</EMA50>\n" # EMA (v2.0)
            f"  <VWAP>{data['vwap']}</VWAP>\n" # VWAP (v2.0)
            f"  <DiffMA20>{data['ma20_diff']}</DiffMA20>\n" # MA Diff (v2.0)
            f"  <DiffMA50>{data['ma50_diff']}</DiffMA50>\n" # MA Diff (v2.0)
            f"  <DiffVWAP>{data['vwap_diff']}</DiffVWAP>\n" # VWAP Diff (v2.0)
            f"  <BBUpper>{data['bb_upper']}</BBUpper>\n"
            f"  <BBLower>{data['bb_lower']}</BBLower>\n"
            f"  <BBWidth>{data['bb_width']}</BBWidth>\n" # BB Width (v2.0)
            f"  <MACD>{data['macd']}</MACD>\n"
            f"  <MACDSignal>{data['macd_signal']}</MACDSignal>\n"
            f"  <MACDHist>{data['macd_hist']}</MACDHist>\n" # MACD Histogram (v2.0)
            f"  <ATR>{data['atr']}</ATR>\n"
            f"  <CandlePattern>{data['candle_pattern']}</CandlePattern>\n" # 캔들 패턴 (v2.0)
            f"</Timeframe>\n" # Timeframe XML 태그 종료 (v2.0)
        )

    # Fear & Greed Index (classification, value)
    fng_class, fng_value = extended_data.get("fear_and_greed_index", ("N/A", "N/A"))

    prompt = f"""<TradeBotPrompt>
    <Context> # Context XML 그룹화 (v2.0)
        <Account>
            <WalletBalance>{wallet_balance}</WalletBalance>
            <CurrentPosition>{position_info}</CurrentPosition>
        </Account>
        <Market>
            <Regime>{market_regime.upper()}</Regime>
            <PrimaryTimeframe>{primary_tf}</PrimaryTimeframe> # Primary Timeframe 정보 추가 (v2.0)
            <MultiTimeframes>
    {multi_tf_summary}
            </MultiTimeframes>
            <AdditionalData>
                <FundingRate>{extended_data.get('funding_rate', 'N/A')}</FundingRate>
                <OpenInterest>{extended_data.get('open_interest', 'N/A')}</OpenInterest>
                <OpenInterestChange>{extended_data.get('open_interest_change_1h', 'N/A')}</OpenInterestChange> # 미결제 약정 변화 (v2.0)
                <LongShortRatio>{extended_data.get('long_short_ratio', 'N/A')}</LongShortRatio> # 롱/숏 비율 (v2.0)
                <BTCDominance>{extended_data.get('btc_dominance', 'N/A')}</BTCDominance> # BTC Dominance (v2.0)
                <FuturePremiumRate>{extended_data.get('future_premium_rate', 'N/A')}</FuturePremiumRate> # 선물 프리미엄 (v2.0)
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
                        Analyze liquidation heatmap. Identify key zones, impact on price, risk side (long/short).
                        Format: "Long Liquidation Zone: <value>; Short Liquidation Zone: <value>; Impact: <analysis text>; Risk: <which side is at higher risk>."
                    </Guide>
                    <Analysis>{heatmap_analysis}</Analysis>
                </LiquidationHeatmap>
                <EconomicCalendar>
                    {econ_summary}
                </EconomicCalendar>
            </AdditionalData>
        </Market>
        <IndicatorThresholds> # IndicatorThresholds 그룹화 (v2.0)
            <RSI oversold="{thresholds['rsi_oversold']}" overbought="{thresholds['rsi_overbought']}">
                RSI Guide: Oversold below {thresholds['rsi_oversold']}, Overbought above {thresholds['rsi_overbought']}.
                Consider RSI divergence for trend reversal signals. # RSI Divergence 가이드 추가 (v2.0)
            </RSI>
            <CCI> # CCI 지표 추가 (v2.0)
                CCI Guide: CCI above +100 is overbought, below -100 is oversold.
                Use CCI to confirm RSI signals or identify divergences. # CCI 활용 가이드 추가 (v2.0)
            </CCI>
            <MovingAverages>
                <EMA20 ma_band_ratio="{thresholds['ma_band_ratio']}"> # EMA20, ma_band_ratio 속성 추가 (v2.0)
                    EMA20 Guide: Price above EMA20 is bullish, below is bearish.
                    Incorporate EMA20 band ratio ({thresholds['ma_band_ratio']}) for dynamic support/resistance. # EMA20 밴드 활용 가이드 추가 (v2.0)
                </EMA20>
                <EMA50> # EMA50 (v2.0)
                    EMA50 Guide: Dynamic support/resistance, especially in trending markets. {thresholds.get('ma_comment', 'Use EMA50 as dynamic support/resistance.')} # EMA50 활용 가이드 (v2.0)
                </EMA50>
                <VWAP> # VWAP (v2.0)
                    VWAP Guide: Below VWAP bearish, above bullish. Use for intraday bias and entry/exit points. # VWAP 활용 가이드 (v2.0)
                </VWAP>
            </MovingAverages>
            <BollingerBands bb_width_threshold="10"> # Bollinger Bands, bb_width_threshold 속성 추가 (v2.0)
                Bollinger Bands Guide: BB width > 10 indicates high volatility.
                Use BB squeeze for breakout potential and band riding in trends. # BB 활용 가이드 추가 (v2.0)
            </BollingerBands>
            <MACD os_level="{thresholds['macd_os_level']}" ob_level="{thresholds['macd_ob_level']}"> # MACD, os_level, ob_level 속성 추가 (v2.0)
                MACD Guide: {thresholds.get('macd_comment', 'Use MACD crossover, histogram, and divergence signals.')}
                Histogram above 0 bullish, below 0 bearish. Overbought level: {thresholds['macd_ob_level']}, oversold: {thresholds['macd_os_level']}. # MACD Histogram, 임계값 가이드 추가 (v2.0)
            </MACD>
            <Volume volume_spike_factor="{thresholds['vol_spike_factor']}"> # Volume, volume_spike_factor 속성 추가 (v2.0)
                Volume Guide: Volume ROC, Volume Spike (>{thresholds['vol_spike_factor']}x avg) indicate strong momentum. # 거래량 가이드 (v2.0)
            </Volume>
            <CandlePatterns> # CandlePatterns 그룹화 (v2.0)
                Candle Patterns Guide: Analyze patterns for confirmation with indicators.
                Prioritize patterns with higher reliability and timeframe agreement. # 캔들 패턴 가이드 (v2.0)
                List of Patterns: {', '.join(CANDLE_PATTERNS)} # 캔들 패턴 목록 (v2.0)
            </CandlePatterns>
            <OnChainIndicators>
                On-Chain Guide: MVRV < 0 undervaluation, SOPR > 1.1 overvaluation. # 온체인 지표 가이드 (v2.0)
            </OnChainIndicators>
        </IndicatorThresholds>
    </Context>
    <StrategyRules> # StrategyRules 그룹화 (v2.0)
        <TimeframeAgreement primary_tf="{primary_tf}"> # TimeframeAgreement, primary_tf 속성 추가 (v2.0)
            Multi-Timeframe Guide: Trade if 4/5 timeframes agree or primary TF shows strong signal (3/5 support).
            Higher weight on 4h, 1d timeframes. # 장기 TF 가중치 부여 (v2.0)
            Trade Term Guide: Lower TFs (5m, 15m) -> intraday/6h, Mid TFs (1h, 4h) -> 1d, 1d dominant -> 1w.
        </TimeframeAgreement>
        <RiskReward>
            Risk/Reward Guide: Open position if potential reward >= 2x risk.
            Consider liquidation heatmap for stop-loss placement and take-profit targets. # 청산 맵 활용 가이드 (v2.0)
        </RiskReward>
        <AdaptiveStrategy regime="{market_regime}"> # AdaptiveStrategy, regime 속성 추가 (v2.0)
            Strategy Switch Guide: Trend-following in bull/bear trends, mean reversion in sideways.
            Incorporate technicals, on-chain, sentiment, liquidation heatmap, and economic events.
            Adapt strategy based on market volatility and session liquidity. # 변동성, 세션 유동성 고려 (v2.0)
        </AdaptiveStrategy>
        <TradingStyle> # TradingStyle 가이드 추가 (v2.0)
            Trading Style Guide: Mimic top Bitcoin traders - be patient, data-driven, and risk-conscious.
            Focus on high-probability setups, not over-trading. # 트레이딩 스타일 가이드 (v2.0)
        </TradingStyle>
    </StrategyRules>
    <OrderExecution> # OrderExecution 가이드 추가 (v2.0)
        <TPSL>
            TP/SL Guide: Use Fibonacci Retracement/Extension for target levels. # 피보나치 활용 가이드 (v2.0)
            Consider ATR trailing stop for dynamic stop-loss. # ATR Trailing Stop 가이드 (v2.0)
            Set SL below key support or above resistance; TP near next resistance or support. # 지지/저항선 활용 가이드 (v2.0)
            Refine TP/SL based on liquidation heatmap zones. # 청산 맵 기반 TP/SL 조정 (v2.0)
        </TPSL>
        <LimitOrder>
            Limit Order Guide: Use limit orders for precise entry/exit. # 지정가 주문 가이드 (v2.0)
            Set limit order slightly better than current price for buy/sell. # 지정가 설정 팁 (v2.0)
            Consider order book analysis for optimal limit order placement. # 호가창 분석 활용 (v2.0)
        </LimitOrder>
    </OrderExecution>
    <Task>
        Decision Task: Based on all provided data and guidelines, decide: GO_LONG, GO_SHORT, HOLD_LONG, HOLD_SHORT, NO_TRADE.
        Recommend: Leverage (e.g., 3x, 5x), Trade Term (e.g., 6h, 1d, 1w), Take Profit Price, Stop Loss Price, Limit Order Price, and Rationale.
        Output format: Final Action, Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Rationale.
    </Task>
    <OutputExample>
        GO_LONG, 5x, 6h, 11400, 10800, 11300, Bullish momentum across timeframes, RSI dip entry, EMA50 support, R/R 3:1.
    </OutputExample>
</TradeBotPrompt>"""
    return prompt


def generate_trading_decision(wallet_balance, position_info, aggregated_data, extended_data,
                              onchain_data, multi_tf_data, market_regime, thresholds,
                              heatmap_analysis, econ_summary, primary_tf):
    """GPT 통해 프롬프트 전달, 거래 결정 수신 (v2.0)."""
    prompt = generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data,
                                 onchain_data, multi_tf_data, market_regime, thresholds,
                                 heatmap_analysis, econ_summary, primary_tf)

    developer_prompt = (
        "You are a world-class Bitcoin trader AI. Mimic top crypto traders' decision-making process. " # 최고 트레이더 강조 (v2.0)
        "Analyze all factors comprehensively, prioritize risk management, and make data-driven decisions. " # 위험 관리, 데이터 기반 결정 강조 (v2.0)
        "Adhere strictly to XML guidelines. Provide concise, actionable output in specified format." # XML 가이드라인 준수 강조 (v2.0)
        "Incorporate adaptive strategies based on market regime and volatility." # 적응형 전략 강조 (v2.0)
    )
    response = gpt_client.chat.completions.create(
        model="o3-mini", # 필요시 고성능 모델로 변경 (v2.0)
        reasoning_effort="high", # 높은 추론 노력 (v2.0)
        messages=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def parse_trading_decision(response_text):
    """GPT 응답(콤마 구분) 파싱, 거래 결정 필드 추출."""
    response_text = response_text.strip()
    parts = [part.strip() for part in re.split(r'\s*,\s*', response_text)]
    if len(parts) < 7:
        raise ValueError("Incomplete response: at least 7 comma-separated fields required.")
    decision = {
        "final_action": parts[0].replace("GO_", ""), # "GO_" 제거 (v2.0)
        "leverage": parts[1],
        "trade_term": parts[2],
        "tp_price": parts[3],
        "sl_price": parts[4],
        "limit_order_price": parts[5],
        "rationale": ", ".join(parts[6:])
    }
    return decision


# =====================================================
# 10. 포지션 로깅 함수 (v2.0)
# =====================================================
def log_decision(decision, symbol):
    """거래 결정 내용 CSV 파일 기록."""
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
    """신규 포지션 오픈 내역 CSV 파일 기록."""
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
    """청산 포지션 내역 및 수익 CSV 파일 기록."""
    profit = (exit_price - entry_price) if trade_side.upper() == "LONG" else (entry_price - exit_price)
    file_exists = os.path.isfile(CLOSED_POSITIONS_FILE)
    with open(CLOSED_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "trade_side", "entry_price", "exit_price", "profit"])
        writer.writerow([datetime.utcnow(), symbol, trade_side, entry_price, exit_price, profit])
    logging.info(f"{symbol} closed position logged (profit: {profit}).")


# =====================================================
# 11. 포지션 관리 및 메인 트레이딩 로직 (v2.0)
# =====================================================
def compute_risk_reward(decision, entry_price):
    """결정된 거래 가격, 목표, 손절가 이용, 위험/보상 비율 계산."""
    try:
        tp_price = float(decision["tp_price"])
        sl_price = float(decision["sl_price"])
        if decision["final_action"].upper() == "LONG": # "GO_" 제거 (v2.0)
            reward = tp_price - entry_price
            risk = entry_price - sl_price
        elif decision["final_action"].upper() == "SHORT": # "GO_" 제거 (v2.0)
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
    """주문서, 거래소 유입량, 펀딩비, OI, FNG, 롱/숏 비율 등 추가 데이터 fetch (v2.0)."""
    base_data = fetch_order_book(symbol)
    exchange_inflows = fetch_exchange_inflows()
    funding_rate = fetch_funding_rate(symbol)
    open_interest = fetch_open_interest(symbol)
    oi_change_1h = fetch_open_interest_change(symbol, period="1h") # 1시간 봉 기준 OI 변화 (v2.0)
    long_short_ratio = fetch_long_short_ratio() # 롱/숏 비율 (v2.0)
    btc_dominance = fetch_btc_dominance() # BTC 도미넌스 (v2.0)
    future_premium_rate = fetch_future_premium_rate(symbol) # 선물 프리미엄 (v2.0)
    fng = fetch_fear_and_greed_index()  # (classification, value)

    extended_data = {
        "order_book_bid": base_data.get("bid", "N/A"),
        "order_book_ask": base_data.get("ask", "N/A"),
        "order_book_spread": base_data.get("spread", "N/A"),
        "exchange_inflows": exchange_inflows,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "open_interest_change_1h": oi_change_1h, # 미결제 약정 변화 (v2.0)
        "long_short_ratio": long_short_ratio, # 롱/숏 비율 (v2.0)
        "btc_dominance": btc_dominance, # BTC 도미넌스 (v2.0)
        "future_premium_rate": future_premium_rate, # 선물 프리미엄 (v2.0)
        "fear_and_greed_index": fng
    }
    return extended_data


def get_current_session_kst():
    """KST 기준 현재 시간 확인, 트레이딩 세션 결정."""
    now = datetime.now(KST)
    hour = now.hour

    if 0 <= hour < 8:
        return "OVERNIGHT_LOW_LIQUIDITY"  # 00:00 ~ 08:00
    elif 8 <= hour < 16:
        return "ASIAN_SESSION"  # 08:00 ~ 16:00
    else:
        return "EUROPEAN_US_SESSION"  # 16:00 ~ 24:00 (런던/미국 겹치는 구간)


def is_news_volatility_period(minutes_before=10, minutes_after=10):
    """주요 경제 지표 발표 전후 기간, 변동성 상승 여부 판단."""
    major_news_times = [datetime.now(KST).replace(hour=22, minute=30, second=0, microsecond=0)]
    now_kst = datetime.now(KST)

    for news_time in major_news_times:
        diff_in_minutes = abs((now_kst - news_time).total_seconds()) / 60.0
        if diff_in_minutes <= minutes_before or diff_in_minutes <= minutes_after:
            if diff_in_minutes < (minutes_before + minutes_after):
                return True
    return False


def fetch_economic_data():
    """Investing.com 등에서 경제 캘린더 데이터 fetch."""
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


def parse_economic_data(data):
    """Fetch한 경제 데이터 HTML 파싱, 이벤트 정보 추출."""
    if data is None:
        return []

    try:
        html_data = data['data']
        soup = BeautifulSoup(html_data, 'html.parser')
        rows = soup.find_all('tr', class_='js-event-item')

        # 반환값: [(time_str, currency, impact_level, event_name, time_diff_minutes), ...]
        # impact_level: '🔴🔴🔴'
        # time_diff_minutes: 발표 시간 - 현재 시간 (분 단위)
        result_list = []
        now_kst = datetime.now(KST)

        for row in rows:
            time_str = row.find('td', class_='time').text.strip()
            currency = row.find('td', class_='flagCur').text.strip()
            impact_icons = row.find('td', class_='sentiment').find_all('i')
            impact_level = "🔴" * len(impact_icons)
            event_name = row.find('td', class_='event').text.strip()

            # 실제 발표 시간 KST 날짜/시간으로 해석
            try:
                hour_min = time_str.split(':')
                event_hour = int(hour_min[0])
                event_min = int(hour_min[1])
                event_dt = now_kst.replace(hour=event_hour, minute=event_min, second=0, microsecond=0)
            except:
                continue

            time_diff = (event_dt - now_kst).total_seconds() / 60.0  # 분 단위
            result_list.append([time_str, currency, impact_level, event_name, time_diff])

        return result_list
    except (KeyError, AttributeError) as e:
        logging.error(f"Error parsing data: {e}")
        return []


def get_economic_event_summary(econ_list):
    """econ_list 기반 경제 이벤트 요약 정보 생성."""
    if not econ_list:
        return "*No major economic events for the day.*" # Markdown 강조 (v2.0)

    lines = ["*Major Economic Events:*"] # Markdown 강조 (v2.0)
    now_kst = datetime.now(KST)
    for item in econ_list:
        time_str, currency, impact, event_name, diff_minutes = item
        if diff_minutes > 0:
            lines.append(f"- `{time_str}` ({currency}, {impact}): {event_name} (about {int(diff_minutes)} min left)") # Markdown 코드 블록, 리스트 (v2.0)
        else:
            lines.append(f"- `{time_str}` ({currency}, {impact}): {event_name} ({abs(int(diff_minutes))} min since release)") # Markdown 코드 블록, 리스트 (v2.0)

    return "\n".join(lines)


def detect_fake_breakout(df, lookback=20, volume_factor=1.5, rsi_threshold=(30, 70), pivot_dist=0.001):
    """봉 데이터 기반 가짜 돌파 감지."""
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
    volume_check = (last_vol < prev_20_vol_mean * volume_factor)  # 거래량 부족 시 의심
    rsi_check = ((breakout_up and last_rsi < rsi_threshold[1] - 5) or  # 위로 돌파인데 RSI 낮으면 힘 부족
                 (breakout_down and last_rsi > rsi_threshold[0] + 5))  # 아래로 돌파인데 RSI 높으면 가짜
    # 되돌림 체크: 종가-고가/저가 차이 비율
    candle_range = last_candle['high'] - last_candle['low']
    if candle_range == 0:
        retrace_ratio = 0
    else:
        if breakout_up:
            retrace_ratio = (last_candle['high'] - last_candle['close']) / candle_range
        else:
            retrace_ratio = (last_candle['close'] - last_candle['low']) / candle_range
    retrace_check = (retrace_ratio > 0.6)  # 돌파 후 60% 이상 되돌림 => 윗꼬리/아랫꼬리 심함

    # 가중치 합산
    suspicion_score = 0
    if volume_check:
        suspicion_score += 1
    if rsi_check:
        suspicion_score += 1
    if retrace_check:
        suspicion_score += 1

    # 2점 이상이면 가짜 돌파
    if suspicion_score >= 2:
        return True
    return False


def analyze_session_open_volatility(df, open_hour=8, open_minute=0, threshold_factor=1.5, window=5):
    """세션 오픈 시 변동성 분석."""
    now_kst = datetime.now(KST)
    open_dt = now_kst.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)

    # 세션 오픈 전후 30분 정도 감지
    diff_minutes = (now_kst - open_dt).total_seconds() / 60.0
    if abs(diff_minutes) > 30:
        return False  # 세션 오픈 30분 후 체크 X

    # window 분 내 봉 추출
    recent_time = open_dt + timedelta(minutes=window)
    subdf = df[(df['timestamp'] >= open_dt) & (df['timestamp'] <= recent_time)]
    if len(subdf) < 1:
        return False

    price_range = subdf['high'].max() - subdf['low'].min()
    # 최근 ATR 평균 (~20봉)
    last_atr = df['atr'].iloc[-20:].mean() if len(df) >= 20 else df['atr'].mean()
    # 실제 변동성 체크
    if last_atr and last_atr > 0:
        if price_range / last_atr > threshold_factor:
            return True
    return False


def main():
    logging.info("Trading bot started. (v2.0)")
    wallet_balance = "1000 USDT"
    position_info = "NONE"
    in_position = False
    current_side = None
    entry_price = 0.0

    # 경제 지표 파싱 및 요약 (v2.0)
    econ_data_raw = fetch_economic_data()
    econ_list = parse_economic_data(econ_data_raw)
    econ_summary = get_economic_event_summary(econ_list)

    # 멀티 TF 데이터 fetch (v2.0)
    mtf = fetch_multi_tf_data(SYMBOL, TIMEFRAMES, limit=300) # TIMEFRAMES 상수 사용 (v2.0)
    if not mtf or "1h" not in mtf:
        logging.error("Not enough TF data.")
        return

    cprice = mtf["1h"]["current_price"]
    ext_data = fetch_additional_data(SYMBOL) # 확장 데이터 fetch (v2.0)
    onchain_data = fetch_onchain_data(SYMBOL)

    # 세션 및 변동성 체크 (v2.0)
    session_name = get_current_session_kst()
    logging.info(f"Current session: {session_name}")

    if "5m" in mtf: # 5분봉 가짜 돌파 감지 (v2.0)
        df_5m = mtf["5m"]["df_full"]
        if detect_fake_breakout(df_5m):
            logging.info("Fake breakout suspicion on 5m. Strategy may filter signals.")

        df_5m_vol = mtf["5m"]["df_full"] # 5분봉 세션 오픈 변동성 분석 (v2.0)
        if analyze_session_open_volatility(df_5m_vol, 8, 0, threshold_factor=1.5, window=6):
            logging.info("High volatility at Asia open. Breakout strategy alert.")
        if analyze_session_open_volatility(df_5m_vol, 16, 0, threshold_factor=1.5, window=6):
            logging.info("High volatility at London open. Breakout strategy alert.")

    # TF별 방향성 분석 및 합의 (v2.0)
    bullish_cnt, bearish_cnt = 0, 0
    directions = {}
    tf_ema50_diff_sum = 0 # EMA50 Diff 합계 (v2.0)
    for tf, data in mtf.items():
        ema50_diff = data["ma50_diff"] # EMA50 Diff (v2.0)
        directions[tf] = "sideways" # 기본 sideways (v2.0)
        if data["ema200"] is not None and data["current_price"] is not None: # EMA200 None 체크 (v2.0)
            if data["current_price"] > data["ema200"]:
                directions[tf] = "bullish"
                bullish_cnt += 1
            elif data["current_price"] < data["ema200"]:
                directions[tf] = "bearish"
                bearish_cnt += 1
        if ema50_diff is not None: # EMA50 Diff 합계 (v2.0)
            tf_ema50_diff_sum += ema50_diff # EMA50 Diff 누적 (v2.0)
    aggregated_trend = "BULL" if bullish_cnt > bearish_cnt else "BEAR" # bullish_cnt >= -> > (v2.0)
    aggregated_data = {"trend": aggregated_trend, "timeframe_directions": directions, "ema50_diff_sum": tf_ema50_diff_sum} # EMA50 Diff 합계 추가 (v2.0)

    primary_tf = choose_primary_timeframe(mtf) # Primary TF 선정 (v2.0)
    primary_dir = directions[primary_tf] if primary_tf else None # Primary TF 방향 (v2.0)

    # 사전 필터링 로직 강화 (v2.0)
    if not (bullish_cnt >= 4 or bearish_cnt >= 4):
        if not (
                (primary_dir == "bullish" and bullish_cnt >= 3 and tf_ema50_diff_sum > 0) or # EMA50 Diff 조건 추가 (v2.0)
                (primary_dir == "bearish" and bearish_cnt >= 3 and tf_ema50_diff_sum < 0) # EMA50 Diff 조건 추가 (v2.0)
        ):
            logging.info("No strong consensus or EMA50 Diff confirmation. No GPT decision.") # EMA50 Diff 조건 추가 (v2.0)
            driver = get_driver()
            driver.quit()
            return

    # 시장 regime 판단 및 임계값 조정 (v2.0)
    regime = determine_market_regime(mtf, onchain_data)
    thresholds = adjust_indicator_thresholds(regime)

    # Liquidation heatmap 분석 (v2.0)
    fetch_liquidation_heatmap()
    heatmap_analysis = analyze_liquidation_heatmap()

    # GPT decision generation (v2.0)
    try:
        gpt_resp = generate_trading_decision(
            wallet_balance=wallet_balance,
            position_info=position_info,
            aggregated_data=aggregated_data,
            extended_data=ext_data,
            onchain_data=onchain_data,
            multi_tf_data=mtf,
            market_regime=regime,
            thresholds=thresholds,
            heatmap_analysis=heatmap_analysis,
            econ_summary=econ_summary,
            primary_tf=primary_tf # Primary TF 정보 추가 (v2.0)
        )
        logging.info(f"GPT raw response: {gpt_resp}") # GPT 응답 로그 (v2.0)
        decision = parse_trading_decision(gpt_resp)
        logging.info(f"Parsed decision: {decision}")
        log_decision(decision, SYMBOL)
    except Exception as e:
        logging.error(f"Error during GPT decision making: {e}") # GPT 에러 로그 (v2.0)
        return

    rr = compute_risk_reward(decision, cprice)
    rr_text = f"{rr:.2f}" if rr else "N/A"

    # 텔레그램 메시지 전송 (v2.0)
    if decision["final_action"].upper() in ["LONG", "SHORT"]: # "GO_" 제거 (v2.0)
        side = "BUY" if decision["final_action"].upper() == "LONG" else "SELL" # "GO_" 제거 (v2.0)
        msg_lines = [ # 메시지 format 변경 (v2.0)
            f"*📊 {side} SIGNAL ({SYMBOL}) 📊*", # 강조 (v2.0)
            f"- *Leverage:* {decision['leverage']}", # 리스트, 강조 (v2.0)
            f"- *R/R Ratio:* {rr_text}", # 리스트, 강조 (v2.0)
            f"- *Trade Term:* {decision['trade_term']}", # 리스트, 강조 (v2.0)
            f"- *Limit Order Price:* {decision['limit_order_price']}", # 리스트, 강조 (v2.0)
            f"- *Take Profit:* {decision['tp_price']}", # 리스트, 강조 (v2.0)
            f"- *Stop Loss:* {decision['sl_price']}", # 리스트, 강조 (v2.0)
            f"\n*Market Analysis:*\n{decision['rationale']}", # 섹션 분리, 강조 (v2.0)
            f"\n_Market Regime:_ `{regime.upper()}`", # 섹션 분리, Markdown 코드 블록 (v2.0)
            f"_Primary Timeframe:_ `{primary_tf}`" # 섹션 분리, Markdown 코드 블록 (v2.0)
        ]
        msg = "\n".join(msg_lines)
        send_telegram_message(msg)

    if not in_position:
        if rr and rr >= 2 and decision["final_action"].upper() in ["LONG", "SHORT"]: # "GO_" 제거 (v2.0)
            logging.info(f"Opening {decision['final_action']} position @ {cprice} (v2.0)") # "GO_" 제거 (v2.0)
            log_open_position(SYMBOL, decision, cprice)
            in_position = True
            current_side = decision["final_action"].upper() # "GO_" 제거 (v2.0)
            entry_price = cprice
        else:
            logging.info("No new position (R/R < 2 or not GO_LONG/GO_SHORT).")
    else:
        if decision["final_action"].upper() not in ["HOLD_LONG", "HOLD_SHORT"]: # "GO_" 제거 (v2.0)
            logging.info(f"Exiting {current_side} position @ {cprice} (v2.0)") # current_side 변수 활용 (v2.0)
            log_closed_position(SYMBOL, entry_price, cprice, current_side)
            in_position = False
        else:
            logging.info(f"Holding current {current_side} position. (v2.0)") # current_side 변수 활용 (v2.0)


if __name__ == "__main__":
    main()