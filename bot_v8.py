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

# OPENAI_API_KEY must be set in the environment
openai_api_key = os.environ.get('OPENAI_API_KEY')
gpt_client = OpenAI(api_key=openai_api_key)

# GOOGLE_API_KEY must be set in the environment
google_api_key = os.environ.get('GOOGLE_API_KEY')
gemini_client = genai.Client(api_key=google_api_key)

# Telegram variables (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# [새로 추가] KST 타임존 설정
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
    DataFrame에 다양한 기술적 지표를 계산하여 추가한다.
    - 거래량 변화율, 거래량 급증/급감
    - RSI, CCI
    - EMA, VWAP (거래량 가중 이동평균)
    - MACD, Bollinger Bands, ATR
    """
    # 1. 거래량 지표
    df['volume_change'] = df['volume'].pct_change() * 100  # 거래량 변화율 (%)
    df['volume_surge'] = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5  # 거래량 급증 여부 (20MA 대비 1.5배)
    df['volume_drop'] = df['volume'] < df['volume'].rolling(window=20).mean() * 0.5  # 거래량 급감 여부 (20MA 대비 0.5배)

    # 2. 모멘텀 지표: RSI, CCI (MFI 제거, RSI 집중 및 CCI 추가)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()

    # 3. 이동평균선: EMA, VWAP (SMA 대신 EMA, VWAP 활용)
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'],
                                                      window=20).volume_weighted_average_price()

    # 4. 추세 및 변동성 지표 (MACD, Bollinger Bands, ATR) - 기존 지표 활용
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()  # MACD 선
    df['macd_signal'] = macd.macd_signal()  # Signal 선
    df['macd_hist'] = macd.macd_diff()  # 히스토그램

    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']  # 볼린저 밴드 폭 추가

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # 5. 가격-이동평균선 괴리율 (EMA 기준)
    df['ema50_diff'] = (df['close'] - df['ema50']) / df['ema50'] * 100
    df['ema200_diff'] = (df['close'] - df['ema200']) / df['ema200'] * 100

    logging.info("Technical indicators calculated (enhanced version)")
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


# =====================================================
# 5. 확장 데이터 수집 (크롤링, API 등)
# =====================================================
def fetch_exchange_inflows():
    """
    CryptoQuant에서 거래소 순입출금 데이터를 크롤링한다.
    """
    url = "https://cryptoquant.com/asset/btc/chart/exchange-flows"
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


def fetch_open_interest_change(symbol, timeframe='1h', period_hours=24):
    """
    최근 period_hours 동안의 Open Interest 변화량(증가/감소)을 계산한다.
    """
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        symbol_futures = symbol.replace("/", "")
        now_timestamp_ms = futures_exchange.milliseconds()
        since_timestamp_ms = now_timestamp_ms - period_hours * 3600 * 1000  # period_hours 시간 전

        # 시간별로 Open Interest 데이터 fetch (Binance는 시간별 데이터 제공)
        ohlcv_oi = futures_exchange.fetch_ohlcv(
            symbol=symbol_futures,
            timeframe=timeframe,  # 1시간봉 기준
            since=since_timestamp_ms,
            limit=period_hours,  # period_hours 만큼 데이터 요청
            params={'fields': ['openInterest']}
        )
        if not ohlcv_oi or len(ohlcv_oi) < 2:
            return "N/A"

        # 수정된 부분: 컬럼명 수정 (6개 컬럼에 맞게)
        oi_df = pd.DataFrame(ohlcv_oi, columns=['timestamp', 'oi_open_interest', 'open', 'high', 'low', 'volume'])
        oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
        oi_df['oi_open_interest'] = pd.to_numeric(oi_df['oi_open_interest'])  # open_interest -> oi_open_interest 로 변경

        # Open Interest 변화율 계산 (최근 데이터 vs period_hours 전 데이터)
        current_oi = oi_df['oi_open_interest'].iloc[-1]  # open_interest -> oi_open_interest 로 변경
        past_oi = oi_df['oi_open_interest'].iloc[0]  # open_interest -> oi_open_interest 로 변경
        oi_change_percent = ((current_oi - past_oi) / past_oi) * 100 if past_oi != 0 else 0

        logging.info(f"{symbol} open interest change ({period_hours}H) fetched successfully: {oi_change_percent:.2f}%")
        return round(oi_change_percent, 2)

    except Exception as e:
        logging.error(f"Error fetching open interest change for {symbol}: {e}")
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


def fetch_future_spot_diff(symbol):
    """
    Binance Futures와 현물(Spot) 가격 차이를 계산한다.
    """
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        spot_exchange = ccxt.binance()
        symbol_futures = symbol.replace("/", "")

        # 선물, 현물 가격 fetch
        futures_ticker = futures_exchange.fetch_ticker(symbol=symbol_futures)
        spot_ticker = spot_exchange.fetch_ticker(symbol=symbol)

        future_price = futures_ticker['last'] if futures_ticker else None
        spot_price = spot_ticker['last'] if spot_ticker else None

        if future_price is not None and spot_price is not None:
            price_diff_percent = ((future_price - spot_price) / spot_price) * 100
            logging.info(f"{symbol} futures vs spot price diff: {price_diff_percent:.2f}%")
            return round(price_diff_percent, 2)
        else:
            return "N/A"

    except Exception as e:
        logging.error(f"Error fetching future vs spot price diff for {symbol}: {e}")
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
    온체인 지표(MVRV, SOPR 등)를 TradingView에서 크롤링하여 가져온다.
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
        latest = df.iloc[-1]
        multi_tf_data[tf] = {
            "current_price": round(latest['close'], 2),
            "volume_change": round(latest['volume_change'], 2) if not np.isnan(latest['volume_change']) else None,
            "volume_surge": latest['volume_surge'],
            "volume_drop": latest['volume_drop'],
            "rsi": round(latest['rsi'], 2) if not np.isnan(latest['rsi']) else None,
            "cci": round(latest['cci'], 2) if not np.isnan(latest['cci']) else None,
            "ema20": round(latest['ema20'], 2) if not np.isnan(latest['ema20']) else None,
            "ema50": round(latest['ema50'], 2) if not np.isnan(latest['ema50']) else None,
            "ema200": round(latest['ema200'], 2) if not np.isnan(latest['ema200']) else None,
            "vwap": round(latest['vwap'], 2) if not np.isnan(latest['vwap']) else None,
            "ema50_diff": round(latest['ema50_diff'], 2) if not np.isnan(latest['ema50_diff']) else None,
            "ema200_diff": round(latest['ema200_diff'], 2) if not np.isnan(latest['ema200_diff']) else None,
            "bb_upper": round(latest['bb_upper'], 2) if not np.isnan(latest['bb_upper']) else None,
            "bb_lower": round(latest['bb_lower'], 2) if not np.isnan(latest['bb_lower']) else None,
            "bb_width": round(latest['bb_width'], 2) if not np.isnan(latest['bb_width']) else None,
            "macd": round(latest['macd'], 2) if not np.isnan(latest['macd']) else None,
            "macd_signal": round(latest['macd_signal'], 2) if not np.isnan(latest['macd_signal']) else None,
            "macd_hist": round(latest['macd_hist'], 2) if not np.isnan(latest['macd_hist']) else None,
            "atr": round(latest['atr'], 2) if not np.isnan(latest['atr']) else None,
            "timestamp": latest['timestamp'],
            "df_full": df
        }
    logging.info("Multi-timeframe data and enhanced indicators calculated")
    return multi_tf_data


# =====================================================
# 6. 청산 히트맵 데이터 및 분석 (기존과 동일)
# =====================================================
def fetch_liquidation_heatmap():
    """
    청산 히트맵 데이터를 CoinAnk 사이트에서 다운로드한다.
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


def analyze_liquidation_heatmap():
    """
    다운로드된 청산 히트맵 이미지를 Gemini를 통해 분석하고,
    지정된 포맷의 분석 결과를 반환한다.
    """
    image_path = "/Users/changpt/Downloads/Liquidation Heat Map.png"  # 사용자 다운로드 경로에 맞춰 수정 필요
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

    # Delete the image file after processing
    try:
        os.remove(image_path)
        logging.info("Deleted the liquidation heatmap image file after processing.")
    except Exception as e:
        logging.error("Error deleting the image file: " + str(e))

    logging.info("Liquidation heatmap analysis result:")
    logging.info(analysis_result)
    return analysis_result


# =====================================================
# 7. 시장 상태 결정 및 지표 임계값 조정 (장세 분류 및 가중치 조정)
# =====================================================
def determine_market_regime(multi_tf_data, onchain_data, extended_data):
    """
    다양한 지표와 데이터를 활용하여 시장 상태를 상세하게 분류하고,
    각 시장 regime별 가중치를 적용한다. (총 9가지 Regime으로 세분화)
    - 추세 강도, 변동성, 거래량, 캔들 패턴, 심리 지표, 온체인 데이터 종합 분석
    """
    tf_1h_data = multi_tf_data.get("1h")
    if not tf_1h_data:
        logging.warning("1h TF data not available for market regime analysis.")
        return "sideways_neutral"  # 1H 데이터 없으면, 횡보-중립 regime으로 fallback

    price = tf_1h_data["current_price"]
    ema50 = tf_1h_data["ema50"]
    ema200 = tf_1h_data["ema200"]
    atr_ratio = tf_1h_data["atr"] / price if price else 0
    bb_width_ratio = tf_1h_data["bb_width"] / price if price else 0
    volume_change = tf_1h_data["volume_change"] or 0  # 거래량 변화율
    fng_value = extended_data.get("fear_and_greed_index", (None, 50))[1] or 50  # Fear & Greed Index 값, default 50
    btc_dominance = extended_data.get("btc_dominance", 50) or 50  # BTC 도미넌스, default 50
    future_spot_diff = extended_data.get("future_spot_diff", 0) or 0  # 선물-현물 가격차이, default 0
    long_short_ratio = extended_data.get("long_short_ratio", 0.5) or 0.5  # 롱/숏 비율, default 0.5

    regime = "sideways_neutral"  # 기본 regime: 횡보-중립

    # 1. 추세 및 방향성 판단
    if price > ema50 and price > ema200:  # EMA50, EMA200 상회
        if volume_change > 10 and fng_value > 70 and future_spot_diff > 1:  # 거래량, FNG, 선물-현물 차이 > 특정 기준
            regime = "bull_strong_momentum"  # 강세 - 강력 추세 (momentum)
        elif atr_ratio > 0.03 or bb_width_ratio > 0.05:  # 변동성 기준 추가
            regime = "bull_volatile"  # 강세 - 변동성 장세
        else:
            regime = "bull_gradual_rise"  # 강세 - 점진적 상승
    elif price < ema50 and price < ema200:  # EMA50, EMA200 하회
        if volume_change < -10 and fng_value < 30 and future_spot_diff < -1:  # 거래량, FNG, 선물-현물 차이 < 특정 기준
            regime = "bear_strong_momentum"  # 약세 - 강력 하락 (momentum)
        elif atr_ratio > 0.03 or bb_width_ratio > 0.05:  # 변동성 기준 추가
            regime = "bear_volatile"  # 약세 - 변동성 장세
        else:
            regime = "bear_gradual_decline"  # 약세 - 점진적 하락
    else:  # EMA50, EMA200 수렴 구간 (횡보)
        if atr_ratio > 0.04 or bb_width_ratio > 0.06:  # 높은 변동성 기준
            regime = "sideways_volatile"  # 횡보 - 변동성 장세
        elif fng_value < 25 and btc_dominance > 60 and long_short_ratio < 0.4:  # 극단적 공포 심리 + BTC dominance 강세
            regime = "sideways_consolidation"  # 횡보 - 수렴/매집 (consolidation)
        else:
            regime = "sideways_neutral"  # 횡보 - 중립

    logging.info(f"Market Regime: {regime.upper()}")
    return regime


def adjust_indicator_thresholds(market_regime):
    """
    시장 상태(regime)에 따라 RSI, MACD, MA 등의 임계값을 동적으로 조정한다.
    """
    thresholds = {  # 기본 thresholds (sideways_neutral)
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "cci_oversold": -150,  # CCI thresholds 추가
        "cci_overbought": 150,
        "macd_hist_strong_bull": 5,  # MACD Histogram thresholds 추가 (강세/약세 기준)
        "macd_hist_strong_bear": -5,
        "ma_diff_threshold_medium": 1.0,  # MA diff thresholds (중간/강한 추세 기준)
        "ma_diff_threshold_strong": 3.0,
        "atr_volatility_threshold_high": 0.02,  # ATR volatility thresholds (높은 변동성 기준)
        "atr_volatility_threshold_moderate": 0.01
    }

    if "bull" in market_regime:  # 강세 regime
        thresholds["rsi_oversold"] = 40  # RSI oversold 기준 상향
        thresholds["rsi_overbought"] = 80  # RSI overbought 기준 상향
        thresholds["cci_oversold"] = -100  # CCI oversold 기준 상향 (덜 민감하게)
        thresholds["cci_overbought"] = 200  # CCI overbought 기준 상향 (더욱 과열)
        thresholds["macd_hist_strong_bull"] = 3  # MACD histogram bull 기준 완화
        thresholds["ma_diff_threshold_medium"] = 0.5  # MA diff 중간 추세 기준 완화
        if "strong_momentum" in market_regime:  # 강력 추세 강세장
            thresholds["rsi_oversold"] = 50  # RSI oversold 기준 추가 상향 (추세 추종 강화)
            thresholds["rsi_overbought"] = 90  # RSI overbought 기준 추가 상향 (극단적 과열)
            thresholds["macd_hist_strong_bull"] = 1  # MACD histogram bull 기준 더욱 완화
            thresholds["ma_diff_threshold_strong"] = 2.0  # MA diff 강한 추세 기준 강화
        elif "volatile" in market_regime:  # 변동성 강세장
            thresholds["rsi_oversold"] = 35  # RSI oversold 기준 약간 상향
            thresholds["rsi_overbought"] = 75  # RSI overbought 기준 약간 상향
            thresholds["cci_oversold"] = -120  # CCI oversold 기준 약간 상향
            thresholds["cci_overbought"] = 180  # CCI overbought 기준 약간 상향
            thresholds["atr_volatility_threshold_high"] = 0.03  # 높은 변동성 기준 상향
    elif "bear" in market_regime:  # 약세 regime
        thresholds["rsi_overbought"] = 60  # RSI overbought 기준 하향
        thresholds["rsi_oversold"] = 20  # RSI oversold 기준 하향
        thresholds["cci_overbought"] = 100  # CCI overbought 기준 하향 (덜 민감하게)
        thresholds["cci_oversold"] = -200  # CCI oversold 기준 하향 (더욱 과매도)
        thresholds["macd_hist_strong_bear"] = -3  # MACD histogram bear 기준 완화
        thresholds["ma_diff_threshold_medium"] = 0.5  # MA diff 중간 추세 기준 완화
        if "strong_momentum" in market_regime:  # 강력 추세 약세장
            thresholds["rsi_overbought"] = 50  # RSI overbought 기준 추가 하향 (추세 추종 강화)
            thresholds["rsi_oversold"] = 10  # RSI oversold 기준 추가 하향 (극단적 과매도)
            thresholds["macd_hist_strong_bear"] = -1  # MACD histogram bear 기준 더욱 완화
            thresholds["ma_diff_threshold_strong"] = 2.0  # MA diff 강한 추세 기준 강화
        elif "volatile" in market_regime:  # 변동성 약세장
            thresholds["rsi_overbought"] = 65  # RSI overbought 기준 약간 하향
            thresholds["rsi_oversold"] = 25  # RSI oversold 기준 약간 하향
            thresholds["cci_overbought"] = 120  # CCI overbought 기준 약간 하향
            thresholds["cci_oversold"] = -180  # CCI oversold 기준 약간 하향
            thresholds["atr_volatility_threshold_high"] = 0.03  # 높은 변동성 기준 상향
    elif "sideways" in market_regime:  # 횡보 regime
        if "volatile" in market_regime:  # 변동성 횡보장세
            thresholds["rsi_oversold"] = 35  # RSI oversold 기준 약간 상향
            thresholds["rsi_overbought"] = 65  # RSI overbought 기준 약간 하향
            thresholds["cci_oversold"] = -130  # CCI oversold 기준 약간 상향
            thresholds["cci_overbought"] = 130  # CCI overbought 기준 약간 하향
            thresholds["atr_volatility_threshold_high"] = 0.03  # 높은 변동성 기준 상향
        elif "consolidation" in market_regime:  # 수렴형 횡보장세
            thresholds["rsi_oversold"] = 25  # RSI oversold 기준 하향 (더욱 과매도)
            thresholds["rsi_overbought"] = 75  # RSI overbought 기준 상향 (더욱 과열)
            thresholds["cci_oversold"] = -180  # CCI oversold 기준 하향
            thresholds["cci_overbought"] = 180  # CCI overbought 기준 상향

    logging.info(f"Indicator thresholds adjusted for market regime: {market_regime}")
    return thresholds


def choose_primary_timeframe(multi_tf_data, market_regime):
    """
    시장 상태 및 변동성을 고려하여 메인 타임프레임을 선정한다.
    - 변동성이 높을 때는 단기 TF, 추세 추종 or 횡보장에는 1h/4h TF
    """
    if not multi_tf_data:
        return "1h"  # 데이터 없을 경우, 1시간봉 default

    if "volatile" in market_regime:  # 변동성 장세: 5m, 15m
        if "5m" in multi_tf_data and "15m" in multi_tf_data:
            logging.info("Primary TF: 15m (Volatile Market)")
            return "15m"  # 15분봉 우선 (단기 변동성 활용)
        elif "5m" in multi_tf_data:
            logging.info("Primary TF: 5m (Volatile Market, fallback to 5m)")
            return "5m"  # 5분봉 fallback
    elif "strong_momentum" in market_regime:  # 강력 추세 장세: 1h, 4h (추세 추종)
        if "4h" in multi_tf_data:
            logging.info("Primary TF: 4h (Strong Momentum Market)")
            return "4h"  # 4시간봉 우선 (추세 추종)
        elif "1h" in multi_tf_data:
            logging.info("Primary TF: 1h (Strong Momentum Market, fallback to 1h)")
            return "1h"  # 1시간봉 fallback
    else:  # 횡보 또는 점진적 추세: 1h (안정적 TF)
        if "1h" in multi_tf_data:
            logging.info("Primary TF: 1h (Sideways/Gradual Trend Market)")
            return "1h"  # 1시간봉 (default)
        elif "4h" in multi_tf_data:
            logging.info("Primary TF: 4h (Sideways/Gradual Trend Market, fallback to 4h)")
            return "4h"  # 4시간봉 fallback
    return "1h"  # 모든 조건 안 맞을 경우, 1시간봉 default


# =====================================================
# 8. GPT 프롬프트 생성 및 거래 결정 (최상위 트레이더 프롬프트)
# =====================================================
def generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data,
                        onchain_data, multi_tf_data, market_regime, thresholds,
                        heatmap_analysis, primary_tf):
    """
    GPT 프롬프트 생성 (최상위 개인 트레이더 수준의 매매 전략 및 판단 기준 적용)
    - XML 기반 상세 프롬프트 (계좌 정보, 시장 상황, 지표, 가이드라인, 전략, 룰 포함)
    - 시장 regime, 타임프레임, 데이터 유형별 가중치 및 우선순위 명시
    - TP/SL, 지정가 설정 기준 고도화 (피보나치, ATR Trailing Stop, 지지/저항선, 청산 맵 활용)
    - 다중 전략 시스템, 적응형 전략 배분, 고급 캔들/거래량/호가창 분석, 자금 관리 비법 반영
    """
    # Multi-timeframe summary (enhanced indicators 포함)
    multi_tf_summary = ""
    for tf, data in multi_tf_data.items():
        multi_tf_summary += (
            f"{tf} - Price: {data['current_price']}, VolChange: {data['volume_change']}%, VolSurge: {data['volume_surge']}, VolDrop: {data['volume_drop']}, "
            f"RSI: {data['rsi']}, CCI: {data['cci']}, EMA20: {data['ema20']}, EMA50: {data['ema50']} (Diff: {data['ema50_diff']}%), "
            f"EMA200: {data['ema200']} (Diff: {data['ema200_diff']}%), VWAP: {data['vwap']}, "
            f"BB Upper: {data['bb_upper']}, BB Lower: {data['bb_lower']}, BB Width: {data['bb_width']}, "
            f"MACD: {data['macd']} (Signal: {data['macd_signal']}, Hist: {data['macd_hist']}), ATR: {data['atr']}\n"
        )

    # Fear & Greed Index (classification, value)
    fng_class, fng_value = extended_data.get("fear_and_greed_index", ("N/A", "N/A"))

    prompt = f"""<TradeBotPrompt>
     <Persona>
         You are a world-class Bitcoin individual trader, known for your exceptional market timing and risk management.
         Your trading style is a blend of technical, on-chain, and sentiment analysis, with a focus on capturing high-probability trades in any market condition.
         You always prioritize risk management and aim for consistent, long-term profitability over high-risk, high-reward gambles.
     </Persona>

     <Account>
         <WalletBalance>{wallet_balance}</WalletBalance>
         <CurrentPosition>{position_info}</CurrentPosition>
     </Account>

     <MarketContext>
         <MarketRegime>{market_regime.upper()}</MarketRegime>
         <Timeframe>{primary_tf}</Timeframe>
         <Price>{multi_tf_data[primary_tf]['current_price']}</Price>
         <VolumeChange>{multi_tf_data[primary_tf]['volume_change']}</VolumeChange>
         <VolumeSurge>{multi_tf_data[primary_tf]['volume_surge']}</VolumeSurge>
         <VolumeDrop>{multi_tf_data[primary_tf]['volume_drop']}</VolumeDrop>
         <RSI>{multi_tf_data[primary_tf]['rsi']}</RSI>
         <CCI>{multi_tf_data[primary_tf]['cci']}</CCI>
         <EMA20>{multi_tf_data[primary_tf]['ema20']}</EMA20>
         <EMA50>{multi_tf_data[primary_tf]['ema50']}</EMA50>
         <EMA50Diff>{multi_tf_data[primary_tf]['ema50_diff']}</EMA50Diff>
         <EMA200>{multi_tf_data[primary_tf]['ema200']}</EMA200>
         <EMA200Diff>{multi_tf_data[primary_tf]['ema200_diff']}</EMA200Diff>
         <VWAP>{multi_tf_data[primary_tf]['vwap']}</VWAP>
         <BB_Upper>{multi_tf_data[primary_tf]['bb_upper']}</BB_Upper>
         <BB_Lower>{multi_tf_data[primary_tf]['bb_lower']}</BB_Lower>
         <BB_Width>{multi_tf_data[primary_tf]['bb_width']}</BB_Width>
         <MACD>{multi_tf_data[primary_tf]['macd']}</MACD>
         <MACD_Signal>{multi_tf_data[primary_tf]['macd_signal']}</MACD_Signal>
         <MACD_Hist>{multi_tf_data[primary_tf]['macd_hist']}</MACD_Hist>
         <ATR>{multi_tf_data[primary_tf]['atr']}</ATR>
     </MarketContext>

     <MultiTimeframeAnalysis>
         {multi_tf_summary}
     </MultiTimeframeAnalysis>

     <ExtendedMarketData>
         <OrderBook>
             <Bid>{aggregated_data['order_book']['bid']}</Bid>
             <Ask>{aggregated_data['order_book']['ask']}</Ask>
             <Spread>{aggregated_data['order_book']['spread']}</Spread>
         </OrderBook>
         <FundingRate>{extended_data['funding_rate']}</FundingRate>
         <OpenInterest>{extended_data['open_interest']}</OpenInterest>
         <OpenInterestChange>{extended_data['open_interest_change']}</OpenInterestChange>
         <LongShortRatio>{extended_data['long_short_ratio']}</LongShortRatio>
         <BTCDominance>{extended_data['btc_dominance']}</BTCDominance>
         <FutureSpotDiff>{extended_data['future_spot_diff']}</FutureSpotDiff>
         <FearGreedIndex>
             <Classification>{fng_class}</Classification>
             <Value>{fng_value}</Value>
         </FearGreedIndex>
         <ExchangeInflows>{extended_data['exchange_inflows']}</ExchangeInflows>
     </ExtendedMarketData>

     <OnChainData>
         <MVRV>{onchain_data['mvrv']}</MVRV>
         <SOPR>{onchain_data['sopr']}</SOPR>
     </OnChainData>

     <LiquidationHeatmapAnalysis>
         {heatmap_analysis}
     </LiquidationHeatmapAnalysis>

     <IndicatorThresholds>
         <RSI_Oversold>{thresholds['rsi_oversold']}</RSI_Oversold>
         <RSI_Overbought>{thresholds['rsi_overbought']}</RSI_Overbought>
         <CCI_Oversold>{thresholds['cci_oversold']}</CCI_Oversold>
         <CCI_Overbought>{thresholds['cci_overbought']}</CCI_Overbought>
         <MACD_HistStrongBull>{thresholds['macd_hist_strong_bull']}</MACD_HistStrongBull>
         <MACD_HistStrongBear>{thresholds['macd_hist_strong_bear']}</MACD_HistStrongBear>
         <MA_DiffThresholdMedium>{thresholds['ma_diff_threshold_medium']}</MA_DiffThresholdMedium>
         <MA_DiffThresholdStrong>{thresholds['ma_diff_threshold_strong']}</MA_DiffThresholdStrong>
         <ATR_VolatilityThresholdHigh>{thresholds['atr_volatility_threshold_high']}</ATR_VolatilityThresholdHigh>
         <ATR_VolatilityThresholdModerate>{thresholds['atr_volatility_threshold_moderate']}</ATR_VolatilityThresholdModerate>
     </IndicatorThresholds>

     <TradingStrategyAndRules>
         Based on the comprehensive market analysis provided, and acting as a world-class Bitcoin trader,
         formulate a detailed trading strategy.
         Consider the market regime, multi-timeframe analysis, extended market data, on-chain data,
         liquidation heatmap analysis, economic summary, and adjusted indicator thresholds.

         Your strategy should include:
         - Entry criteria: Specific conditions for opening a long or short position based on technical indicators, market context, and sentiment.
         - Exit criteria: Clear rules for taking profit and setting stop-loss orders. Consider using techniques like ATR trailing stops, Fibonacci levels, and liquidation zones for dynamic TP/SL placement.
         - Position sizing: Determine appropriate position size based on risk tolerance and market volatility. Do not risk more than 1% of your capital per trade.
         - Leverage: If using leverage, manage it prudently based on market volatility and conviction level. Start with low leverage and adjust based on performance and risk assessment.
         - Trade duration: Decide whether to aim for short-term scalps, day trades, or swing trades based on the market regime and chosen timeframe.
         - Adaptability: Explain how the strategy adapts to different market regimes (bull, bear, sideways, volatile, momentum).

         Example Strategy elements (adapt and expand upon these based on current market conditions):
         - In BULL_STRONG_MOMENTUM regime, focus on trend-following strategies. Look for dips to enter long positions when RSI approaches oversold levels (adjusted for bull market). Use 4h or 1h timeframe for entries.
         - In BEAR_STRONG_MOMENTUM regime, focus on counter-trend strategies or short selling rallies. Look for rallies to enter short positions when RSI approaches overbought levels (adjusted for bear market).
         - In SIDEWAYS_VOLATILE regime, consider range-bound trading strategies. Use Bollinger Bands and CCI to identify potential buy and sell zones. Use tighter stop-losses and take profits.
         - In SIDEWAYS_CONSOLIDATION regime, be cautious and consider waiting for a breakout. Analyze volume and order book for clues on potential direction.
     </TradingStrategyAndRules>

     <RiskManagement>
         Outline your risk management plan for this specific trade. As a world-class trader, you must prioritize capital preservation.
         - Maximum risk per trade: Adhere to a strict rule of risking no more than 1% of your total capital on any single trade.
         - Stop-loss strategy: Determine where to place stop-loss orders for both long and short positions. Consider using ATR-based stop-loss, or placing stops based on key support/resistance levels identified in multi-timeframe analysis or liquidation heatmap zones.
         - Take-profit strategy: Define your take-profit levels. Consider using Fibonacci extension levels, prior swing highs/lows, or reacting to overbought/oversold conditions. For trend trades, consider trailing stop-losses.
         - Emergency exit plan: Describe conditions that would trigger an immediate manual exit from the trade, regardless of pre-set TP/SL levels (e.g., unexpected black swan event, extreme volatility spike against your position).
         - Leverage management: If using leverage, ensure it is appropriate for the market conditions and your risk tolerance. Reduce leverage in volatile markets or when conviction is low.
     </RiskManagement>

     <DecisionGuidance>
         Based on the entirety of the analysis and strategy formulated above, provide a concise trading decision.
         Choose ONE from the following: BUY, SELL, HOLD, WAIT.
         Justify your decision in one sentence, briefly summarizing the key factors that led to your recommendation.
     </DecisionGuidance>

 </TradeBotPrompt>
 """
    return prompt


def get_gpt_trading_decision(prompt):
    """
    GPT 모델을 이용하여 트레이딩 결정을 생성한다.
    """
    try:
        sys_instruct = '''You are a world-class Bitcoin individual trader, known for your exceptional market timing and risk management.
         Your trading style is a blend of technical, on-chain, and sentiment analysis, with a focus on capturing high-probability trades in any market condition.
         You always prioritize risk management and aim for consistent, long-term profitability over high-risk, high-reward gambles.'''
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=sys_instruct),
            contents=[prompt])
        logging.info("GPT trading decision received")
        return response.text
        # response = gpt_client.chat.completions.create(
        #     model="gpt-4-turbo-preview",  # 최신 GPT-4 터보 모델 사용
        #     messages=[
        #         {"role": "system", "content": "You are a world-class cryptocurrency trading expert."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.5,  # temperature 조절 (0.0 ~ 1.0, 낮을수록 결정론적)
        #     max_tokens=1000,  # 응답 최대 토큰 수 제한
        #     frequency_penalty=0.0,  # frequency_penalty, presence_penalty 조절
        #     presence_penalty=0.0
        # )
        # decision_text = response.choices[0].message.content.strip()
        # logging.info("GPT trading decision received")
        # return decision_text
    except Exception as e:
        logging.error(f"Error generating GPT trading decision: {e}")
        return "N/A"


# =====================================================
# 9. 포지션 관리 및 자동 매매 (파일 기반 Logging)
# =====================================================
def record_decision_to_csv(decision_text, market_regime, primary_tf, thresholds, extended_data, onchain_data,
                           multi_tf_data, heatmap_analysis, econ_summary):
    """
    거래 결정을 CSV 파일에 기록한다. (결정 텍스트, 시장 regime, thresholds, 지표 데이터 포함)
    """
    timestamp_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")  # KST 타임존 적용

    # Fear & Greed Index (classification, value)
    fng_class, fng_value = extended_data.get("fear_and_greed_index", ("N/A", "N/A"))

    # 5m, 15m, 1h, 4h, 1d 데이터 요약 (current_price, rsi, ema50_diff, ema200_diff, macd_hist)
    tf_summary = {}
    for tf in ["5m", "15m", "1h", "4h", "1d"]:
        tf_data = multi_tf_data.get(tf, {})
        tf_summary[tf] = {
            "price": tf_data.get("current_price", "N/A"),
            "rsi": tf_data.get("rsi", "N/A"),
            "ema50_diff": tf_data.get("ema50_diff", "N/A"),
            "ema200_diff": tf_data.get("ema200_diff", "N/A"),
            "macd_hist": tf_data.get("macd_hist", "N/A")
        }

    # 온체인 데이터 (mvrv, sopr)
    onchain_summary = {
        "mvrv": onchain_data.get("mvrv", "N/A"),
        "sopr": onchain_data.get("sopr", "N/A")
    }

    header_row = ["Timestamp_KST", "Decision_Text", "Market_Regime", "Primary_TF", "RSI_Oversold", "RSI_Overbought",
                  "CCI_Oversold", "CCI_Overbought", "MACD_HistStrongBull", "MACD_HistStrongBear",
                  "MA_DiffThresholdMedium", "MA_DiffThresholdStrong", "ATR_VolatilityThresholdHigh",
                  "ATR_VolatilityThresholdModerate",
                  "Funding_Rate", "Open_Interest", "OI_Change_24h", "Long_Short_Ratio", "BTC_Dominance",
                  "Future_Spot_Diff",
                  "FNG_Classification", "FNG_Value", "Exchange_Inflows", "Liquidation_Heatmap_Analysis",
                  "Economic_Summary",
                  "5m_Price", "5m_RSI", "5m_EMA50_Diff", "5m_EMA200_Diff", "5m_MACD_Hist",
                  "15m_Price", "15m_RSI", "15m_EMA50_Diff", "15m_EMA200_Diff", "15m_MACD_Hist",
                  "1h_Price", "1h_RSI", "1h_EMA50_Diff", "1h_EMA200_Diff", "1h_MACD_Hist",
                  "4h_Price", "4h_RSI", "4h_EMA50_Diff", "4h_EMA200_Diff", "4h_MACD_Hist",
                  "1d_Price", "1d_RSI", "1d_EMA50_Diff", "1d_EMA200_Diff", "1d_MACD_Hist",
                  "OnChain_MVRV", "OnChain_SOPR"]

    data_row = [timestamp_kst, decision_text, market_regime, primary_tf, thresholds['rsi_oversold'],
                thresholds['rsi_overbought'],
                thresholds['cci_oversold'], thresholds['cci_overbought'], thresholds['macd_hist_strong_bull'],
                thresholds['macd_hist_strong_bear'],
                thresholds['ma_diff_threshold_medium'], thresholds['ma_diff_threshold_strong'],
                thresholds['atr_volatility_threshold_high'], thresholds['atr_volatility_threshold_moderate'],
                extended_data.get('funding_rate', 'N/A'), extended_data.get('open_interest', 'N/A'),
                extended_data.get('open_interest_change', 'N/A'),
                extended_data.get('long_short_ratio', 'N/A'), extended_data.get('btc_dominance', 'N/A'),
                extended_data.get('future_spot_diff', 'N/A'),
                fng_class, fng_value, extended_data.get('exchange_inflows', 'N/A'), heatmap_analysis, econ_summary,
                tf_summary['5m']['price'], tf_summary['5m']['rsi'], tf_summary['5m']['ema50_diff'],
                tf_summary['5m']['ema200_diff'], tf_summary['5m']['macd_hist'],
                tf_summary['15m']['price'], tf_summary['15m']['rsi'], tf_summary['15m']['ema50_diff'],
                tf_summary['15m']['ema200_diff'], tf_summary['15m']['macd_hist'],
                tf_summary['1h']['price'], tf_summary['1h']['rsi'], tf_summary['1h']['ema50_diff'],
                tf_summary['1h']['ema200_diff'], tf_summary['1h']['macd_hist'],
                tf_summary['4h']['price'], tf_summary['4h']['rsi'], tf_summary['4h']['ema50_diff'],
                tf_summary['4h']['ema200_diff'], tf_summary['4h']['macd_hist'],
                tf_summary['1d']['price'], tf_summary['1d']['rsi'], tf_summary['1d']['ema50_diff'],
                tf_summary['1d']['ema200_diff'], tf_summary['1d']['macd_hist'],
                onchain_summary['mvrv'], onchain_summary['sopr']]

    file_exists = os.path.isfile(DECISIONS_LOG_FILE)
    with open(DECISIONS_LOG_FILE, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(header_row)  # 파일 없으면 헤더 row 추가
        csv_writer.writerow(data_row)
    logging.info("Trading decision recorded to CSV")


def place_order(symbol, order_type, side, amount, price=None):
    """
    실제 거래소에 주문을 제출하는 함수 (미구현, paper trading or mock order)
    """
    # ** Caution: 실제 자동 매매 로직은 매우 신중하게 구현해야 합니다.**
    # 현재는 logging만 처리하고, 실제 주문은 simulation 또는 paper trading으로 대체.
    if order_type not in ["MARKET", "LIMIT"]:
        raise ValueError("Invalid order type. Choose MARKET or LIMIT.")
    if side not in ["BUY", "SELL"]:
        raise ValueError("Invalid order side. Choose BUY or SELL.")

    timestamp_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")  # KST 타임존
    order_details = {
        "timestamp_kst": timestamp_kst,
        "symbol": symbol,
        "order_type": order_type,
        "side": side,
        "amount": amount,
        "price": price if price else "MARKET_PRICE"
    }

    logging.info(f"Paper Order Placed: {order_details}")
    send_telegram_message(f"🚨 Paper Order Placed: {order_details}")  # 텔레그램 메시지 전송

    # TODO: 실제 거래소 연동 및 주문 제출 로직 (추후 구현, 매우 신중하게!)
    return order_details  # 주문 결과 (paper order detail) 반환


def update_position_status(order_detail, trade_result):
    """
    포지션 상태를 업데이트하고, open/closed position file에 기록한다.
    """
    timestamp_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")  # KST 타임존
    position_update_time = timestamp_kst

    if trade_result["order_status"] == "OPEN":  # 신규 포지션 open
        position_data = {
            "entry_timestamp_kst": position_update_time,
            "symbol": order_detail["symbol"],
            "order_type": order_detail["order_type"],
            "order_side": order_detail["side"],
            "entry_price": trade_result["entry_price"],  # or order_detail["price"] if limit order
            "amount": order_detail["amount"],
            "stop_loss": trade_result["stop_loss"],  # if set
            "take_profit": trade_result["take_profit"],  # if set
            "current_price": trade_result["current_price"],  # for tracking
            "unrealized_pnl": trade_result["unrealized_pnl"]
        }
        write_position_to_csv(position_data, OPEN_POSITIONS_FILE, mode='a')  # open positions 파일에 append
        logging.info(f"New position opened and recorded: {position_data}")
        send_telegram_message(f"✅ New Position Opened: {position_data}")

    elif trade_result["order_status"] == "CLOSED":  # 기존 포지션 close
        closed_position_data = {
            "exit_timestamp_kst": position_update_time,
            "symbol": order_detail["symbol"],
            "order_side": order_detail["side"],  # exit 방향 (BUY: 숏 청산, SELL: 롱 청산)
            "exit_price": trade_result["exit_price"],
            "pnl": trade_result["realized_pnl"],
            "pnl_percentage": trade_result["pnl_percentage"],
            "position_duration": trade_result["position_duration"]
        }
        write_position_to_csv(closed_position_data, CLOSED_POSITIONS_FILE, mode='a')  # closed positions 파일에 append
        delete_position_from_csv(order_detail, OPEN_POSITIONS_FILE)  # open positions 파일에서 해당 position 삭제
        logging.info(f"Position closed and recorded: {closed_position_data}")
        send_telegram_message(f"⛔️ Position Closed: {closed_position_data}")

    else:
        logging.error(f"Invalid trade result status: {trade_result['order_status']}")


def write_position_to_csv(position_data, filename, mode='w'):
    """
    포지션 데이터를 CSV 파일에 기록하는 helper 함수 (open/closed positions 파일에 공통 사용)
    """
    file_exists = os.path.isfile(filename)
    header_row = ["timestamp_kst", "symbol", "order_type", "order_side", "entry_price", "amount",
                  "stop_loss", "take_profit", "current_price", "unrealized_pnl",
                  "exit_timestamp_kst", "exit_price", "pnl", "pnl_percentage",
                  "position_duration"]  # 모든 column header 포함

    with open(filename, mode=mode, newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=header_row)
        if mode == 'w' or not file_exists:
            csv_writer.writeheader()  # 파일 없거나, write mode ('w') 이면 header
        csv_writer.writerow(position_data)


def delete_position_from_csv(order_detail, filename):
    """
    Open positions CSV 파일에서 특정 포지션을 삭제한다. (symbol, side, entry_price 기준으로 unique position 식별)
    """
    temp_positions = []
    deleted = False
    try:
        with open(filename, mode='r', newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                if not (row['symbol'] == order_detail['symbol'] and row['order_side'] == order_detail['side']):
                    temp_positions.append(row)  # keep positions not matching delete 조건
                else:
                    deleted = True  # position 찾아서 delete

        if deleted:  # delete 성공 시, update 파일 write
            header_row = csv_reader.fieldnames
            with open(filename, mode='w', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=header_row)
                csv_writer.writeheader()
                csv_writer.writerows(temp_positions)
            logging.info(f"Position deleted from {filename}: {order_detail}")
        else:
            logging.warning(f"Position not found in {filename} for deletion: {order_detail}")

    except Exception as e:
        logging.error(f"Error deleting position from {filename}: {e}")


def get_current_positions():
    """
    Open positions CSV 파일에서 현재 open position list를 읽어온다.
    """
    current_positions = []
    if not os.path.exists(OPEN_POSITIONS_FILE):
        return current_positions  # 파일 없으면 empty list return

    try:
        with open(OPEN_POSITIONS_FILE, mode='r', newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                current_positions.append(row)
    except Exception as e:
        logging.error(f"Error reading current positions from CSV: {e}")
    return current_positions


def get_position_info():
    """
    현재 보유 포지션에 대한 요약 정보 (count, 총 amount, 평균 entry price, unrealized PNL 등) 를 계산한다.
    """
    positions = get_current_positions()
    if not positions:
        return "No open positions"  # No positions

    total_amount = 0
    total_value = 0  # current value 기준 총 평가금액
    total_unrealized_pnl = 0
    position_count = len(positions)

    for pos in positions:
        amount = float(pos['amount'])
        entry_price = float(pos['entry_price'])
        current_price = float(pos['current_price'])  # from real-time price
        unrealized_pnl = amount * (current_price - entry_price) if pos['order_side'] == 'BUY' else amount * (
                entry_price - current_price)

        total_amount += amount
        total_value += amount * current_price
        total_unrealized_pnl += unrealized_pnl

    avg_entry_price = total_value / total_amount if total_amount else 0
    pnl_percentage = (total_unrealized_pnl / (total_amount * avg_entry_price)) * 100 if avg_entry_price else 0

    position_summary = (
        f"Current Positions: {position_count} | "
        f"Total Amount: {total_amount:.4f} {SYMBOL} | "
        f"Avg Entry Price: ${avg_entry_price:.2f} | "
        f"Unrealized PNL: ${total_unrealized_pnl:.2f} ({pnl_percentage:.2f}%)"
    )
    logging.info("Current Position Summary: " + position_summary)
    return position_summary


# =====================================================
# 10. 메인 실행 함수 (자동 트레이딩 Bot)
# =====================================================
def main():
    logging.info("==================== Trading Bot Started (Single Run) ====================")

    wallet_balance = 10000  # 초기 자본금 (USDT)

    # 1. Data Aggregation & Analysis
    multi_tf_data = fetch_multi_tf_data(SYMBOL, timeframes=TIMEFRAMES)
    if not multi_tf_data:
        logging.error("Multi-timeframe data fetch failed. Exiting.")
        return  # 데이터 수집 실패 시, 프로그램 종료

    aggregated_data = {
        "order_book": fetch_order_book(SYMBOL)
    }
    extended_data = {
        "funding_rate": fetch_funding_rate(SYMBOL),
        "open_interest": fetch_open_interest(SYMBOL),
        "open_interest_change": fetch_open_interest_change(SYMBOL),
        "long_short_ratio": fetch_long_short_ratio(),
        "btc_dominance": fetch_btc_dominance(),
        "future_spot_diff": fetch_future_spot_diff(SYMBOL),
        "fear_and_greed_index": fetch_fear_and_greed_index(),
        "exchange_inflows": fetch_exchange_inflows()
    }
    onchain_data = fetch_onchain_data(SYMBOL)
    fetch_liquidation_heatmap()  # 청산 히트맵 다운로드 (한 번만 실행)
    heatmap_analysis = analyze_liquidation_heatmap()

    # 2. 시장 Regime 결정 및 Threshold 조정, Primary TF 선정
    market_regime = determine_market_regime(multi_tf_data, onchain_data, extended_data)
    thresholds = adjust_indicator_thresholds(market_regime)
    primary_tf = choose_primary_timeframe(multi_tf_data, market_regime)

    # 3. GPT 프롬프트 생성 및 Trading Decision
    position_info = get_position_info()  # 현재 position summary
    prompt = generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data,
                                    onchain_data, multi_tf_data, market_regime, thresholds,
                                    heatmap_analysis, primary_tf)
    decision_text = get_gpt_trading_decision(prompt)

    if decision_text != "N/A":
        record_decision_to_csv(decision_text, market_regime, primary_tf, thresholds, extended_data,
                                onchain_data, multi_tf_data, heatmap_analysis, econ_summary)
        send_telegram_message(
            f"🔮 Trading Decision (Regime: {market_regime.upper()}, TF: {primary_tf}):\n{decision_text}")

        # Paper order placement (예시)
        if "BUY" in decision_text.upper():
            order_detail = place_order(SYMBOL, "MARKET", "BUY", amount=0.01)
            trade_result = {  # Mock trade result
                "order_status": "OPEN", "entry_price": multi_tf_data[primary_tf]['current_price'],
                "stop_loss": None, "take_profit": None,
                "current_price": multi_tf_data[primary_tf]['current_price'], "unrealized_pnl": 0
            }
            update_position_status(order_detail, trade_result)

        elif "SELL" in decision_text.upper():
            order_detail = place_order(SYMBOL, "MARKET", "SELL", amount=0.01)
            trade_result = {  # Mock trade result
                "order_status": "OPEN", "entry_price": multi_tf_data[primary_tf]['current_price'],
                "stop_loss": None, "take_profit": None,
                "current_price": multi_tf_data[primary_tf]['current_price'], "unrealized_pnl": 0
            }
            update_position_status(order_detail, trade_result)
        else:
            logging.info(f"Decision is HOLD or WAIT. No order placed.")

    else:
        logging.warning("GPT decision generation failed.")

    logging.info("==================== Trading Bot Finished (Single Run) ====================")


if __name__ == "__main__":
    main()
