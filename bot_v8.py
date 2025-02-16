import os
import re
import ta
import csv
import time
import ccxt
import pytz
import logging
import requests
import pandas_ta

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
HYPE_SYMBOL = "BTC/USDC:USDC"
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

# Hyperliquid 거래소 객체 생성 (환경변수에서 API 키 로드)
HYPERLIQUID_WALLET_ADDRESS = os.environ.get('HYPE_ADDRESS')
HYPERLIQUID_PRIVATE_KEY = os.environ.get('HYPE_PRIVATE_KEY')

exchange = ccxt.hyperliquid({
    'walletAddress': HYPERLIQUID_WALLET_ADDRESS,
    'privateKey': HYPERLIQUID_PRIVATE_KEY,
    'options': {
        'defaultType': 'swap',  # 선물 거래를 위한 설정
    },
})


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
def get_driver():
    """
    undetected_chromedriver의 새로운 인스턴스를 생성하여 반환한다.
    """
    options = uc.ChromeOptions()
    # Configure options if needed (e.g., headless mode)
    # options.add_argument('--headless') # 필요에 따라 headless 모드 활성화
    driver = uc.Chrome(options=options)
    return driver


# =====================================================
# 4. 데이터 수집 및 기술적 지표 계산
# =====================================================
def fetch_ohlcv(symbol, timeframe, limit=300):
    """
    ccxt를 사용하여 OHLCV 데이터를 가져옵니다. Hyperliquid 거래소 사용.
    """
    try:
        exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        logging.info(f"{symbol} / {timeframe} OHLCV data fetched successfully from Hyperliquid")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(KST)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logging.error(f"Error fetching {symbol} / {timeframe} OHLCV data from Hyperliquid: {e}")
        return None


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
        future_ticker = exchange.fetch_ticker(symbol.replace('/', ''))  # 선물 심볼 가정

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
        driver.quit()
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
        driver.quit()

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
        funding_info = exchange.fetch_funding_rate(symbol=symbol)
        latest_funding = funding_info['info']['funding'] if 'info' in funding_info else None
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
        oi_response = exchange.fetch_open_interest(symbol=symbol)
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


def fetch_onchain_data():
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

        driver.quit()
        logging.info("On-chain data (MVRV, SOPR) fetched successfully")
        return {"mvrv": mvrv, "sopr": sopr}
    except Exception as e:
        logging.error(f"Error fetching on-chain data: {e}")
        return {"mvrv": "N/A", "sopr": "N/A"}


def calculate_donchian_channel(df, window=20):
    """
    Donchian Channels를 계산하여 박스권 상단/하단을 반환합니다.
    """
    df['donchian_upper'] = df['high'].rolling(window=window).max()
    df['donchian_lower'] = df['low'].rolling(window=window).min()
    df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2  # 중간값 계산
    return df


def fetch_multi_tf_data(symbol, timeframes=None, limit=300, thresholds=None):
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
        df = calculate_volume_divergence(df)

        # Donchian Channels 계산 (thresholds에서 window 가져옴)
        window = thresholds.get(f'donchian_window_{tf}', 20) if thresholds else 20  # thresholds 없으면 기본값 20
        df = calculate_donchian_channel(df, window=window)

        latest = df.iloc[-1]
        multi_tf_data[tf] = {
            "current_price": round(latest['close'], 2),
            "rsi": round(latest['rsi'], 2) if not np.isnan(latest['rsi']) else None,
            "ema20": round(latest['ema20'], 2) if not np.isnan(latest['ema20']) else None,  # EMA 추가
            "ema50": round(latest['ema50'], 2) if not np.isnan(latest['ema50']) else None,  # EMA 추가
            "ema200": round(latest['ema200'], 2) if not np.isnan(latest['ema200']) else None,  # EMA 추가
            "ema50_diff": round(latest['ema50_diff'], 2) if not np.isnan(latest['ema50_diff']) else None,
            # EMA diff
            "ema200_diff": round(latest['ema200_diff'], 2) if not np.isnan(latest['ema200_diff']) else None,
            # EMA diff
            "bb_upper": round(latest['bb_upper'], 2) if not np.isnan(latest['bb_upper']) else None,
            "macd": round(latest['macd'], 2) if not np.isnan(latest['macd']) else None,
            "donchian_upper": round(latest['donchian_upper'], 2),
            "donchian_lower": round(latest['donchian_lower'], 2),
            "donchian_middle": round(latest['donchian_middle'], 2),  # 중간값 추가
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
# 6. 청산맵 다운로드
# =====================================================
def fetch_liquidation_map():
    """
    청산 히트맵 데이터를 CoinAnk 사이트에서 다운로드한다.
    """
    url = "https://coinank.com/liqMapChart"
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
    더 세분화된 시장 상태를 결정한다. + 골든/데드 크로스 추가
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

    # Golden Cross/Dead Cross (1시간봉 기준)
    golden_cross_1h = ema50_1h > ema200_1h and tf_1h_data['df_full']['ema50'].iloc[-2] <= \
                      tf_1h_data['df_full']['ema200'].iloc[-2]
    dead_cross_1h = ema50_1h < ema200_1h and tf_1h_data['df_full']['ema50'].iloc[-2] >= \
                    tf_1h_data['df_full']['ema200'].iloc[-2]

    # 1. 추세 기반 Regime 판단 (EMA, Price 위치 관계, Golden/Dead Cross)
    if price_1h > ema50_1h and price_1h > ema200_1h and price_4h > ema200_4h and price_1d > ema200_1d:
        regime = "bull_trend"  # 강세 추세
        if golden_cross_1h:  # 골든크로스 발생 시, 더 강한 상승 신호
            regime = "bull_trend_strong"
    elif price_1h < ema50_1h and price_1h < ema200_1h and price_4h < ema200_4h and price_1d < ema200_1d:
        regime = "bear_trend"  # 약세 추세
        if dead_cross_1h:
            regime = "bear_trend_strong"
    else:
        regime = "sideways_normal_volatility"  # 횡보 (EMA 밀집 or 혼조세)

    donchian_width_1h = tf_1h_data['donchian_upper'] - tf_1h_data['donchian_lower']
    donchian_width_percent_1h = (donchian_width_1h / tf_1h_data['current_price']) * 100

    if "sideways" in regime:  # 횡보로 이미 판단된 경우에만
        if donchian_width_percent_1h < 3:
            regime = "tight_sideways"  # 강한 횡보 (좁은 박스권)
        elif donchian_width_percent_1h > 7:
            regime = "wide_sideways"  # 변동성이 큰 횡보 (넓은 박스권)

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
    시장 상태에 따라 RSI, MACD, MA, Donchian Channel 등의 임계값을 동적으로 조정.
    """
    thresholds = {}

    if "bull_trend" in market_regime:
        if "high_volatility" in market_regime:
            thresholds = {
                "rsi_oversold": 40,
                "rsi_overbought": 85,
                "rsi_trend_follow": 50,
                "macd_signal_cross_lookback": 5,
                "ema50_correction_percent": 2.0,
                "atr_stop_loss_multiplier": 2.5,
                "atr_take_profit_multiplier": 3.5,
                "donchian_window_5m": 20,  # 예시
                "donchian_window_15m": 20,
                "donchian_window_1h": 25,
                "donchian_window_4h": 50,
                "donchian_window_1d": 50
            }
        else:  # bull_trend + normal_volatility
            thresholds = {
                "rsi_oversold": 45,
                "rsi_overbought": 80,
                "rsi_trend_follow": 55,
                "macd_signal_cross_lookback": 3,
                "ema50_correction_percent": 1.5,
                "atr_stop_loss_multiplier": 2.0,
                "atr_take_profit_multiplier": 3.0,
                "donchian_window_5m": 20,  # 예시
                "donchian_window_15m": 20,
                "donchian_window_1h": 25,
                "donchian_window_4h": 50,
                "donchian_window_1d": 50
            }
    elif "bear_trend" in market_regime:
        if "high_volatility" in market_regime:
            thresholds = {
                "rsi_oversold": 15,
                "rsi_overbought": 60,
                "rsi_trend_follow": 50,
                "macd_signal_cross_lookback": 5,
                "ema50_bounce_percent": 2.0,
                "atr_stop_loss_multiplier": 2.5,
                "atr_take_profit_multiplier": 3.5,
                "donchian_window_5m": 20,  # 예시
                "donchian_window_15m": 20,
                "donchian_window_1h": 25,
                "donchian_window_4h": 50,
                "donchian_window_1d": 50
            }
        else:  # bear_trend + normal_volatility
            thresholds = {
                "rsi_oversold": 20,
                "rsi_overbought": 55,
                "rsi_trend_follow": 45,
                "macd_signal_cross_lookback": 3,
                "ema50_bounce_percent": 1.5,
                "atr_stop_loss_multiplier": 2.0,
                "atr_take_profit_multiplier": 3.0,
                "donchian_window_5m": 20,  # 예시
                "donchian_window_15m": 20,
                "donchian_window_1h": 25,
                "donchian_window_4h": 50,
                "donchian_window_1d": 50
            }
    elif "sideways" in market_regime:  # Sideways Regime
        if "high_volatility" in market_regime:
            thresholds = {
                "rsi_oversold": 35,
                "rsi_overbought": 75,
                "rsi_reversal": 40,
                "macd_histogram_divergence_lookback": 7,
                "bb_band_bounce_percent": 1.0,
                "atr_stop_loss_multiplier": 1.5,
                "atr_take_profit_multiplier": 2.0,
                "donchian_window_5m": 15,  # 더 짧은 기간
                "donchian_window_15m": 15,
                "donchian_window_1h": 20,
                "donchian_window_4h": 20,
                "donchian_window_1d": 20
            }
        elif "tight_sideways" in market_regime:  # tight_sideways 추가
            thresholds = {
                "rsi_oversold": 25,  # 더 민감하게
                "rsi_overbought": 75,
                "rsi_reversal": 40,
                "macd_histogram_divergence_lookback": 5,
                "bb_band_bounce_percent": 0.5,  # 더 작은 반등에도 반응
                "atr_stop_loss_multiplier": 1.0,  # 타이트하게
                "atr_take_profit_multiplier": 1.5,
                "donchian_window_5m": 10,  # 매우 짧은 기간
                "donchian_window_15m": 10,
                "donchian_window_1h": 15,
                "donchian_window_4h": 15,
                "donchian_window_1d": 20  # 조금 더 길게
            }
        elif "wide_sideways" in market_regime:  # wide_sideways 추가
            thresholds = {
                "rsi_oversold": 35,
                "rsi_overbought": 65,  # 덜 민감하게
                "rsi_reversal": 45,
                "macd_histogram_divergence_lookback": 7,
                "bb_band_bounce_percent": 1.2,  # 더 큰 반등을 기다림
                "atr_stop_loss_multiplier": 1.8,
                "atr_take_profit_multiplier": 2.5,  # 더 넓게
                "donchian_window_5m": 20,  # 더 긴 기간
                "donchian_window_15m": 25,
                "donchian_window_1h": 30,
                "donchian_window_4h": 50,
                "donchian_window_1d": 50
            }
        else:  # sideways + normal_volatility
            thresholds = {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "rsi_reversal": 45,
                "macd_histogram_divergence_lookback": 5,
                "bb_band_bounce_percent": 0.8,
                "atr_stop_loss_multiplier": 1.2,
                "atr_take_profit_multiplier": 1.8,
                "donchian_window_5m": 15,
                "donchian_window_15m": 20,
                "donchian_window_1h": 20,
                "donchian_window_4h": 25,
                "donchian_window_1d": 25
            }
    else:  # Default Thresholds (for unexpected regime)
        thresholds = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_stop_loss_multiplier": 1.5,
            "atr_take_profit_multiplier": 2.0,
            "donchian_window_5m": 20,  # 기본값
            "donchian_window_15m": 20,
            "donchian_window_1h": 20,
            "donchian_window_4h": 20,
            "donchian_window_1d": 20
        }

    logging.info(f"Indicator thresholds set for market regime: {market_regime}")
    return thresholds


def choose_primary_timeframe(market_regime):
    """
    시장 Regime 및 전략에 따라 Primary Timeframe 선정 로직 변경.
    """
    if "bull_trend_strong" in market_regime:
        return "1d"  # 강한 상승 추세: 일봉
    elif "bull_trend" in market_regime:
        return "4h"  # 상승 추세: 4시간봉
    elif "bear_trend_strong" in market_regime:
        return "1d"  # 강한 하락 추세: 일봉
    elif "bear_trend" in market_regime:
        return "4h"  # 하락 추세: 4시간봉
    elif "sideways_wide" in market_regime:
        return "1h"  # 넓은 횡보: 1시간봉
    elif "sideways_normal_volatility" in market_regime:
        return "15m"  # 일반 횡보: 15분봉
    elif "tight_sideways" in market_regime:
        return "5m"  # 좁은 횡보: 5분봉
    else:  # Default timeframe
        return "1h"  # 1시간봉 - default


# =====================================================
# 8. Gemini 프롬프트 생성 및 거래 결정
# =====================================================
def generate_gemini_prompt(extended_data,
                           onchain_data, multi_tf_data, market_regime, thresholds,
                           econ_summary, primary_tf, current_session,
                           fake_breakout_info, session_volatility_info):
    """
    Gemini Pro 모델에 전달할 Prompt 생성.
    """

    # Multi-Timeframe Analysis Summary (간결하게 유지)
    tf_summary_lines = []
    for tf, data in multi_tf_data.items():
        tf_summary_lines.append(
            f"**{tf}:** Price: {data['current_price']:.2f}, RSI: {data['rsi']:.2f}, "
            f"EMA50 Diff: {data['ema50_diff']:.2f}%, Donchian: ({data['donchian_lower']:.2f}-{data['donchian_middle']:.2f}-{data['donchian_upper']:.2f})"
            # middle 추가
        )
    multi_tf_summary = "\n".join(tf_summary_lines)

    fng_class, fng_value = extended_data.get("fear_and_greed_index", ("N/A", "N/A"))
    order_book_data = extended_data.get('order_book', {})

    prompt_text_1 = f"""
**Objective:** Make optimal trading decisions for BTC/USDT based on the provided market data.

**Market Context:**
- Regime: **{market_regime.upper()}**
- Primary Timeframe: **{primary_tf}**
- Session (KST): **{current_session}**
- Confidence: {get_timeframe_agreement(multi_tf_data, market_regime)['confidence_level']}

**Technical Analysis Summary:**
{multi_tf_summary}

**Additional Market Data:**
- Funding Rate: {extended_data.get('funding_rate', 'N/A')}
- Open Interest: {extended_data.get('open_interest', 'N/A')}
- Order Book: Bid={order_book_data.get('bid', 'N/A')}, Ask={order_book_data.get('ask', 'N/A')}, Spread={order_book_data.get('spread', 'N/A')}
- Exchange Inflow: {extended_data.get('exchange_inflows', 'N/A')}
- Fear & Greed: {fng_class} ({fng_value})
- On-Chain: MVRV={onchain_data.get('mvrv', 'N/A')}, SOPR={onchain_data.get('sopr', 'N/A')}
- Economic Events: {econ_summary}
- Spot-Future Diff: {extended_data.get('spot_future_price_diff', 'N/A')}%
- Bitcoin Dominance: {extended_data.get('bitcoin_dominance', 'N/A')}%
- Fake Breakout: {fake_breakout_info}
- Session Volatility: {session_volatility_info}"""

    prompt_text_2 = f"""**Liquidation Map Analysis Guide(Image Provided):**
- **Support and Resistance:**  "Treat large liquidation clusters as *potential* support (for longs) and resistance (for shorts).  However, *emphasize* that once these levels are breached, they can trigger rapid price movements."
- **Cascading Liquidations:** "If liquidations are stacked closely together, explain the *risk* of cascading liquidations.  If the price breaks through one level, it's likely to trigger a chain reaction, leading to a larger move."
- **Volatility Prediction:**  "The *density* and *proximity* of liquidation levels to the current price are indicators of potential volatility.  Closely stacked, large liquidation levels near the current price suggest *high* potential volatility."
- **Trade Entry/Exit:** "Use the liquidation levels to inform potential trade entry and exit points.  For example, a long entry might be placed slightly *above* a large cluster of long liquidations (acting as support), with a stop-loss *below* that cluster. A short entry might be placed slightly *below* a cluster of short liquidations (acting as resistance), with a stop loss *above*."
- **Risk Assessment:** "Compare the overall size and distribution of long vs. short liquidations to assess which side (longs or shorts) faces greater risk."
- **Combine with Other Data:** "Crucially, *integrate* the liquidation map analysis with the other market data (technical indicators, on-chain data, etc.) to make a final decision. The liquidation map is a *high-priority* input, but it's not the only factor."
- **Invalidation Levels**: "Identify price that if reached, would invalidate your trade thesis. This should be heavily informed by the liquidation data. (If price runs to X, the trade no longer makes sense.)"

**Indicator Guidelines:**

| Indicator         | Priority | Bull Trend                                                                                                                                                                                             | Bear Trend                                                                                                                                                                                             | Sideways (Tight)                                                                                                                                        | Sideways (Wide)                                                                                                                                          |
|-------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| RSI               | High     | Oversold: <={thresholds.get('rsi_oversold', 30)}, Overbought: >={thresholds.get('rsi_overbought', 70)}, Use RSI to confirm momentum and identify potential continuation or pullback entries. Trend Follow: {thresholds.get('rsi_trend_follow', 50)}. | Oversold: <={thresholds.get('rsi_oversold', 30)}, Overbought: >={thresholds.get('rsi_overbought', 70)}, Use RSI to confirm momentum and identify potential continuation or pullback entries. Trend Follow: {thresholds.get('rsi_trend_follow', 50)}. | Oversold: <={thresholds.get('rsi_oversold', 25)}, Overbought: >={thresholds.get('rsi_overbought', 75)}, Use RSI for reversal signals at extremes. Reversal: {thresholds.get('rsi_reversal', 40)}. | Oversold: <={thresholds.get('rsi_oversold', 35)}, Overbought: >={thresholds.get('rsi_overbought', 65)}, Use RSI for reversal signals at extremes. Reversal: {thresholds.get('rsi_reversal', 45)}. |
| EMA (50, 200)    | High     | Price above both EMA50 and EMA200 indicates a bullish trend. Look for pullbacks to the EMA50 as potential buying opportunities. A Golden Cross (EMA50 crossing above EMA200) confirms a bullish trend.   | Price below both EMA50 and EMA200 indicates a bearish trend. Look for rallies to the EMA50 as potential selling opportunities. A Dead Cross (EMA50 crossing below EMA200) confirms a bearish trend.   | Price oscillating around EMA50 indicates sideways movement. Frequent EMA crossovers may occur.                                                                 | Price oscillating around EMA50 indicates sideways movement. Frequent EMA crossovers may occur.                                                                 |
| MACD              | Medium   | Use signal line crossovers for potential entry/exit signals, particularly in early trend stages.                                                                                                    | Use signal line crossovers for potential entry/exit signals, particularly in early trend stages.                                                                                                    | Use histogram divergence from price action to signal potential reversals. Lookback period for histogram divergence: {thresholds.get('macd_histogram_divergence_lookback', 5)} candles.          | Use histogram divergence from price action to signal potential reversals. Lookback period for histogram divergence: {thresholds.get('macd_histogram_divergence_lookback', 7)} candles.         |
| Bollinger Bands   | Medium   | In strong trends, price may "walk" along the upper (bullish) or lower (bearish) band.                                                                                                                 | In strong trends, price may "walk" along the upper (bullish) or lower (bearish) band.                                                                                                                 | Bounces off Bollinger Bands can signal mean reversion opportunities. Look for price to revert to the mean (middle band) after touching the upper or lower band. Band Bounce Percent: ±{thresholds.get('bb_band_bounce_percent', 0.5)}%. | Bounces off Bollinger Bands can signal mean reversion opportunities.  Band Bounce Percent: ±{thresholds.get('bb_band_bounce_percent', 1.2)}%.       |
| Donchian Channel  | High     | N/A                                                                                                                                                                                                    | N/A                                                                                                                                                                                                    | **Primary indicator for sideways markets.** Buy near the lower channel, sell near the upper channel.                                                       | **Primary indicator for sideways markets.** Buy near the lower channel, sell near the upper channel.                                                        |
| Volume            | Medium   | Increasing volume confirms breakouts in the direction of the trend.                                                                                                                                 | Increasing volume confirms breakdowns in the direction of the trend.                                                                                                                                | Look for divergences between price and volume. Price rising with decreasing volume (bearish divergence) may indicate a weakening uptrend.                  | Look for divergences between price and volume. Price falling with decreasing volume (bullish divergence) may indicate a weakening downtrend.                 |
| ATR               | High     | Use ATR to assess volatility and adjust leverage/stop-loss accordingly. Higher ATR suggests higher volatility.                                                                                    | Use ATR to assess volatility and adjust leverage/stop-loss accordingly. Higher ATR suggests higher volatility.                                                                                    | Use ATR to gauge the width of the trading range.                                                                                                           | Use ATR to gauge the width of the trading range.                                                                                                           |

**Session Strategies (KST):**

*   **OVERNIGHT (00:00-08:00):** Low liquidity. *Prioritize risk management*. Watch for fake breakouts. Use tighter stops, lower leverage.
*   **ASIAN (08:00-16:00):** Medium volatility.
    *   08:00-09:00: Potential volatility spike. Look for strong directional moves *with increasing volume*.
    *   After 09:00:
        *   If a clear trend develops: Follow the trend using EMAs, RSI, and price action on 1h/4h timeframes.
        *   If sideways/range-bound: Employ mean reversion strategies using Bollinger Bands, Donchian Channel, and RSI on 15m/1h timeframes. Look for reversals at key levels.
*   **LONDON (16:00-22:00):** High liquidity.
    *   16:00 Open: Expect high volatility and potential breakouts/breakdowns.
    *   After the initial volatility, identify and trade the dominant trend. Use 1h/4h timeframes.
*   **US (22:00-06:00):** Highest volume and volatility.
    *   Often follows the London session trend, but *be prepared for reversals*, especially around key support/resistance levels.
    *   22:30-23:30: Economic news releases. *Avoid new entries immediately before/after major news releases. Wait for the market to digest the news and establish a clear direction.*
*   **TRANSITION (06:00-08:00):** Cautious trading. Low liquidity, potential trend formation before the Asian open.

**Regime-Specific Strategy Guidelines (within Sessions):**

- **Bull Trend Regime:**
    - Strategy: Trend Following, Buy the Dip.
    - Entry Signals:
        - Price above EMA50 and EMA200 on 1h, 4h, and 1d timeframes.
        - Golden Cross (EMA50 crosses above EMA200) on 1h or 4h timeframe.
        - RSI pullback to {thresholds.get('rsi_trend_follow', 50)} level.
        - Price correction to EMA50 support.
        - Bullish candle patterns at EMA50 support.
        - Exit/Take Profit: ATR Take Profit Multiplier: {thresholds.get('atr_take_profit_multiplier', 3.0)}x ATR from entry, resistance levels.
    - Stop Loss: ATR Stop Loss Multiplier: {thresholds.get('atr_stop_loss_multiplier', 2.0)}x ATR from entry, below EMA50 support.

- **Bear Trend Regime:**
    - Strategy: Trend Following, Sell the Rally.
    - Entry Signals:
        - Price below EMA50 and EMA200 on 1h, 4h, and 1d timeframes.  <-- EMA 조건 강화
        - Dead Cross (EMA50 crosses below EMA200) on 1h or 4h timeframe. <-- 데드 크로스 추가
        - RSI rebound to {thresholds.get('rsi_trend_follow', 50)} level.
        - Price bounce off EMA50 resistance.
        - Bearish candle patterns at EMA50 resistance.
    - Exit/Take Profit: ATR Take Profit Multiplier: {thresholds.get('atr_take_profit_multiplier', 3.0)}x ATR from entry, support levels.
    - Stop Loss: ATR Stop Loss Multiplier: {thresholds.get('atr_stop_loss_multiplier', 2.0)}x ATR from entry, above EMA50 resistance.

- **Sideways Regime:**
    - Strategy: Mean Reversion, Range Trading.
    - Entry Signals:
        - Price near Donchian Channel boundaries (Upper: {data['donchian_upper']:.2f}, Lower: {data['donchian_lower']:.2f}).  **<-- MOST IMPORTANT**
        - RSI Reversal from extremes (Overbought >=70 or Oversold <=30). Use 'rsi_reversal' threshold: {thresholds.get('rsi_reversal', 45)}.
        - MACD Histogram Divergence. (Optional)
        - Bollinger Band bounces (±{thresholds.get('bb_band_bounce_percent', 0.8)}% from bands). (Optional)
        - Order Book Imbalance suggests potential reversal (if available). (Optional)
    - Exit/Take Profit:
        - Donchian Channel Middle.
        - Opposite Donchian Channel boundary.
        - Bollinger Band Middle Band (EMA20).
    - Stop Loss:
        - Beyond Donchian Channel boundaries
    - **Important Note:** In a sideways regime, *prioritize entries near the Donchian Channel boundaries*.  Look for *quick reversals* (scalping) or *short-term mean reversion*.

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
    - For Sideways:
        - If the price is near the lower Bollinger Band, place the buy limit order a few ticks above the lower band.
        - If the price is near the upper Bollinger Band, place the sell limit order a few ticks below the upper band.
    - Monitor Order Book: Check bid/ask spread and liquidity when setting limit orders.


**Task:**

Based on the comprehensive market analysis and guidelines above, decide the best course of action: **GO LONG, GO SHORT, HOLD LONG, HOLD SHORT, or NO TRADE.**

Choose **NO TRADE** if:
- *Major* conflicting signals across *multiple* timeframes, and no clear signal on the primary timeframe ({primary_tf}).
- *Extremely* low timeframe agreement confidence (e.g., Low or Very Low).
- *No clear trading opportunity based on the defined strategies.*  <-- IMPORTANT
- Imminent major economic news release.
- Suspected fake breakout (unless trading reversals).

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

**Example Outputs:**

* **Bull Trend, GO LONG:**
GO LONG, 5x, 1d, 48500.00, 46000.00, 47050.00, Price above EMA50 and EMA200 on 1h, 4h, and 1d. Golden Cross on 1h. RSI is above 50 on 1h and 4h, confirming bullish momentum.


* **Bear Trend, GO SHORT:**
GO SHORT, 3x, 6h, 45000.00, 47500.00, 46800.00, Price below EMA50 and EMA200 on multiple timeframes. Bearish engulfing candle on 1h.


* **Fake Breakout, NO TRADE:**
NO TRADE, N/A, N/A, N/A, N/A, N/A, Suspected bullish fake breakout on 5m. Waiting for confirmation.

"""

    return prompt_text_1, prompt_text_2


def generate_trading_decision(extended_data,
                              onchain_data, multi_tf_data, market_regime, thresholds,
                              econ_summary, primary_tf, current_session,
                              fake_breakout_info, session_volatility_info):
    """
    Gemini Pro 모델을 통해 프롬프트를 전달하고, 거래 결정을 받아온다.
    """
    prompt_part_1, prompt_part_2 = generate_gemini_prompt(extended_data,
                                                          onchain_data, multi_tf_data, market_regime, thresholds,
                                                          econ_summary, primary_tf, current_session,
                                                          fake_breakout_info, session_volatility_info)

    image_path = "/Users/changpt/Downloads/Liquidation Map.png"
    image = Image.open(image_path)

    logging.info("------- Gemini Prompt -------")
    logging.info(f"{prompt_part_1}\n{prompt_part_2}")
    logging.info("------- End Prompt -------")

    sys_instruct = "You are a world-class cryptocurrency trader specializing in BTC/USDT."
    response = gemini_client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05",
        config=types.GenerateContentConfig(system_instruction=sys_instruct),
        contents=[prompt_part_1, image, prompt_part_2]
    )

    try:
        os.remove(image_path)
        logging.info("Deleted the liquidation heatmap image file after processing.")
    except Exception as e:
        logging.error(f"Error deleting the image file: {e}")

    return response.text


def escape_markdown_v2(text):
    """
    Telegram Markdown V2에서 문제가 될 수 있는 모든 특수 문자를 이스케이프 처리.
    """
    escape_chars = r"[_*\[\]()~`>#\+\-=|{}\.!]"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


def parse_trading_decision(response_text):
    """
    Gemini 응답 텍스트를 파싱하여 거래 결정 dict 형태로 반환.  (텔레그램 메시지 전송 로직 없음)
    """
    decision = {
        "final_action": "NO TRADE",
        "leverage": "1x",
        "trade_term": "N/A",
        "tp_price": "N/A",
        "sl_price": "N/A",
        "limit_order_price": "N/A",
        "rationale": "N/A"
    }

    if not response_text:
        logging.warning("parse_trading_decision received empty response_text.")
        return decision

    try:
        match = re.search(r"GO (LONG|SHORT).*?,(.*?)x, *(.*?), *(.*?), *(.*?), *(.*?), *(.*)", response_text,
                          re.DOTALL | re.IGNORECASE)
        if match:
            decision["final_action"] = f"GO {match.group(1).upper()}"
            decision["leverage"] = match.group(2).strip()
            decision["trade_term"] = match.group(3).strip()
            decision["tp_price"] = match.group(4).strip()
            decision["sl_price"] = match.group(5).strip()
            decision["limit_order_price"] = match.group(6).strip()
            # Rationale에서 마지막 \n``` 제거 (이스케이프는 main에서)
            decision["rationale"] = match.group(7).strip().replace("\n```", "")

    except Exception as e:
        logging.error(f"Error parsing Gemini response: {e}")
        decision["rationale"] = response_text  # parsing error시 raw response 전체를 rationale로

    logging.info("Parsed Trading Decision:")
    logging.info(decision)

    return decision


# =====================================================
# 9. 포지션 로깅 함수
# =====================================================
def log_decision(decision, symbol):
    """
    거래 결정 내용을 CSV 파일에 기록한다.
    """
    file_exists = os.path.isfile(DECISIONS_LOG_FILE)
    with open(DECISIONS_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["timestamp", "symbol", "final_action", "leverage", "trade_term", "tp_price", "sl_price",
                 "limit_order_price", "rationale"])
        writer.writerow([datetime.utcnow(), symbol, decision["final_action"], decision["leverage"],
                         decision["trade_term"], decision["tp_price"], decision["sl_price"],
                         decision["limit_order_price"], decision["rationale"]])
    logging.info("Trading decision logged to file.")


# =====================================================
# 10. 포지션 관리 및 메인 트레이딩 로직
# =====================================================
def compute_risk_reward(decision, entry_price, atr_value, thresholds, market_regime, donchian_upper,
                        donchian_lower):
    """
    ATR 기반 또는 횡보장 박스권(Donchian Channel) 기반 Stop Loss와 Take Profit 사용하여 Risk/Reward Ratio 계산.
    -> 이제는 Gemini가 제안한 TP/SL 값이 유효한지 검증하는 역할.

    Args:
        decision (dict): 거래 결정 정보 (Gemini 제안 TP/SL 포함)
        entry_price (float): 진입 가격
        atr_value (float): ATR 값
        thresholds (dict): 시장 상황별 지표 임계값
        market_regime (str): 시장 상황 (e.g., "bull_trend", "sideways_normal_volatility")
        donchian_upper (float): Donchian Channel 상단
        donchian_lower (float): Donchian Channel 하단

    Returns:
        tuple: (Risk/Reward Ratio, Take Profit Price, Stop Loss Price) 또는 (None, None, None)
        -> 유효하면 (rr_ratio, tp_price_str, sl_price_str) 반환.  유효하지 않으면 (None, None, None) 반환.
    """
    try:
        # decision에 이미 문자열로 저장된 tp_price, sl_price를 float으로 변환
        tp_price = float(decision["tp_price"])
        sl_price = float(decision["sl_price"])

        if "sideways" in market_regime.lower():  # 횡보장일 경우
            # Donchian Channel 기반 TP/SL 계산 (검증 로직)
            if decision["final_action"].upper() == "GO LONG":
                reward = tp_price - entry_price
                risk = entry_price - sl_price

            elif decision["final_action"].upper() == "GO SHORT":
                reward = entry_price - tp_price
                risk = sl_price - entry_price

            else:
                return None, None, None


        else:  # 추세장일 경우
            # ATR 기반 TP/SL 계산 (검증 로직)
            if decision["final_action"].upper() == "GO LONG":
                reward = tp_price - entry_price
                risk = entry_price - sl_price

            elif decision["final_action"].upper() == "GO SHORT":
                reward = entry_price - tp_price
                risk = sl_price - entry_price
            else:
                return None, None, None

        if risk <= 0 or reward <= 0:  # TP, SL 가격이 적절하지 않을 때
            return None, None, None

        rr_ratio = reward / risk
        tp_price_str = f"{tp_price:.2f}"
        sl_price_str = f"{sl_price:.2f}"
        return rr_ratio, tp_price_str, sl_price_str


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
    spot_future_price_diff = fetch_spot_future_price_diff('BTC/USDT')  # 선물-현물 가격 차이
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
    API response (HTML) 파싱 및 요약 정보 추출.
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


def get_hyperliquid_position():
    """
    Hyperliquid 거래소에서 현재 포지션을 가져온다.
    """
    try:
        positions = exchange.fetch_positions()
        if positions:
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error fetching position from Hyperliquid: {e}")
        return False


def get_hyperliquid_balance():
    """
    Hyperliquid 거래소에서 사용 가능한 잔고를 가져온다.
    """
    try:
        balance = exchange.fetch_balance()
        return float(balance['USDC']['free'])  # 사용 가능한 USDT 잔고
    except Exception as e:
        logging.error(f"Error fetching balance from Hyperliquid: {e}")
        return 0.0


def create_hyperliquid_order(symbol, decision, leverage):
    """
    Hyperliquid 거래소에 지정가 주문을 생성한다.
    """
    try:
        order_type = 'limit'
        side = 'buy' if decision['final_action'] == 'GO LONG' else 'sell'
        amount = round(float(decision['amount']), 5)
        price = float(decision['limit_order_price'])  # 지정가

        # TP/SL 가격 (문자열 -> 숫자)
        tp_price = float(decision['tp_price'])
        sl_price = float(decision['sl_price'])

        exchange.set_margin_mode('isolated', symbol, params={'leverage': leverage})

        # create_order에 필요한 형식으로 주문 목록 생성
        orders = [
            {
                'symbol': symbol,
                'type': 'limit',  # 지정가
                'side': side,
                'amount': amount,
                'price': price,
                'params': {
                    'takeProfitPrice': tp_price,  # Take Profit (별도 주문)
                    'stopLossPrice': sl_price,  # Stop Loss (별도 주문)
                    'reduceOnly': False,  # 새 포지션
                }
            },
            {
                'symbol': symbol,
                'type': 'limit',
                'side': 'sell' if side == 'buy' else 'buy',  # Use opposite side for TP
                'amount': amount,
                'price': tp_price,
                'params': {
                    'reduceOnly': True,  # Reduce Only for TP
                    'takeProfitPrice': tp_price,  # TP/SL 추가
                    'stopLossPrice': sl_price
                }
            },
            {
                'symbol': symbol,
                'type': 'limit',
                'side': 'sell' if side == 'buy' else 'buy',  # Use opposite side for SL
                'amount': amount,
                'price': sl_price,
                'params': {
                    'reduceOnly': True,  # Reduce Only for SL
                    'takeProfitPrice': tp_price,  # TP/SL 추가
                    'stopLossPrice': sl_price
                }
            }]

        order_response = exchange.create_orders(orders)  # create_order -> create_orders 변경
        logging.info(f"Hyperliquid order created: {order_response}")
        return order_response
    except Exception as e:
        logging.error(f"Error creating order on Hyperliquid: {e}")
        return None


def calculate_position_size(balance, entry_price, leverage):
    """
    진입 가격과 레버리지를 고려하여 주문 수량을 계산한다. (10의 자리 버림)
    """
    # 사용 가능한 잔고 전체 사용 (10의 자리 버림)
    amount = (int(balance / 100) * 100) * leverage / entry_price
    # amount = int(balance * leverage / entry_price / 10) * 10
    return amount


def calculate_daily_performance():
    """
    CLOSED_POSITIONS_FILE을 읽어 일일 수익률, 승률 등을 계산한다.
    """
    if not os.path.isfile(CLOSED_POSITIONS_FILE):
        return 0, 0, 0  # 파일 없으면 0 반환

    total_profit = 0
    total_trades = 0
    winning_trades = 0

    with open(CLOSED_POSITIONS_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # 오늘 날짜의 거래만 필터링
                trade_date = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S.%f").date()
                if trade_date == datetime.utcnow().date():
                    total_trades += 1
                    profit = float(row['profit'])
                    total_profit += profit
                    if row['is_win'] == 'True':
                        winning_trades += 1
            except (ValueError, KeyError) as e:
                logging.error(f"Error parsing row in closed positions file: {e}, row: {row}")  # 더 자세한 에러 로깅
                continue

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    return total_profit, total_trades, win_rate


def main():
    logging.info("Trading bot started.")

    # 초기 포지션 및 잔고 확인
    in_position = get_hyperliquid_position()
    if in_position:
        logging.info("Existing position found.  Exiting early.")
        return

    balance = get_hyperliquid_balance()

    logging.info(f"Initial balance: {balance:.2f} USDC")

    # 경제 지표 파싱 및 요약
    econ_data_raw = fetch_economic_data()
    econ_summary = parse_economic_data(econ_data_raw)

    # 포지션 없으면 전체 로직 실행

    # 1. 데이터 수집 및 가공
    #   - 멀티 타임프레임 데이터 및 지표 계산
    mtf = fetch_multi_tf_data(HYPE_SYMBOL, TIMEFRAMES, limit=300)
    if not mtf or "1h" not in mtf:
        logging.error("Not enough TF data.")
        return

    cprice = mtf["1h"]["current_price"]
    ext_data = fetch_additional_data(HYPE_SYMBOL)
    onchain_data = fetch_onchain_data()

    # 2. 시장 상황 업데이트
    regime = determine_market_regime(mtf, onchain_data)
    thresholds = adjust_indicator_thresholds(regime)
    primary_tf = choose_primary_timeframe(regime)  # Regime 기반 Primary TF 선정

    # 3. 타임프레임 합의 점수 계산
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

    # 4. 가짜 돌파 및 세션 변동성 감지 (5분봉 기준)
    fake_breakout_info = ""
    session_volatility_info = ""

    if "5m" in mtf:
        df_5m = mtf["5m"]["df_full"]

        is_asian_session_volatile = analyze_session_open_volatility(df_5m, 8, 0, threshold_factor=1.5, window=6)
        is_london_session_volatile = analyze_session_open_volatility(df_5m, 16, 0, threshold_factor=1.5, window=6)

        if is_asian_session_volatile:
            logging.info("High volatility detected at Asian session open.")
            session_volatility_info += "High volatility at Asian session open. "

        if is_london_session_volatile:
            logging.info("High volatility detected at London session open.")
            session_volatility_info += "High volatility at London session open. "

    # 5. Gemini Pro를 이용한 최종 거래 결정
    try:
        fetch_liquidation_map()
        gemini_resp_text = generate_trading_decision(
            extended_data=ext_data,
            onchain_data=onchain_data,
            multi_tf_data=mtf,
            market_regime=regime,
            thresholds=thresholds,
            econ_summary=econ_summary,
            primary_tf=primary_tf,
            current_session=current_session,
            fake_breakout_info=fake_breakout_info,
            session_volatility_info=session_volatility_info
        )

        logging.info(f"Gemini Raw Response: {gemini_resp_text}")
        decision = parse_trading_decision(gemini_resp_text)  # Gemini response parsing
        log_decision(decision, SYMBOL)  # Trading decision logging

    except Exception as e:
        logging.error(f"Error in Gemini Pro interaction or decision parsing: {e}")
        return

    if decision['final_action'].upper() == 'NO TRADE':
        logging.info("No Trade")
        return

    # Risk-Reward Ratio 계산 및 TP/SL 가격 설정 (ATR 기반)
    atr_value = mtf[primary_tf]["atr"]
    rr_ratio, tp_price_str, sl_price_str = compute_risk_reward(
        decision, cprice, atr_value, thresholds, regime,
        mtf[primary_tf]['donchian_upper'], mtf[primary_tf]['donchian_lower']
    )

    if rr_ratio:
        decision["rr_ratio"] = f"{rr_ratio:.2f}"  # Risk-Reward Ratio decision 딕셔너리에 추가
        decision["tp_price"] = tp_price_str  # TP, SL 가격 decision 딕셔너리에 업데이트 (string)
        decision["sl_price"] = sl_price_str

    rr_text = decision.get("rr_ratio", "N/A")  # decision 딕셔너리에서 R/R ratio text 가져오기

    # 자동 매매 로직 (Hyperliquid)
    if decision['final_action'] in ['GO LONG', 'GO SHORT']:
        # 포지션 크기 계산
        amount = calculate_position_size(balance, cprice, float(decision['leverage'].replace('x', '')))
        decision['amount'] = str(amount)  # Gemini 프롬프트에 'amount' 추가

        # 주문 생성 (이미 TP/SL 설정 포함)
        order = create_hyperliquid_order(HYPE_SYMBOL, decision, float(decision['leverage'].replace('x', '')))

        if order:
            # 거래 성공
            current_side = decision['final_action'].split()[-1]  # "LONG" or "SHORT"
            entry_price = float(order['price'])  # cprice에서 order['price']로 변경

            # 거래 후 텔레그램 메시지 전송
            side_emoji = "🟢 매수" if current_side == "LONG" else "🔴 매도"
            message = (
                f"*{side_emoji} 포지션 진입* ({SYMBOL})\n\n"
                f"*레버리지:* {decision['leverage']}\n"
                f"*기간:* {decision['trade_term']}\n"
                f"*진입 가격:* {entry_price:.2f}\n"
                f"*목표 가격 (TP):* {decision['tp_price']}\n"
                f"*손절 가격 (SL):* {decision['sl_price']}\n"
                f"*R/R Ratio:* {rr_text}\n\n"
                f"*분석:* {escape_markdown_v2(decision['rationale'])}"
            )
            send_telegram_message(message)
        else:
            # 거래 실패
            message = (
                f"*거래 실패* ({SYMBOL})\n\n"
                f"*이유:* {escape_markdown_v2(decision['rationale'])}\n"
            )
            send_telegram_message(message)

    else:  # NO TRADE
        message = (
            f"*거래 없음 (NO TRADE)*\n\n"
            f"*이유:* {escape_markdown_v2(decision['rationale'])}\n"
            f"*R/R Ratio:* {rr_text}"  # R/R Ratio가 None일 경우도 처리
        )
        send_telegram_message(message)

    # 일일 보고 (매일 자정에)
    now_kst = datetime.now(KST)
    if now_kst.hour == 0 and now_kst.minute == 0:
        total_profit, total_trades, win_rate = calculate_daily_performance()
        message = (
            f"*일일 보고* (KST {now_kst.strftime('%Y-%m-%d')})\n\n"
            f"*총 거래 횟수:* {total_trades}\n"
            f"*총 수익:* {total_profit:.2f} USDT\n"
            f"*승률:* {win_rate:.2f}%\n"
        )
        send_telegram_message(message)


if __name__ == "__main__":
    main()
