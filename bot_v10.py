import os
import re
import ta
import csv
import time
import ccxt
import pytz
import logging
import requests
import holidays
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
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# =====================================================
# 1. 기본 설정 및 글로벌 변수
# =====================================================

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 상수 및 API 초기화 (이전 코드와 동일)
SYMBOL = "BTC/USDT"
HYPE_SYMBOL = "BTC/USDC:USDC"
DECISIONS_LOG_FILE = "/Users/changpt/PycharmProjects/chill_trader/trading_decisions.csv"
CLOSED_POSITIONS_FILE = "closed_positions.csv"
TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

# 히트맵 초기화
image_path = "/Users/changpt/Downloads/Liquidation Map.png"

if os.path.exists(image_path):
    os.remove(image_path)

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
# 2. 텔레그램 메시지 전송 함수 (이전 코드와 동일)
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
# 3. Persistent Driver Setup (Data Crawling) (이전 코드와 동일)
# =====================================================

def get_driver():
    """
    undetected_chromedriver의 새로운 인스턴스를 생성하여 반환한다.
    """
    options = uc.ChromeOptions()
    options.add_argument("--disable-popup-blocking")
    # Configure options if needed (e.g., headless mode)
    # options.add_argument('--headless') # 필요에 따라 headless 모드 활성화
    driver = uc.Chrome(options=options)
    return driver


# =====================================================
# 4. 데이터 수집 및 기술적 지표 계산 (이전 코드와 거의 동일, 일부 지표 추가/수정)
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
    입력받은 DataFrame에 기술적 지표를 계산하여 추가합니다.
    - 추세 지표 (EMA, MACD, ADX, DMI, Aroon)
    - 모멘텀 지표 (RSI)
    - 변동성 지표 (ATR, Bollinger Bands)
    - 캔들 패턴
    - 거래량 분석 지표
    """
    # 1. 추세 지표 (EMA, MACD, ADX, DMI, Aroon)
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # ADX, DMI (추세 강도)
    adx = df.ta.adx(length=14)  # pandas_ta 사용, DataFrame 반환
    df['adx'] = adx['ADX_14']
    df['plus_di'] = adx['DMP_14']  # +DI
    df['minus_di'] = adx['DMN_14']  # -DI

    # Aroon (추세 방향 및 강도)
    aroon = df.ta.aroon(length=25)  # pandas_ta 사용, DataFrame 반환
    df['aroon_up'] = aroon['AROONU_25']
    df['aroon_down'] = aroon['AROOND_25']

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


def calculate_donchian_channel(df, window=20):
    """
    Donchian Channels를 계산하여 박스권 상단/하단을 반환합니다.
    """
    df['donchian_upper'] = df['high'].rolling(window=window).max()
    df['donchian_lower'] = df['low'].rolling(window=window).min()
    df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2  # 중간값 계산
    return df


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
        df = calculate_volume_divergence(df)

        # Donchian Channels 계산 (thresholds에서 window 가져옴)
        window = 20
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
            "adx": round(latest['adx'], 2),  # ADX 추가
            "plus_di": round(latest['plus_di'], 2),  # +DI 추가
            "minus_di": round(latest['minus_di'], 2),  # -DI 추가
            "aroon_up": round(latest['aroon_up'], 2),  # Aroon Up 추가
            "aroon_down": round(latest['aroon_down'], 2),  # Aroon Down 추가
            "df_full": df
        }
    logging.info("Multi-timeframe data and indicators calculated")
    return multi_tf_data


# =====================================================
# 5. 확장 데이터 수집 (이전 코드에서 필요한 부분만 사용)
# =====================================================

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


def fetch_open_interest(symbol, timeframe='1h', limit=5):  # timeframe, limit 파라미터 추가
    """
    Binance Futures 데이터를 이용하여 open interest 데이터 및 변화량을 가져온다.
    """
    try:
        # Binance의 fetch_open_interest_history 사용
        binanace_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        oi_history = binanace_exchange.fetch_open_interest_history(symbol, timeframe, limit=limit)

        # 최신 OI 값 (리스트의 마지막 요소)
        current_oi = oi_history[-1]['openInterestAmount']

        # 이전 OI 값들의 평균 (limit 개수만큼)
        previous_oi_values = [item['openInterestAmount'] for item in oi_history[:-1]]
        previous_oi_avg = sum(previous_oi_values) / len(previous_oi_values) if previous_oi_values else 0

        # 변화량 계산
        oi_change = current_oi - previous_oi_avg

        logging.info(f"{symbol} open interest history fetched from Binance")
        return current_oi, oi_change

    except Exception as e:
        logging.error(f"Error fetching open interest history for {symbol} from Binance: {e}")
        return "N/A", 0


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


# =====================================================
# 6. 청산맵, 차트 다운로드 (이전 코드와 동일)
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
            time.sleep(2)
            driver.quit()

        logging.info("Liquidation heatmap data fetched successfully")
    except Exception as e:
        logging.error(f"Error fetching liquidation heatmap data: {e}")


def fetch_chart(tf):
    """
    청산 히트맵 데이터를 CoinAnk 사이트에서 다운로드한다.
    """
    url = "https://www.tradingview.com/chart/?symbol=BITSTAMP%3ABTCUSD"
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(5)  # Increased wait time

        tf_str = timeframe_str_for_chart(tf)
        if driver.find_element(By.XPATH, '//div[@id="header-toolbar-intervals"]//button').is_displayed():
            driver.find_element(By.XPATH, '//div[@id="header-toolbar-intervals"]//button').click()
            time.sleep(1)
            driver.find_element(By.XPATH, f'//span[text()="{tf_str}"]').click()
            time.sleep(2)
            chart_screenshot(driver)
            time.sleep(1)
            driver.execute_script("window.onbeforeunload = null;")
            driver.quit()

        logging.info("Chart fetched successfully")
    except Exception as e:
        logging.error(f"Error fetching liquidation heatmap data: {e}")


def timeframe_str_for_chart(tf):
    if "m" in tf:
        minutes = tf.replace("m", "")
        return f"{minutes} minute{'s' if minutes != '1' else ''}"
    elif "h" in tf:
        hours = tf.replace("h", "")
        return f"{hours} hour{'s' if hours != '1' else ''}"
    elif "d" in tf:
        days = tf.replace("d", "")
        return f"{days} day{'s' if days != '1' else ''}"
    return tf


def chart_screenshot(driver):
    actions = ActionChains(driver)
    actions.key_down(Keys.ALT)
    actions.key_down(Keys.COMMAND)
    actions.send_keys('s')
    actions.key_up(Keys.ALT)
    actions.key_up(Keys.COMMAND)
    actions.perform()
    time.sleep(1)


# =====================================================
# 7. 시장 상황(장세) 결정 및 지표 임계값 조정
# =====================================================

def determine_market_regime(multi_tf_data, additional_data, current_session):
    """
    개선된 알고리즘을 사용하여 시장 상황(장세)을 결정.
    - 모호한 조건 제거
    - 세션 정보 포함
    """
    if not multi_tf_data:
        return "undefined"

    # 1. 변동성 판단 (Volatility) - ATR, Bollinger Bands (1시간봉 기준)
    volatility = "normal"
    atr_1h = multi_tf_data.get("1h", {}).get("atr", 0)
    price_1h = multi_tf_data.get("1h", {}).get("current_price", 1)

    atr_percent = (atr_1h / price_1h) * 100

    if atr_percent > 2.5:
        volatility = "high"
    elif atr_percent < 0.5:
        volatility = "low"

    bb_width_1h = multi_tf_data.get("1h", {}).get('bb_upper', 0) - multi_tf_data.get("1h", {}).get('bb_lower', 0)
    bb_width_percent_1h = (bb_width_1h / price_1h) * 100 if price_1h else 0

    donchian_width_1h = multi_tf_data.get("1h", {}).get('donchian_upper', 0) - multi_tf_data.get("1h", {}).get(
        'donchian_lower', 0)
    donchian_width_percent_1h = (donchian_width_1h / price_1h) * 100 if price_1h else 0

    # 2. 추세 판단 (Trend) - EMA, MACD, Aroon (1일봉 & 1시간봉)
    trend = "sideways"  # 기본값: 횡보
    ema20_1d = multi_tf_data.get("1d", {}).get("ema20", None)
    ema50_1d = multi_tf_data.get("1d", {}).get("ema50", None)
    ema200_1d = multi_tf_data.get("1d", {}).get("ema200", None)
    price_1d = multi_tf_data.get("1d", {}).get("current_price", None)

    ema20_1h = multi_tf_data.get("1h", {}).get("ema20", None)
    ema50_1h = multi_tf_data.get("1h", {}).get("ema50", None)
    ema200_1h = multi_tf_data.get("1h", {}).get("ema200", None)
    price_1h = multi_tf_data.get("1h", {}).get("current_price", None)

    macd_1h = multi_tf_data.get("1h", {}).get("macd", 0)
    macd_signal_1h = multi_tf_data.get("1h", {}).get("macd_signal", 0)

    aroon_up_1h = multi_tf_data.get("1h", {}).get("aroon_up", 0)
    aroon_down_1h = multi_tf_data.get("1h", {}).get("aroon_down", 0)

    # 1일봉 기준 장기 추세 (EMA)
    if (price_1d is not None and ema20_1d is not None and
            ema50_1d is not None and ema200_1d is not None):
        if price_1d > ema20_1d and price_1d > ema50_1d and price_1d > ema200_1d:
            long_term_trend = "bull"
        elif price_1d < ema20_1d and price_1d < ema50_1d and price_1d < ema200_1d:
            long_term_trend = "bear"
        else:
            long_term_trend = "neutral"
    else:
        long_term_trend = "neutral"

    # 1시간봉 기준 단기 추세 (EMA, MACD, Aroon)
    if (price_1h is not None and ema20_1h is not None and
            ema50_1h is not None and ema200_1h is not None):
        if price_1h > ema20_1h and price_1h > ema50_1h and price_1h > ema200_1h:
            if macd_1h > macd_signal_1h and aroon_up_1h > aroon_down_1h:
                short_term_trend = "bull"
            else:
                short_term_trend = "neutral"  # MACD, Aroon 불일치
        elif price_1h < ema20_1h and price_1h < ema50_1h and price_1h < ema200_1h:
            if macd_1h < macd_signal_1h and aroon_down_1h > aroon_up_1h:
                short_term_trend = "bear"
            else:
                short_term_trend = "neutral"  # MACD, Aroon 불일치
        else:
            short_term_trend = "neutral"
    else:
        short_term_trend = "neutral"

    # 장/단기 추세 조합
    if long_term_trend == "bull" and short_term_trend == "bull":
        trend = "strong_bull"
    elif long_term_trend == "bull" and short_term_trend == "neutral":
        trend = "weak_bull"
    elif long_term_trend == "bear" and short_term_trend == "bear":
        trend = "strong_bear"
    elif long_term_trend == "bear" and short_term_trend == "neutral":
        trend = "weak_bear"
    elif long_term_trend == "neutral" and short_term_trend == "bull":
        trend = "undefined_bull"  # 단기적 상승
    elif long_term_trend == "neutral" and short_term_trend == "bear":
        trend = "undefined_bear"  # 단기적 하락
    else:
        trend = "sideways"  # 횡보

    # 3. 추세 강도 (Trend Strength) - ADX, DMI (1시간봉)
    adx_1h = multi_tf_data.get("1h", {}).get("adx", 0)
    plus_di_1h = multi_tf_data.get("1h", {}).get("plus_di", 0)
    minus_di_1h = multi_tf_data.get("1h", {}).get("minus_di", 0)

    if trend == "strong_bull":
        if adx_1h < 25 or plus_di_1h <= minus_di_1h:  # 추세 강도 약화 조건
            trend = "weak_bull"
    elif trend == "strong_bear":
        if adx_1h < 25 or minus_di_1h <= plus_di_1h:  # 추세 강도 약화 조건
            trend = "weak_bear"

    # 4. 캔들 패턴 (Candle Patterns) - 1시간봉
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

    # 5. 거래량 분석 (Volume Analysis) - 1시간봉
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

    # 6. 오픈 인터레스트, 펀딩 레이트 분석
    oi_change = additional_data.get("open_interest_change", 0)
    funding_rate = additional_data.get("funding_rate", 0)

    oi_fr_signal = "neutral"
    if oi_change > 1000000 and funding_rate > 0.05:
        oi_fr_signal = "bearish_reversal_likely"
    elif oi_change > 1000000 and funding_rate < -0.05:
        oi_fr_signal = "bullish_reversal_likely"
    elif oi_change > 500000 and -0.01 < funding_rate < 0.01:
        oi_fr_signal = "trend_continuation"
    elif oi_change < -500000:
        oi_fr_signal = "trend_weakening"

    # 종합적인 시장 상황 판단 (모든 조건 명시)
    market_regime = ""

    # 추세 관련 조합
    if trend == "strong_bull":
        market_regime = "strong_bull_trend"
    elif trend == "weak_bull":
        market_regime = "weak_bull_trend"
    elif trend == "strong_bear":
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
    # 단기적 추세 (undefined)
    elif trend == "undefined_bull":
        market_regime = "undefined_bull"
    elif trend == "undefined_bear":
        market_regime = "undefined_bear"

    # 변동성 추가
    market_regime = f"{volatility}_volatility_{market_regime}"

    # 캔들 패턴 정보 추가
    if candle_pattern != "neutral":
        market_regime += f"_{candle_pattern}_candle"

    # 거래량 분석 정보 추가
    if volume_analysis != "neutral":
        market_regime += f"_{volume_analysis}"

    # 오픈 인터레스트/펀딩 레이트 시그널 추가
    if oi_fr_signal != "neutral":
        market_regime += f"_{oi_fr_signal}"

    # 세션별 특성 고려
    if "ASIAN" in current_session:
        if "sideways" in market_regime:
            market_regime = "tight_sideways"  # 아시아 세션 + 횡보 -> 타이트한 횡보
    elif "LONDON" in current_session:
        if "sideways" in market_regime:
            market_regime = "normal_sideways"  # 런던 세션 + 횡보 -> 보통 횡보
    elif "US" in current_session:
        if "strong" in market_regime:
            market_regime += "_high_vol"  # US 세션 + 강한 추세 -> 변동성 높음

    logging.info(f"Market regime determined: {market_regime.upper()}")
    return market_regime


def adjust_indicator_thresholds(market_regime, multi_tf_data):
    """
    시장 상황에 따라 지표 임계값, ATR 배수, 지표 가중치를 동적으로 조정.
    - 지지/저항 레벨 추가 (정확하게)
    """
    thresholds = {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "donchian_window": 20,
        "atr_multiplier_tp": 3,
        "atr_multiplier_sl": 2,
        "indicator_weights": {
            "rsi": 0.2,
            "macd": 0.2,
            "ema": 0.3,
            "donchian": 0.2,
            "volume": 0.1,
            "oi_fr": 0,
            "aroon": 0.0,  # Aroon 지표 가중치
        },
        "support_levels": [],
        "resistance_levels": [],
    }

    # 지지/저항 레벨 계산 (정확하게)
    for tf, data in multi_tf_data.items():
        df = data['df_full']  # 전체 DataFrame 사용

        # 1. Donchian Channel
        thresholds["support_levels"].append(round(data["donchian_lower"], 2))
        thresholds["resistance_levels"].append(round(data["donchian_upper"], 2))

        # 2. EMA (20, 50, 200)
        thresholds["support_levels"].append(round(data["ema20"], 2))
        thresholds["resistance_levels"].append(round(data["ema20"], 2))
        thresholds["support_levels"].append(round(data["ema50"], 2))
        thresholds["resistance_levels"].append(round(data["ema50"], 2))
        thresholds["support_levels"].append(round(data["ema200"], 2))
        thresholds["resistance_levels"].append(round(data["ema200"], 2))

        # 3. 피보나치 되돌림 (0.382, 0.5, 0.618) - 전체 기간
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        thresholds["support_levels"].extend([
            round(high - 0.382 * diff, 2),
            round(high - 0.5 * diff, 2),
            round(high - 0.618 * diff, 2),
        ])
        thresholds["resistance_levels"].extend([
            round(low + 0.382 * diff, 2),
            round(low + 0.5 * diff, 2),
            round(low + 0.618 * diff, 2),
        ])

        # 4. 이전 N일 고가/저가 (N=5, 10) - 최근 데이터
        for n in [5, 10]:
            thresholds["support_levels"].append(round(df['low'].tail(n).min(), 2))
            thresholds["resistance_levels"].append(round(df['high'].tail(n).max(), 2))

        # 5. 볼린저 밴드 (추가)
        thresholds["support_levels"].append(round(data["bb_lower"], 2))
        thresholds["resistance_levels"].append(round(data["bb_upper"], 2))

        # 중복 제거 및 정렬
        thresholds["support_levels"] = sorted(list(set(thresholds["support_levels"])))
        thresholds["resistance_levels"] = sorted(list(set(thresholds["resistance_levels"])))

        # 시장 상황에 따른 조정
        if "strong_bull_trend" in market_regime:
            thresholds["atr_multiplier_tp"] = 4
            thresholds["atr_multiplier_sl"] = 2.5
            thresholds["indicator_weights"]["ema"] = 0.4
            thresholds["indicator_weights"]["volume"] = 0.2
            thresholds["indicator_weights"]["oi_fr"] = 0.1
            thresholds["indicator_weights"]["aroon"] = 0.1  # Aroon 가중치 추가

        elif "weak_bull_trend" in market_regime:
            thresholds["atr_multiplier_tp"] = 3
            thresholds["atr_multiplier_sl"] = 1.8
            thresholds["indicator_weights"]["ema"] = 0.35
            thresholds["indicator_weights"]["rsi"] = 0.25
            thresholds["indicator_weights"]["oi_fr"] = 0.15
            thresholds["indicator_weights"]["aroon"] = 0.05

        elif "strong_bear_trend" in market_regime:
            thresholds["atr_multiplier_tp"] = 4
            thresholds["atr_multiplier_sl"] = 2.5
            thresholds["indicator_weights"]["ema"] = 0.4
            thresholds["indicator_weights"]["volume"] = 0.2
            thresholds["indicator_weights"]["oi_fr"] = 0.1
            thresholds["indicator_weights"]["aroon"] = 0.1

        elif "weak_bear_trend" in market_regime:
            thresholds["atr_multiplier_tp"] = 3
            thresholds["atr_multiplier_sl"] = 1.8
            thresholds["indicator_weights"]["ema"] = 0.35
            thresholds["indicator_weights"]["rsi"] = 0.25
            thresholds["indicator_weights"]["oi_fr"] = 0.15
            thresholds["indicator_weights"]["aroon"] = 0.05

        elif "tight_sideways" in market_regime:
            thresholds["atr_multiplier_tp"] = 2
            thresholds["atr_multiplier_sl"] = 1.2
            thresholds["indicator_weights"]["donchian"] = 0.4
            thresholds["indicator_weights"]["rsi"] = 0.3
            thresholds["indicator_weights"]["oi_fr"] = 0.1

        elif "wide_sideways" in market_regime:
            thresholds["atr_multiplier_tp"] = 2.5
            thresholds["atr_multiplier_sl"] = 1.7
            thresholds["indicator_weights"]["donchian"] = 0.4
            thresholds["indicator_weights"]["rsi"] = 0.3

        elif "normal_sideways" in market_regime:
            thresholds["atr_multiplier_tp"] = 2.8
            thresholds["atr_multiplier_sl"] = 1.5
            thresholds["indicator_weights"]["donchian"] = 0.35
            thresholds["indicator_weights"]["rsi"] = 0.3

        # undefined 상태 추가
        elif "undefined_bull" in market_regime:
            thresholds["atr_multiplier_tp"] = 2.8
            thresholds["atr_multiplier_sl"] = 1.6
            thresholds["indicator_weights"]["ema"] = 0.3
            thresholds["indicator_weights"]["macd"] = 0.25
            thresholds["indicator_weights"]["aroon"] = 0.15
            thresholds["indicator_weights"]["oi_fr"] = 0.1

        elif "undefined_bear" in market_regime:
            thresholds["atr_multiplier_tp"] = 2.8
            thresholds["atr_multiplier_sl"] = 1.6
            thresholds["indicator_weights"]["ema"] = 0.3
            thresholds["indicator_weights"]["macd"] = 0.25
            thresholds["indicator_weights"]["aroon"] = 0.15
            thresholds["indicator_weights"]["oi_fr"] = 0.1

        # 변동성에 따른 추가 조정
        if "high_volatility" in market_regime:
            thresholds["atr_multiplier_sl"] += 0.5
            thresholds["indicator_weights"]["volume"] += 0.1
            thresholds["indicator_weights"]["atr"] = 0.2  # ATR 가중치 추가

        elif "low_volatility" in market_regime:
            thresholds["atr_multiplier_sl"] -= 0.2
            thresholds["indicator_weights"]["ema"] += 0.1
            thresholds["indicator_weights"]["atr"] = 0.05

        return thresholds


# =====================================================
# 8. 전략 템플릿 정의
# =====================================================

strategy_templates = {
    "strong_bull_trend_follow": {
        "name": "Strong Bull Trend Following (Momentum)",
        "description": "강력한 상승 추세 추종",
        "primary_timeframe": "1d",
        "indicators": {
            "ema": {"weight": 0.4, "params": [20, 50, 200]},
            "rsi": {"weight": 0.1, "params": [14, 40, 80]},
            "macd": {"weight": 0.1, "params": []},
            "volume": {"weight": 0.2, "params": []},
            "oi_fr": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.1, "params": [25]},  # Aroon 추가
        },
        "entry_rules": {
            "long": [
                "price > ema20_1d",
                "price > ema50_1d",
                "price > ema200_1d",
                "macd > macd_signal",  # MACD 시그널
                "aroon_up > aroon_down",  # Aroon Up > Aroon Down
                "volume_change > 20",
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 4",  # 목표 가격 확대
            "sl": "atr_multiplier * 2.5",  # 손절 범위 조정
        },
        "trade_term": "1d ~ 3d",
        "leverage": "3x ~ 5x"
    },
    "weak_bull_trend_pullback": {
        "name": "Weak Bull Trend Pullback (Dip Buying)",
        "description": "약한 상승 추세에서 눌림목 매수",
        "primary_timeframe": "4h",
        "indicators": {
            "ema": {"weight": 0.35, "params": [20, 50]},
            "rsi": {"weight": 0.35, "params": [14, 35, 75]},
            "macd": {"weight": 0.1, "params": []},
            "volume": {"weight": 0.1, "params": []},
            "oi_fr": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
        },
        "entry_rules": {
            "long": [
                "price > ema50_4h",
                "rsi < 35",
                "bullish_divergence",
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 3",
            "sl": "atr_multiplier * 1.8",
        },
        "trade_term": "6h ~ 1d",
        "leverage": "3x ~ 5x"
    },
    "strong_bear_trend_follow": {
        "name": "Strong Bear Trend Following (Momentum)",
        "description": "강력한 하락 추세 추종",
        "primary_timeframe": "1d",
        "indicators": {
            "ema": {"weight": 0.4, "params": [20, 50, 200]},
            "rsi": {"weight": 0.1, "params": [14, 20, 60]},
            "macd": {"weight": 0.1, "params": []},
            "volume": {"weight": 0.2, "params": []},
            "oi_fr": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.1, "params": [25]},  # Aroon 추가
        },
        "entry_rules": {
            "short": [
                "price < ema20_1d",
                "price < ema50_1d",
                "price < ema200_1d",
                "macd < macd_signal",  # MACD 시그널
                "aroon_down > aroon_up",  # Aroon Down > Aroon Up
                "volume_change > 20",
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 4",  # 목표 가격 확대
            "sl": "atr_multiplier * 2.5",  # 손절 범위 조정
        },
        "trade_term": "1d ~ 3d",
        "leverage": "3x ~ 5x"
    },
    "weak_bear_trend_bounce": {
        "name": "Weak Bear Trend Bounce (Short Selling)",
        "description": "약한 하락 추세에서 반등 매도",
        "primary_timeframe": "4h",
        "indicators": {
            "ema": {"weight": 0.35, "params": [20, 50]},
            "rsi": {"weight": 0.35, "params": [14, 25, 65]},
            "macd": {"weight": 0.1, "params": []},
            "volume": {"weight": 0.1, "params": []},
            "oi_fr": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
        },
        "entry_rules": {
            "short": [
                "price < ema50_4h",
                "rsi > 65",
                "bearish_divergence",
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 3",
            "sl": "atr_multiplier * 1.8",
        },
        "trade_term": "6h ~ 1d",
        "leverage": "3x ~ 5x"
    },
    "tight_sideways_range": {
        "name": "Tight Sideways Range (Scalping)",
        "description": "좁은 범위 횡보장에서 스캘핑",
        "primary_timeframe": "5m",
        "indicators": {
            "donchian": {"weight": 0.4, "params": [15]},
            "rsi": {"weight": 0.3, "params": [14, 25, 75]},
            "macd": {"weight": 0.2, "params": []},
            "volume": {"weight": 0.1, "params": []},
            "ema": {"weight": 0.0, "params": []},
            "oi_fr": {"weight": 0.0, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
        },
        "entry_rules": {
            "long": [
                "price <= donchian_lower_5m",
                "rsi < 25",
            ],
            "short": [
                "price >= donchian_upper_5m",
                "rsi > 75",
            ],
        },
        "exit_rules": {
            "tp": "donchian_middle_5m",
            "sl": "donchian_lower_5m - atr * 1",
        },
        "trade_term": "5m ~ 15m",
        "leverage": "5x ~ 10x"
    },
    "wide_sideways_range": {
        "name": "Wide Sideways Range (Range Trading)",
        "description": "넓은 범위 횡보장에서 레인지 트레이딩",
        "primary_timeframe": "1h",
        "indicators": {
            "donchian": {"weight": 0.4, "params": [25]},
            "rsi": {"weight": 0.3, "params": [14, 35, 65]},
            "macd": {"weight": 0.2, "params": []},
            "volume": {"weight": 0.1, "params": []},
            "ema": {"weight": 0.0, "params": []},
            "oi_fr": {"weight": 0.0, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
        },
        "entry_rules": {
            "long": [
                "price <= donchian_lower_1h",
                "rsi < 35",
            ],
            "short": [
                "price >= donchian_upper_1h",
                "rsi > 65",
            ],
        },
        "exit_rules": {
            "tp": "donchian_middle_1h",
            "sl": "donchian_lower_1h - atr * 1.5",
        },
        "trade_term": "1h ~ 4h",
        "leverage": "3x ~ 5x"
    },
    "normal_sideways_range": {
        "name": "Normal Sideways Range (Range Trading)",
        "description": "보통 범위 횡보장에서 레인지 트레이딩",
        "primary_timeframe": "15m",
        "indicators": {
            "donchian": {"weight": 0.35, "params": [20]},
            "rsi": {"weight": 0.3, "params": [14, 30, 70]},
            "macd": {"weight": 0.25, "params": []},
            "volume": {"weight": 0.05, "params": []},
            "ema": {"weight": 0.05, "params": []},
            "oi_fr": {"weight": 0.0, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
        },
        "entry_rules": {
            "long": [
                "price <= donchian_lower_15m",
                "rsi < 30",
            ],
            "short": [
                "price >= donchian_upper_15m",
                "rsi > 70",
            ],
        },
        "exit_rules": {
            "tp": "donchian_middle_15m",
            "sl": "donchian_lower_15m - atr * 1.2",
        },
        "trade_term": "15m ~ 1h",
        "leverage": "3x ~ 5x"
    },
    "bearish_reversal": {
        "name": "Bearish Reversal (OI & Funding Rate)",
        "description": "높은 오픈 인터레스트와 양의 펀딩 레이트를 기반으로 하락 반전 포착",
        "primary_timeframe": "1h",
        "indicators": {
            "oi_fr": {"weight": 0.6, "params": []},
            "rsi": {"weight": 0.2, "params": [14, 30, 70]},
            "ema": {"weight": 0.1, "params": [20, 50]},
            "volume": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
        },
        "entry_rules": {
            "short": [
                "oi_change > 1000000",
                "funding_rate > 0.05",
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2",
            "sl": "atr_multiplier * 1.5",
        },
        "trade_term": "4h ~ 12h",
        "leverage": "3x ~ 5x"
    },
    "bullish_reversal": {
        "name": "Bullish Reversal (OI & Funding Rate)",
        "description": "높은 오픈 인터레스트와 음의 펀딩 레이트를 기반으로 상승 반전 포착",
        "primary_timeframe": "1h",
        "indicators": {
            "oi_fr": {"weight": 0.6, "params": []},
            "rsi": {"weight": 0.2, "params": [14, 30, 70]},
            "ema": {"weight": 0.1, "params": [20, 50]},
            "volume": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
        },
        "entry_rules": {
            "long": [
                "oi_change > 1000000",
                "funding_rate < -0.05",
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2",
            "sl": "atr_multiplier * 1.5",
        },
        "trade_term": "4h ~ 12h",
        "leverage": "3x ~ 5x"
    },
    "trend_continuation": {
        "name": "Trend Continuation (OI & Funding Rate)",
        "description": "오픈 인터레스트와 펀딩 레이트를 기반으로 추세 지속 판단",
        "primary_timeframe": "1h",
        "indicators": {
            "oi_fr": {"weight": 0.6, "params": []},
            "rsi": {"weight": 0.1, "params": [14, 30, 70]},
            "ema": {"weight": 0.2, "params": [20, 50]},
            "volume": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
        },
        "entry_rules": {
            "long": [
                "oi_change > 500000",
                "funding_rate > -0.01",
                "funding_rate < 0.01",
                "price > ema20_1h",  # 가격 조건 추가
            ],
            "short": [
                "oi_change > 500000",
                "funding_rate > -0.01",
                "funding_rate < 0.01",
                "price < ema20_1h",  # 가격 조건 추가
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2",
            "sl": "atr_multiplier * 1.5",
        },
        "trade_term": "4h ~ 12h",
        "leverage": "3x ~ 5x"
    },
    "undefined_bull_trend": {  # 수정: 불분명한 상승 추세
        "name": "Undefined Bull Trend",
        "description": "불분명한 상승 추세에서의 전략",
        "primary_timeframe": "1h",
        "indicators": {
            "ema": {"weight": 0.3, "params": [20, 50]},
            "rsi": {"weight": 0.2, "params": [14, 40, 60]},
            "macd": {"weight": 0.2, "params": []},
            "volume": {"weight": 0.1, "params": []},
            "oi_fr": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.1, "params": [25]},
        },
        "entry_rules": {
            "long": [
                "price > ema20_1h",
                "rsi > 40",  # 완만한 상승 조건
                "bullish_divergence",
            ],
            "short": [],  # 필요시 short 규칙 추가
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2.8",
            "sl": "atr_multiplier * 1.6",
        },
        "trade_term": "4h ~ 12h",
        "leverage": "2x ~ 3x"  # 낮은 레버리지
    },
    "undefined_bear_trend": {  # 수정: 불분명한 하락 추세
        "name": "Undefined Bear Trend",
        "description": "불분명한 하락 추세에서의 전략",
        "primary_timeframe": "1h",
        "indicators": {
            "ema": {"weight": 0.3, "params": [20, 50]},
            "rsi": {"weight": 0.2, "params": [14, 40, 60]},
            "macd": {"weight": 0.2, "params": []},
            "volume": {"weight": 0.1, "params": []},
            "oi_fr": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.1, "params": [25]},
        },
        "entry_rules": {
            "long": [],  # 필요시 long 규칙 추가
            "short": [
                "price < ema20_1h",
                "rsi < 60",  # 완만한 하락 조건
                "bearish_divergence",
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2.8",
            "sl": "atr_multiplier * 1.6",
        },
        "trade_term": "4h ~ 12h",
        "leverage": "2x ~ 3x"  # 낮은 레버리지
    },

    # 변동성 돌파 전략 추가
    "high_volatility_breakout": {
        "name": "High Volatility Breakout",
        "description": "높은 변동성 돌파 전략",
        "primary_timeframe": "15m",  # 짧은 타임프레임
        "indicators": {
            "ema": {"weight": 0.2, "params": [20]},
            "atr": {"weight": 0.3, "params": [14]},  # ATR 중요
            "volume": {"weight": 0.3, "params": []},
            "oi_fr": {"weight": 0.1, "params": []},
            "rsi": {"weight": 0.1, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)

        },
        "entry_rules": {
            "long": [
                "price > ema20_15m",
                "volume_change > 50",  # 급격한 거래량 증가
                "atr > previous_atr * 1.5",  # 이전 ATR보다 1.5배 큰 ATR
            ],
            "short": [
                "price < ema20_15m",
                "volume_change > 50",
                "atr > previous_atr * 1.5",
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 3",  # 변동성 고려, 큰 TP
            "sl": "atr_multiplier * 2",  # 변동성 고려, 넓은 SL
        },
        "trade_term": "15m ~ 1h",
        "leverage": "3x ~ 5x"
    },
    # 캔들 패턴 기반 전략 (예시: Bullish Engulfing)
    "bullish_engulfing_pattern": {
        "name": "Bullish Engulfing Pattern",
        "description": "상승 잉태형 캔들 패턴 매매",
        "primary_timeframe": "1h",
        "indicators": {
            "ema": {"weight": 0.2, "params": [50]},  # EMA 추가
            "engulfing_bullish": {"weight": 0.6, "params": []},  # 캔들 패턴에 높은 가중치
            "volume": {"weight": 0.2, "params": []},
            "oi_fr": {"weight": 0.0, "params": []},
            "rsi": {"weight": 0.0, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
            "macd": {"weight": 0.0, "params": []}
        },
        "entry_rules": {
            "long": [
                "engulfing_bullish_1h",  # Bullish Engulfing on 1h
                "price > ema50_1h"  # 가격이 50ema 위에
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2.5",
            "sl": "atr_multiplier * 1.8",
        },
        "trade_term": "1h ~ 4h",
        "leverage": "3x ~ 4x"
    },

    # 캔들 패턴 기반 전략 (예시: Bearish Engulfing)
    "bearish_engulfing_pattern": {
        "name": "Bearish Engulfing Pattern",
        "description": "하락 잉태형 캔들 패턴 매매",
        "primary_timeframe": "1h",
        "indicators": {
            "ema": {"weight": 0.2, "params": [50]},
            "engulfing_bearish": {"weight": 0.6, "params": []},  # 캔들 패턴에 높은 가중치
            "volume": {"weight": 0.2, "params": []},
            "oi_fr": {"weight": 0.0, "params": []},
            "rsi": {"weight": 0.0, "params": []},
            "aroon": {"weight": 0.0, "params": [25]},  # Aroon (가중치 0)
            "macd": {"weight": 0.0, "params": []}
        },
        "entry_rules": {
            "short": [
                "engulfing_bearish_1h",  # Bearish Engulfing on 1h
                "price < ema50_1h"
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2.5",
            "sl": "atr_multiplier * 1.8",
        },
        "trade_term": "1h ~ 4h",
        "leverage": "3x ~ 4x"
    },
}


# =====================================================
# 9. 전략 선택 함수
# =====================================================

def select_strategy(market_regime):
    """
    결정된 시장 상황(장세)에 따라 가장 적합한 전략을 선택.
    - 우선순위 기반 선택 (실제 트레이더의 판단과 유사하게)
    """

    # 우선순위: 특정 조건 -> 일반적인 조건

    # 1. 캔들 패턴 기반 전략 (가장 먼저 확인)
    if "bullish_candle" in market_regime:
        return strategy_templates["bullish_engulfing_pattern"]
    if "bearish_candle" in market_regime:
        return strategy_templates["bearish_engulfing_pattern"]

    # 2. 변동성 돌파 전략 (높은 변동성에서 우선)
    if "high_volatility" in market_regime:
        return strategy_templates["high_volatility_breakout"]

    # 3. 추세 추종/반전 전략
    if "strong_bull_trend" in market_regime:
        return strategy_templates["strong_bull_trend_follow"]
    elif "weak_bull_trend" in market_regime:
        return strategy_templates["weak_bull_trend_pullback"]
    elif "strong_bear_trend" in market_regime:
        return strategy_templates["strong_bear_trend_follow"]
    elif "weak_bear_trend" in market_regime:
        return strategy_templates["weak_bear_trend_bounce"]

    # 4. 불분명한 추세 전략 (추가)
    elif "undefined_bull" in market_regime:
        return strategy_templates["undefined_bull_trend"]
    elif "undefined_bear" in market_regime:
        return strategy_templates["undefined_bear_trend"]

    # 5. 횡보장 전략
    elif "tight_sideways" in market_regime:
        return strategy_templates["tight_sideways_range"]
    elif "wide_sideways" in market_regime:
        return strategy_templates["wide_sideways_range"]
    elif "normal_sideways" in market_regime:
        return strategy_templates["normal_sideways_range"]

    # 6. OI/FR 기반 전략
    elif "bearish_reversal_likely" in market_regime:
        return strategy_templates["bearish_reversal"]
    elif "bullish_reversal_likely" in market_regime:
        return strategy_templates["bullish_reversal"]
    elif "trend_continuation" in market_regime:
        return strategy_templates["trend_continuation"]

    # 어떤 전략도 선택되지 않은 경우 (undefined)
    return None


# =====================================================
# 10. Gemini 프롬프트 생성 및 거래 결정
# =====================================================
def generate_gemini_prompt(multi_tf_data, market_regime, strategy, thresholds,
                           current_session,
                           additional_data, econ_summary):  # econ_summary 파라미터 추가
    """
    Gemini Pro 모델에 전달할 Prompt를 생성합니다. (전략 템플릿 기반)
    """
    # 전략 정보 (선택된 전략 우선순위 명시)
    strategy_name = strategy["name"]
    strategy_description = strategy["description"]
    primary_tf = strategy["primary_timeframe"]
    strategy_priority = ""

    if "candle" in market_regime:
        strategy_priority = "Highest (Candle Pattern)"
    elif "high_volatility" in market_regime:
        strategy_priority = "High (Volatility Breakout)"
    elif "strong" in market_regime or "weak" in market_regime:
        strategy_priority = "Medium (Trend Following/Reversal)"
    elif "undefined" in market_regime:
        strategy_priority = "Medium (Undefined Trend)"
    elif "sideways" in market_regime:
        strategy_priority = "Low (Range Trading)"
    elif "reversal_likely" in market_regime or "trend_continuation" in market_regime:
        strategy_priority = "Low (OI/FR Based)"

    # 지표 요약 (간결하게, Aroon 추가)
    indicators_summary = ""
    for tf, data in multi_tf_data.items():
        indicators_summary += f"**{tf}:**\n"
        indicators_summary += f"  - Price: {data['current_price']:.2f}\n"
        for ind_name, ind_params in strategy["indicators"].items():
            if ind_name in data and data[ind_name] is not None:
                # 'oi_fr' (Open Interest & Funding Rate)는 제외
                if ind_name != 'oi_fr':
                    indicators_summary += f"  - {ind_name.upper()}: {data[ind_name]:.2f}\n"

    # 진입/청산 규칙
    entry_rules_long = "\n".join([f"  - {rule}" for rule in strategy["entry_rules"].get("long", [])])
    entry_rules_short = "\n".join([f"  - {rule}" for rule in strategy["entry_rules"].get("short", [])])

    # TP/SL 규칙
    tp_rule = strategy["exit_rules"]["tp"]
    sl_rule = strategy["exit_rules"]["sl"]

    # trader_term, leverage 전략
    trade_term = strategy["trade_term"]
    leverage = strategy["leverage"]

    # 현재 시간 (KST)
    now_kst = datetime.now(KST)
    current_time_kst = now_kst.strftime("%Y-%m-%d %H:%M:%S (KST)")

    # 세션별 가이드 (상세하게)
    session_guides = {
        "OVERNIGHT": (
            "Low liquidity. Be cautious of fake breakouts and increased slippage. "
            "Consider using tighter stops and smaller position sizes."
        ),
        "ASIAN": (
            "08:00-09:00 KST: Potential volatility spike due to market open. "
            "After 09:00: Trend may develop, but be aware of potential reversals. "
            "Lower liquidity compared to London/US sessions."
        ),
        "LONDON": (
            "16:00 KST open: Expect high volatility and increased trading volume. "
            "Good for trend trading, but be mindful of potential whipsaws."
        ),
        "US": (
            "Highest volume and volatility. Be prepared for sharp moves and potential reversals. "
            "News events can have a significant impact."
        ),
        "TRANSITION": (
            "Low liquidity, potential trend formation before Asian open. "
            "Be cautious of low volume moves."
        ),
        "ASIAN_WEEKEND": (
            "Lower liquidity, increased volatility. Be cautious. "
            "Avoid holding positions over the weekend if possible."
        ),
        "LONDON_WEEKEND": (
            "Lower liquidity, increased volatility. Be cautious. "
            "Avoid holding positions over the weekend if possible."
        ),
        "US_WEEKEND": (
            "Lower liquidity, increased volatility. Be cautious. "
            "Watch for sudden price swings and potential manipulation. "
            "Avoid holding positions over the weekend."
        ),
        "US_US_HOLIDAY": (
            "US market closed. Expect lower liquidity and potentially erratic price movements. "
            "Other markets may still be active."
        ),
    }
    session_guide = session_guides.get(current_session, "No specific guidance for this session.")

    # 지지/저항 정보 (자세하게)
    support_levels_str = ", ".join([f"{level:.2f}" for level in thresholds["support_levels"]])
    resistance_levels_str = ", ".join([f"{level:.2f}" for level in thresholds["resistance_levels"]])

    prompt_text_1 = f"""
**Objective:** Make optimal trading decisions for BTC/USDT.

**Market Context:**
- Regime: **{market_regime.upper()}**
- Strategy: **{strategy_name}** ({strategy_description})
- Strategy Priority: **{strategy_priority}**
- Primary Timeframe: **{primary_tf}**
- Current Session: **{current_session}** ({current_time_kst})
- Session Guide: {session_guide}
- Economic Events: {econ_summary}
- Recommended Trade Term: **{trade_term}**
- Recommended Leverage: **{leverage}**

**Technical Analysis Summary:**
{indicators_summary}

**Key Indicator Thresholds:**
- RSI Oversold: {thresholds.get('rsi_oversold', 'N/A')}
- RSI Overbought: {thresholds.get('rsi_overbought', 'N/A')}
- Donchian Window: {thresholds.get('donchian_window', 'N/A')}

**Additional Market Data:**
- Funding Rate: {additional_data.get('funding_rate', 'N/A')}
- Open Interest: {additional_data.get('open_interest', 'N/A')}
- Open Interest Change: {additional_data.get('open_interest_change', 'N/A')}
- Order Book: Bid={additional_data.get('order_book', {}).get('bid', 'N/A')}, Ask={additional_data.get('order_book', {}).get('ask', 'N/A')}, Spread={additional_data.get('order_book', {}).get('spread', 'N/A')}
- Fear & Greed: {additional_data.get('fear_and_greed_index', ('N/A', 'N/A'))[0]} ({additional_data.get('fear_and_greed_index', ('N/A', 'N/A'))[1]})

**Strategy Guidelines:**

- **Entry Rules (Long):**
{entry_rules_long}

- **Entry Rules (Short):**
{entry_rules_short}

- **Take Profit (TP):** {tp_rule}
- **Stop Loss (SL):** {sl_rule}

**Key Support Levels:** {support_levels_str}
**Key Resistance Levels:** {resistance_levels_str}

"""
    prompt_text_2 = f"""
**Liquidation Map Analysis Guide (Image Provided):**
- **Support and Resistance:** Identify potential support and resistance levels based on liquidation clusters. Larger clusters indicate stronger levels.
- **Cascading Liquidations:** Assess the risk of cascading liquidations. If large clusters are close to the current price, a small move could trigger a chain reaction.
- **Volatility Prediction:** Estimate potential volatility. Widely spaced clusters suggest lower volatility, while closely spaced clusters suggest higher volatility.
- **Risk Assessment:** Compare long vs. short liquidation levels. If one side has significantly larger clusters, it indicates higher risk for that side.
- **Prioritize:** If the liquidation map provides clear and strong support/resistance levels that align with other indicators, use them to inform your TP/SL and entry decisions. If the map is unclear or contradicts other signals, rely more on other indicators and market context.

"""

    chart_guide = f"""
**Chart Analysis Guide (Image Provided):**
- **Trendlines:** Identify any visible trendlines (upward, downward, or sideways).
- **Support/Resistance:** Look for horizontal support and resistance levels, paying attention to areas where the price has previously reversed.
- **Chart Patterns:**  Identify any classic chart patterns (e.g., head and shoulders, double top/bottom, triangles).
- **Candlestick Patterns:** Note any significant candlestick patterns (e.g., engulfing, doji, hammer) that might suggest reversals or continuations.
- **Indicator Confirmation:** Check if the chart visually confirms the signals from your indicators (e.g., price crossing above EMA for a long signal).
- **Divergences:** Look for any divergences between price action and indicators (e.g., bearish divergence where price makes a higher high but RSI makes a lower high).
- **Prioritize:** If the chart provides a clear and strong visual confirmation of your trading decision, increase your confidence in the trade. If the chart contradicts your indicators or strategy, re-evaluate your decision.

"""

    prompt_text_3 = f"""
**Task:**

Based on all provided information, decide: **GO LONG, GO SHORT, or NO TRADE.**

If GO LONG or GO SHORT, also determine:
- **Recommended Leverage:** (Based on strategy's recommendation and market context. Adjust if necessary.)
- **Trade Term:** (Based on strategy's recommendation. Adjust if necessary.)
- **Take Profit Price:** (Consider key support/resistance levels, liquidation map clusters, and the ATR-based suggestion. Prioritize risk management.)
- **Stop Loss Price:** (Consider key support/resistance levels, liquidation map clusters, and the ATR-based suggestion. **Your SL should never risk more than 2% of your account balance.**)
- **Limit Order Price:** (For long positions, consider a price slightly below the current market price to ensure a fill. For short positions, consider a price slightly above.)
- **Rationale:** (Explain your decision in detail, referencing specific indicators, market context, liquidation map analysis, and chart analysis. Maximum 5 sentences.)

**Output Format (Comma-Separated):**

Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Rationale

**Example Outputs:**
```
GO LONG, 5x, 1d, 48500.00, 46000.00, 47050.00, Rationale
```
```
NO TRADE, N/A, N/A, N/A, N/A, N/A, Rationale
```"""
    return prompt_text_1, prompt_text_2, chart_guide, prompt_text_3


def generate_trading_decision(multi_tf_data, market_regime, strategy, thresholds, current_session, additional_data,
                              econ_summary):  # 파라미터 추가
    """
    Gemini Pro 모델을 통해 프롬프트를 전달하고, 거래 결정을 받아온다. (청산맵 이미지 포함)
    """
    prompt_part_1, prompt_part_2 = generate_gemini_prompt(multi_tf_data, market_regime, strategy, thresholds,
                                                          current_session, additional_data, econ_summary)

    image = Image.open(image_path)

    logging.info("------- Gemini Prompt -------")
    logging.info(f"{prompt_part_1}\n{prompt_part_2}")
    logging.info("------- End Prompt -------")

    sys_instruct = "You are a world-class cryptocurrency trader specializing in BTC/USDT."
    response = gemini_client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05",
        # model="gemini-2.0-flash-thinking-exp-01-21",
        config=types.GenerateContentConfig(system_instruction=sys_instruct),
        contents=[prompt_part_1, image, prompt_part_2]  # 이미지 추가
    )

    try:
        os.remove(image_path)
        logging.info("Deleted the liquidation heatmap image file after processing.")
    except Exception as e:
        logging.error(f"Error deleting the image file: {e}")

    return response.text


def parse_trading_decision(response_text):
    """
    Gemini 응답 텍스트를 파싱하여 거래 결정 dict 형태로 반환.
    NO TRADE인 경우에도 rationale 추출 가능하도록 개선.
    """
    decision = {
        "final_action": "NO TRADE",
        "leverage": "N/A",
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
        # 1. GO LONG/SHORT 케이스 먼저 시도 (기존 로직)
        match_trade = re.search(r"GO (LONG|SHORT).*?,(.*?)x, *(.*?), *(.*?), *(.*?), *(.*?), *(.*)", response_text,
                                re.DOTALL | re.IGNORECASE)
        if match_trade:
            decision["final_action"] = f"GO {match_trade.group(1).upper()}"
            decision["leverage"] = match_trade.group(2).strip()
            decision["trade_term"] = match_trade.group(3).strip()
            decision["tp_price"] = match_trade.group(4).strip()
            decision["sl_price"] = match_trade.group(5).strip()
            decision["limit_order_price"] = match_trade.group(6).strip()
            decision["rationale"] = match_trade.group(7).strip().replace('`', '').replace('\\', '')

        else:  # 2. GO LONG/SHORT 매칭 실패 시, NO TRADE 케이스 시도
            match_no_trade = re.search(r"NO TRADE.*?N/A.*?N/A.*?N/A.*?N/A.*?N/A.*?,(.*?)$", response_text,
                                       re.DOTALL | re.IGNORECASE)
            if match_no_trade:
                decision["final_action"] = "NO TRADE"
                decision["rationale"] = match_no_trade.group(1).strip().replace('`', '').replace('\\', '')
            else:
                logging.warning(f"parse_trading_decision: NO TRADE regex also failed. Raw response: {response_text}")
                decision["rationale"] = response_text  # No trade regex 매칭 실패시 raw response 전체를 rationale로

    except Exception as e:
        logging.error(f"Error parsing Gemini response: {e}")
        decision["rationale"] = response_text  # parsing error시 raw response 전체를 rationale로

    logging.info("Parsed Trading Decision:")
    logging.info(decision)

    return decision


# =====================================================
# 11. 포지션 로깅 함수 (이전 코드와 동일)
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
# 12. 포지션 관리 및 메인 트레이딩 로직
# =====================================================

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
    Hyperliquid 거래소에 지정가 주문을 생성한다. (TP/SL 포함)
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

        # TP/SL 주문 side 결정 (진입 방향에 따라 다르게 설정)
        tp_sl_side = 'sell' if decision['final_action'] == 'GO LONG' else 'buy'  # <-- 조건부 side 설정

        # Hyperliquid는 여러 개의 주문을 하나의 list로 받는다.
        orders = [
            {  # 1. 지정가 매수 주문
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price,
            },
            {  # 2. Take Profit (TP) 주문
                'symbol': symbol,
                'type': order_type,
                'side': tp_sl_side,
                'amount': amount,
                'price': tp_price,
                'params': {'reduceOnly': True, 'triggerPrice': tp_price, 'takeProfitPrice': tp_price},
                # triggerPrice, stopPrice, takeProfitPrice 모두 시도
            },
            {  # 3. Stop Loss (SL) 주문
                'symbol': symbol,
                'type': order_type,
                'side': tp_sl_side,
                'amount': amount,
                'price': sl_price,
                'params': {'reduceOnly': True, 'stopLossPrice': sl_price, 'triggerPrice': sl_price},
                # triggerPrice, stopPrice, stopLossPrice 모두 시도
            },
        ]

        order_response = exchange.create_orders(orders)  # create_orders 함수 사용

        logging.info(f"Hyperliquid order created: {order_response}")
        return order_response

    except Exception as e:
        logging.error(f"Error creating order on Hyperliquid: {e}")
        return None


def close_expired_positions(symbol):
    """Hyperliquid 거래소에서 거래 기간이 만료된 포지션을 종료"""
    try:
        position = get_hyperliquid_position()
        orders = exchange.fetch_open_orders(symbol)

        if not orders and not position:
            return

        utc_entry_time_str = orders[0]['datetime']
        utc_entry_time = datetime.strptime(utc_entry_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")

        # 한국 시간대로 변환
        korea_timezone = pytz.timezone('Asia/Seoul')

        # UTC 시간대 정보 추가
        entry_time = pytz.utc.localize(utc_entry_time).astimezone(korea_timezone)
        df = pd.read_csv('trading_decisions.csv')
        trade_term = list(df['trade_term'])[-1]

        now_kst = datetime.now(KST)

        match = re.match(r"(\d+)([mhdw])", trade_term)
        if not match:
            logging.warning(f"Invalid trade_term format: {trade_term}")
            return

        term_value = int(match.group(1))
        term_unit = match.group(2)

        if term_unit == 'm':
            expiration_time = entry_time + timedelta(minutes=term_value)
        elif term_unit == 'h':
            expiration_time = entry_time + timedelta(hours=term_value)
        elif term_unit == 'd':
            expiration_time = entry_time + timedelta(days=term_value)
        elif term_unit == 'w':
            expiration_time = entry_time + timedelta(weeks=term_value)
        else:
            logging.warning(f"Invalid trade_term unit: {term_unit}")
            return

        if not position:
            if len(orders) >= 3:
                if term_unit == 'm':
                    if now_kst >= expiration_time:
                        order_ids = [order['id'] for order in orders]
                        exchange.cancel_orders(order_ids, symbol)
                        return
                else:
                    if now_kst >= entry_time + timedelta(minutes=30):
                        order_ids = [order['id'] for order in orders]
                        exchange.cancel_orders(order_ids, symbol)
                        return

        if now_kst >= expiration_time:
            logging.info(f"Closing expired position: {position}")
            try:
                close_side = 'sell' if position['side'] == 'long' else 'buy'
                current_price = exchange.fetch_ticker('BTC/USDC:USDC')['last']
                closing_order = exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=abs(float(position['contracts'])),
                    price=current_price,
                    params={'reduceOnly': True}
                )
                logging.info(f"Position closed: {closing_order}")
                # 텔레그램 메시지 전송 (telegram_utils.py 활용)
                from telegram_utils import send_telegram_message  # 상대 경로 import
                send_telegram_message(
                    f"*포지션 자동 종료* ({symbol})\n\n"
                    f"*만료 시간:* {expiration_time.strftime('%Y-%m-%d %H:%M:%S (KST)')}\n"
                    f"*사유:* 거래 기간({trade_term}) 만료"
                )

            except Exception as e:
                logging.error(f"Error creating market order on Hyperliquid in close expired positions: {e}")
                return None

    except Exception as e:
        logging.error(f"Error closing expired positions: {e}")


def calculate_position_size(balance, entry_price, leverage):
    """
    진입 가격과 레버리지를 고려하여 주문 수량을 계산한다. (10의 자리 버림)
    """
    # 사용 가능한 잔고 전체 사용 (10의 자리 버림)
    amount = (int(balance / 100) * 100) * leverage / entry_price
    # amount = int(balance * leverage / entry_price / 10) * 10
    return amount


def get_current_session_kst():
    """
    KST 기준 현재 시간, 미국 공휴일/주말 여부를 고려하여 세션 결정.
    """
    now_kst = datetime.now(KST)

    # 주말 여부 (KST 기준)
    is_weekend = now_kst.weekday() >= 5

    # 미국 공휴일 여부 (KST 기준)
    us_holidays = holidays.US(years=now_kst.year)
    is_us_holiday = now_kst.date() in us_holidays

    # KST 기준 세션
    hour_kst = now_kst.hour
    if 0 <= hour_kst < 8:
        session = "OVERNIGHT"
    elif 8 <= hour_kst < 16:
        session = "ASIAN"
    elif 16 <= hour_kst < 22:
        session = "LONDON"
    elif 22 <= hour_kst < 24 or 0 <= hour_kst < 6:
        session = "US"
    elif 6 <= hour_kst < 8:
        session = "TRANSITION"
    else:
        session = "UNDEFINED"

    if is_weekend:
        session += "_WEEKEND"
    if is_us_holiday:
        session += "_US_HOLIDAY"

    return session


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


def escape_markdown_v2(text):
    """
    Telegram Markdown V2에서 문제가 될 수 있는 모든 특수 문자를 이스케이프 처리.
    """
    escape_chars = r"[_*\[\]()~`>#\+\-=|{}\.!]"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


# =====================================================
# 13. 메인 함수
# =====================================================

def main():
    logging.info("Trading bot started.")

    # 1. 포지션 종료 (거래 기간 만료)
    close_expired_positions(HYPE_SYMBOL)

    # 2. 초기 포지션 확인
    position = get_hyperliquid_position()
    if position:
        return

    # 3. 잔고 확인
    balance = get_hyperliquid_balance()
    logging.info(f"Initial balance: {balance:.2f} USDC")

    # 4. 데이터 수집
    multi_tf_data = fetch_multi_tf_data(HYPE_SYMBOL, TIMEFRAMES, limit=300)
    if not multi_tf_data:
        logging.error("Failed to fetch multi-timeframe data.")
        return

    # 추가 데이터
    oi, oi_change = fetch_open_interest(SYMBOL)  # SYMBOL 사용 (BTC/USDT)
    additional_data = {
        "funding_rate": fetch_funding_rate(HYPE_SYMBOL),
        "open_interest": oi,
        "open_interest_change": oi_change,
        "order_book": fetch_order_book(HYPE_SYMBOL),  # Order Book 데이터 추가
        "fear_and_greed_index": fetch_fear_and_greed_index()
    }

    # 미국 주요 경제 지표 발표 요약
    econ_data_raw = fetch_economic_data()
    econ_summary = parse_economic_data(econ_data_raw)

    # 5. 현재 세션 확인
    current_session = get_current_session_kst()

    # 6. 시장 상황(장세) 결정
    market_regime = determine_market_regime(multi_tf_data, additional_data, current_session)
    thresholds = adjust_indicator_thresholds(market_regime, multi_tf_data)  # 지표 가중치

    # 7. 전략 선택
    strategy = select_strategy(market_regime)
    if not strategy:
        logging.info(f"No suitable strategy found for market regime: {market_regime}")
        return

    # 8. Gemini Pro를 이용한 최종 거래 결정
    try:
        fetch_liquidation_map()  # 청산맵 다운로드
        fetch_chart(strategy['primary_timeframe'])
        gemini_response_text = generate_trading_decision(
            multi_tf_data=multi_tf_data,
            market_regime=market_regime,
            strategy=strategy,
            thresholds=thresholds,
            current_session=current_session,
            additional_data=additional_data,
            econ_summary=econ_summary
        )
        logging.info(f"Gemini Raw Response: {gemini_response_text}")
        decision = parse_trading_decision(gemini_response_text)
        log_decision(decision, SYMBOL)

    except Exception as e:
        logging.error(f"Error in Gemini Pro interaction or decision parsing: {e}")
        return

    # 9. 거래 실행
    if decision['final_action'].upper() == 'NO TRADE':
        logging.info("No Trade")
        send_telegram_message(f"*거래 없음 (NO TRADE)*\n\n*이유:* {escape_markdown_v2(decision['rationale'])}")  # 텔레그램 메시지 전송
        return

    # 자동 매매 로직 (Hyperliquid)
    if decision['final_action'] in ['GO LONG', 'GO SHORT']:
        # 포지션 크기 계산
        current_price = multi_tf_data[strategy['primary_timeframe']]['current_price']  # 현재 가격
        amount = calculate_position_size(balance, current_price, float(decision['leverage'].replace('x', '')))
        decision['amount'] = str(amount)

        # 주문 생성 (이미 TP/SL 설정 포함)
        order = create_hyperliquid_order(HYPE_SYMBOL, decision, float(decision['leverage'].replace('x', '')))

        if order:
            # 거래 성공
            current_side = decision['final_action'].split()[-1]  # "LONG" or "SHORT"
            entry_price = float(decision['limit_order_price'])  # 체결 가격
            strategy_name = strategy["name"]
            strategy_description = strategy["description"]
            primary_tf = strategy["primary_timeframe"]

            # 거래 후 텔레그램 메시지 전송
            side_emoji = "🟢 매수" if current_side == "LONG" else "🔴 매도"
            side_emoji2 = "📉" if current_side == "SHORT" else "📈"
            message = (
                f"*{side_emoji} 포지션 진입* ({SYMBOL})\n\n"
                f"*레버리지:* {decision['leverage']}\n"
                f"*기간:* {decision['trade_term']}\n"
                f"*진입 가격:* {entry_price:.2f}\n"
                f"*목표 가격 (TP):* {decision['tp_price']}\n"
                f"*손절 가격 (SL):* {decision['sl_price']}\n"
                "=======================\n"
                f"*#{side_emoji2} 분석*\n"
                f"*Market Regime*: {market_regime.replace('_', ' ').upper()}\n"
                f"*Strategy*: {strategy_name} - {strategy_description}\n"
                f"Primary Time Frame: {primary_tf}\n"
                f"Gemini Analysis: {escape_markdown_v2(decision['rationale'])}\n"
            )
            send_telegram_message(message)
        else:
            # 거래
            message = (
                f"*거래 실패* ({SYMBOL})\n\n"
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
