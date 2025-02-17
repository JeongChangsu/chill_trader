import os
import re
import ta
import csv
import time
import ccxt
import pytz
import uuid
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
DECISIONS_LOG_FILE = "/Users/changpt/PycharmProjects/chill_trader/trading_logs/trading_decisions.csv"
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

# --- CSV 파일 관리를 위한 변수 추가 ---
TRADING_LOG_DIR = "/Users/changpt/PycharmProjects/chill_trader/trading_logs"  # 매매일지 저장 디렉토리
current_trading_log_version = None  # 현재 버전
current_trading_log_file = None  # 현재 버전의 CSV 파일 경로
trading_active = False  # 거래 활성화 플래그

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

def send_telegram_message(message, photo=None): #image도 받도록
    """
    Telegram API를 사용하여 메시지를 전송한다. (사진 첨부 기능 추가)
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
        if photo:  # 사진이 있으면 sendPhoto 사용
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': photo}
            # files와 params 같이 전송
            response = requests.post(url, data=params, files=files) # post로 변경
        else:  # 사진 없으면 기존 sendMessage 사용
            response = requests.get(url, params=params)  # requests.get 사용

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
# 6. 청산맵 다운로드 (이전 코드와 동일)
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
# 7. 시장 상황(장세) 결정 및 지표 임계값 조정
# =====================================================

def determine_market_regime(multi_tf_data, additional_data, current_session):  # current_session 추가
    """
    강화된 알고리즘을 사용하여 시장 상황(장세)을 결정.
    오픈 인터레스트, 펀딩 레이트 변화, 세션 정보 포함.
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
            trend = "bull"
        elif price_1d < ema200_1d:
            trend = "bear"

    # 3. 추세 강도 (Trend Strength) - ADX, DMI (1시간봉)
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

    # 6. 오픈 인터레스트, 펀딩 레이트 분석
    oi_change = additional_data.get("open_interest_change", 0)
    funding_rate = additional_data.get("funding_rate", 0)

    oi_fr_signal = "neutral"
    if oi_change > 1000000 and funding_rate > 0.05:
        oi_fr_signal = "bearish_reversal_likely"  # 롱 과열, 반전 가능성
    elif oi_change > 1000000 and funding_rate < -0.05:
        oi_fr_signal = "bullish_reversal_likely"  # 숏 과열, 반전 가능성
    elif oi_change > 500000 and -0.01 < funding_rate < 0.01:
        oi_fr_signal = "trend_continuation"  # 추세 지속
    elif oi_change < -500000:
        oi_fr_signal = "trend_weakening"  # 추세 약화

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

    # 세션별 특성 고려 (추가)
    if "ASIAN" in current_session:
        if "sideways" in market_regime:  # 아시아 세션 + 횡보
            market_regime = "tight_sideways"  # 타이트한 횡보로 간주

    if "LONDON" in current_session:
        if "sideways" in market_regime:
            market_regime = "normal_sideways"  # 런던 세션 + 횡보는 넓어질 가능성.
    if "US" in current_session:
        if "strong" in market_regime:
            market_regime += "_high_vol"  # us 세션 + 강한 추세

    logging.info(f"Market regime determined: {market_regime.upper()}")
    return market_regime


def adjust_indicator_thresholds(market_regime, multi_tf_data):
    """
    시장 상황에 따라 지표 임계값, ATR 배수, 지표 가중치를 동적으로 조정.
    """
    thresholds = {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "donchian_window": 20,
        "atr_multiplier_tp": 3,  # 기본값
        "atr_multiplier_sl": 2,  # 기본값
        "indicator_weights": {
            "rsi": 0.2,
            "macd": 0.2,
            "ema": 0.3,
            "donchian": 0.2,
            "volume": 0.1,
            "oi_fr": 0,  # 오픈 인터레스트 & 펀딩 레이트 가중치 (초기값)
        },
        "support_levels": [],  # 지지선 레벨 (Donchian, EMA 등)
        "resistance_levels": [],  # 저항선 레벨
    }

    # Donchian Channel, EMA를 지지/저항 레벨로 추가
    for tf, data in multi_tf_data.items():
        if data.get("donchian_lower") is not None:
            thresholds["support_levels"].append(round(data["donchian_lower"], 2))
        if data.get("donchian_upper") is not None:
            thresholds["resistance_levels"].append(round(data["donchian_upper"], 2))
        if data.get("ema20") is not None:
            thresholds["support_levels"].append(round(data["ema20"], 2))
            thresholds["resistance_levels"].append(round(data["ema20"], 2))
        if data.get("ema50") is not None:
            thresholds["support_levels"].append(round(data["ema50"], 2))
            thresholds["resistance_levels"].append(round(data["ema50"], 2))
        # 더 많은 지지/저항 레벨 추가 가능 (예: 피보나치 되돌림)

    # 중복 제거 및 정렬
    thresholds["support_levels"] = sorted(list(set(thresholds["support_levels"])))
    thresholds["resistance_levels"] = sorted(list(set(thresholds["resistance_levels"])))

    # 시장 상황에 따른 조정
    if "strong_bull_trend" in market_regime:
        thresholds["atr_multiplier_tp"] = 4  # 더 큰 목표 수익
        thresholds["atr_multiplier_sl"] = 2.5  # 약간 넓은 SL
        thresholds["indicator_weights"]["ema"] = 0.4
        thresholds["indicator_weights"]["volume"] = 0.2
        thresholds["indicator_weights"]["oi_fr"] = 0.1  # OI/FR 가중치 증가

    elif "weak_bull_trend" in market_regime:
        thresholds["atr_multiplier_tp"] = 3
        thresholds["atr_multiplier_sl"] = 1.8  # 비교적 타이트한 SL
        thresholds["indicator_weights"]["ema"] = 0.35
        thresholds["indicator_weights"]["rsi"] = 0.25
        thresholds["indicator_weights"]["oi_fr"] = 0.15

    elif "strong_bear_trend" in market_regime:
        thresholds["atr_multiplier_tp"] = 4
        thresholds["atr_multiplier_sl"] = 2.5
        thresholds["indicator_weights"]["ema"] = 0.4
        thresholds["indicator_weights"]["volume"] = 0.2
        thresholds["indicator_weights"]["oi_fr"] = 0.1

    elif "weak_bear_trend" in market_regime:
        thresholds["atr_multiplier_tp"] = 3
        thresholds["atr_multiplier_sl"] = 1.8
        thresholds["indicator_weights"]["ema"] = 0.35
        thresholds["indicator_weights"]["rsi"] = 0.25
        thresholds["indicator_weights"]["oi_fr"] = 0.15

    elif "tight_sideways" in market_regime:
        thresholds["atr_multiplier_tp"] = 2  # 짧은 TP
        thresholds["atr_multiplier_sl"] = 1.2  # 매우 타이트한 SL
        thresholds["indicator_weights"]["donchian"] = 0.4
        thresholds["indicator_weights"]["rsi"] = 0.3
        thresholds["indicator_weights"]["oi_fr"] = 0.1

    elif "wide_sideways" in market_regime:
        thresholds["atr_multiplier_tp"] = 2.5
        thresholds["atr_multiplier_sl"] = 1.7  # 비교적 넓은 SL
        thresholds["indicator_weights"]["donchian"] = 0.4
        thresholds["indicator_weights"]["rsi"] = 0.3

    elif "normal_sideways" in market_regime:
        thresholds["atr_multiplier_tp"] = 2.8
        thresholds["atr_multiplier_sl"] = 1.5

    # 변동성에 따른 추가 조정
    if "high_volatility" in market_regime:
        thresholds["atr_multiplier_sl"] += 0.5  # SL 더 넓게
        thresholds["indicator_weights"]["volume"] += 0.1
        thresholds["indicator_weights"]["atr"] = 0.2

    elif "low_volatility" in market_regime:
        thresholds["atr_multiplier_sl"] -= 0.2  # SL 더 타이트하게 (단, 너무 작지 않게)
        thresholds["indicator_weights"]["ema"] += 0.1
        thresholds["indicator_weights"]["atr"] = 0.05

    return thresholds


# =====================================================
# 8. 전략 템플릿 정의
# =====================================================

strategy_templates = {
    "strong_bull_trend_follow": {
        "name": "Strong Bull Trend Following (Momentum)",
        "description": "Follow the trend in a strong uptrend.",
        "primary_timeframe": "1d",
        "indicators": {
            "ema": {"weight": 0.4, "params": [20, 50, 200]},
            "rsi": {"weight": 0.2, "params": [14, 40, 80]},
            "macd": {"weight": 0.1, "params": []},
            "volume": {"weight": 0.3, "params": []},
            "oi_fr": {"weight": 0, "params": []}  # 초기값
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
            "oi_fr": {"weight": 0, "params": []}  # 초기값
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
        "leverage": "3x ~ 5x"  # 수정
    },
    "strong_bear_trend_follow": {
        "name": "Strong Bear Trend Following (Momentum)",
        "description": "Follow the trend in a strong downtrend.",
        "primary_timeframe": "1d",
        "indicators": {
            "ema": {"weight": 0.4, "params": [20, 50, 200]},
            "rsi": {"weight": 0.2, "params": [14, 20, 60]},
            "macd": {"weight": 0.1, "params": []},
            "volume": {"weight": 0.3, "params": []},
            "oi_fr": {"weight": 0, "params": []}  # 초기값
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
            "oi_fr": {"weight": 0, "params": []}  # 초기값
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
        "leverage": "3x ~ 5x"
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
            "ema": {"weight": 0.0, "params": []},
            "oi_fr": {"weight": 0, "params": []}  # 초기값
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
            "ema": {"weight": 0.0, "params": []},
            "oi_fr": {"weight": 0, "params": []}  # 초기값
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
            "ema": {"weight": 0.05, "params": []},
            "oi_fr": {"weight": 0, "params": []}  # 초기값
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
    },
    "bearish_reversal": {
        "name": "Bearish Reversal (OI & Funding Rate)",
        "description": "Identify potential bearish reversals based on high open interest and positive funding rate.",
        "primary_timeframe": "1h",
        "indicators": {
            "oi_fr": {"weight": 0.6, "params": []},  # OI & FR에 높은 가중치
            "rsi": {"weight": 0.2, "params": [14, 30, 70]},
            "ema": {"weight": 0.1, "params": [20, 50]},
            "volume": {"weight": 0.1, "params": []},
        },
        "entry_rules": {
            "short": [
                "oi_change > 1000000",  # OI 급증 (임계값 조정 필요)
                "funding_rate > 0.05",  # 높은 양의 펀딩 레이트
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2",
            "sl": "atr_multiplier * 1.5",  # 비교적 타이트한 SL
        },
        "trade_term": "4h ~ 12h",
        "leverage": "3x ~ 5x"
    },
    "bullish_reversal": {
        "name": "Bullish Reversal (OI & Funding Rate)",
        "description": "Identify potential bullish reversals based on high open interest and negative funding rate.",
        "primary_timeframe": "1h",
        "indicators": {
            "oi_fr": {"weight": 0.6, "params": []},
            "rsi": {"weight": 0.2, "params": [14, 30, 70]},
            "ema": {"weight": 0.1, "params": [20, 50]},
            "volume": {"weight": 0.1, "params": []},
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
        "description": "Identify potential trend continuation.",
        "primary_timeframe": "1h",
        "indicators": {
            "oi_fr": {"weight": 0.6, "params": []},
            "rsi": {"weight": 0.2, "params": [14, 30, 70]},
            "ema": {"weight": 0.1, "params": [20, 50]},
            "volume": {"weight": 0.1, "params": []},
        },
        "entry_rules": {
            "long": [
                "oi_change > 500000",
                "funding_rate > -0.01",
                "funding_rate < 0.01"
            ],
            "short": [
                "oi_change > 500000",
                "funding_rate > -0.01",
                "funding_rate < 0.01"
            ],
        },
        "exit_rules": {
            "tp": "atr_multiplier * 2",
            "sl": "atr_multiplier * 1.5",
        },
        "trade_term": "4h ~ 12h",
        "leverage": "3x ~ 5x"
    }
}


# =====================================================
# 9. 전략 선택 함수
# =====================================================

def select_strategy(market_regime):
    """
    결정된 시장 상황(장세)에 따라 가장 적합한 전략을 선택합니다.
    """
    if "strong_bull_trend" in market_regime:
        if "trend_continuation" in market_regime:
            return strategy_templates["trend_continuation"]
        return strategy_templates["strong_bull_trend_follow"]
    elif "weak_bull_trend" in market_regime:
        return strategy_templates["weak_bull_trend_pullback"]
    elif "strong_bear_trend" in market_regime:
        if "trend_continuation" in market_regime:
            return strategy_templates["trend_continuation"]
        return strategy_templates["strong_bear_trend_follow"]
    elif "weak_bear_trend" in market_regime:
        return strategy_templates["weak_bear_trend_bounce"]
    elif "tight_sideways" in market_regime:
        return strategy_templates["tight_sideways_range"]
    elif "wide_sideways" in market_regime:
        return strategy_templates["wide_sideways_range"]
    elif "normal_sideways" in market_regime:
        return strategy_templates["normal_sideways_range"]
    elif "bearish_reversal_likely" in market_regime:  # 추가
        return strategy_templates["bearish_reversal"]
    elif "bullish_reversal_likely" in market_regime:  # 추가
        return strategy_templates["bullish_reversal"]
    else:
        return None  # 적합한 전략 없음


# =====================================================
# 10. Gemini 프롬프트 생성 및 거래 결정
# =====================================================
def generate_gemini_prompt(multi_tf_data, market_regime, strategy, thresholds,
                           current_session,
                           additional_data, econ_summary):  # econ_summary 파라미터 추가
    """
    Gemini Pro 모델에 전달할 Prompt를 생성합니다. (전략 템플릿 기반)
    """

    # 전략 정보
    strategy_name = strategy["name"]
    strategy_description = strategy["description"]
    primary_tf = strategy["primary_timeframe"]

    # 지표 요약 (간결하게)
    indicators_summary = ""
    for tf, data in multi_tf_data.items():
        indicators_summary += f"**{tf}:**\n"
        indicators_summary += f"  - Price: {data['current_price']:.2f}\n"
        for ind_name, ind_params in strategy["indicators"].items():
            if ind_name in data and data[ind_name] is not None:
                # 지표가 'oi_fr' (Open Interest & Funding Rate)가 아닐 때만 값 출력
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
    now_kst = datetime.now(pytz.timezone('Asia/Seoul'))
    current_time_kst = now_kst.strftime("%Y-%m-%d %H:%M:%S (KST)")

    # 세션별 가이드 (예시)
    session_guides = {
        "OVERNIGHT": "Low liquidity. Be cautious of fake breakouts. Use tighter stops.",
        "ASIAN": "08:00-09:00 KST: Potential volatility spike. After 09:00: Trend may develop.",
        "LONDON": "16:00 KST open: Expect high volatility. Trade the dominant trend.",
        "US": "Highest volume and volatility. Be prepared for reversals.",
        "TRANSITION": "Low liquidity, potential trend formation before Asian open.",
        "ASIAN_WEEKEND": "Lower liquidity, increased volatility. Be cautious.",
        "LONDON_WEEKEND": "Lower liquidity, increased volatility. Be cautious.",
        "US_WEEKEND": "Lower liquidity, increased volatility. Be cautious. Watch for sudden price swings.",
        "US_US_HOLIDAY": "US market closed. Expect lower liquidity and potentially erratic price movements."
    }
    session_guide = session_guides.get(current_session, "No specific guidance for this session.")

    # 지지/저항 정보 추가 (프롬프트에)
    support_levels_str = ", ".join([f"{level:.2f}" for level in thresholds["support_levels"]])
    resistance_levels_str = ", ".join([f"{level:.2f}" for level in thresholds["resistance_levels"]])

    prompt_text_1 = f"""
**Objective:** Make optimal trading decisions for BTC/USDT.

**Market Context:**
- Regime: **{market_regime.upper()}**
- Strategy: **{strategy_name}** ({strategy_description})
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
**Liquidation Map Analysis Guide(Image Provided):**
- **Support and Resistance:** Identify potential support and resistance levels based on liquidation clusters.
- **Cascading Liquidations:** Assess the risk of cascading liquidations (large clusters close to the current price).
- **Volatility Prediction:** Estimate potential volatility based on the distance between liquidation clusters.
- **Risk Assessment:** Compare long vs. short liquidation levels to gauge overall market risk.
- **If the liquidation map provides clear support/resistance levels, use them to inform your TP/SL decisions. If the map is unclear or provides no strong signals, you may rely more on other indicators and market context.** 

**Task:**

Based on all provided information, decide: **GO LONG, GO SHORT, or NO TRADE.**

If GO LONG or GO SHORT, also determine:
- **Recommended Leverage:** (Based on strategy's recommendation)
- **Trade Term:** (Based on strategy's recommendation)
- **Take Profit Price:** (Consider setting TP near resistance levels (for long positions) or support levels (for short positions).  You can also use the ATR-based suggestion or a combination of both.)
- **Stop Loss Price:** (Consider setting SL near support levels (for long positions) or resistance levels (for short positions). You can also use the ATR-based suggestion or a combination of both. Prioritize risk management:  **Your SL should never risk more than 5% of your account balance.**)
- **Limit Order Price:**
- **Rationale:** (Explain your decision, including liquidation map, indicators, and market context.)

**Output Format (Comma-Separated):**

Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Rationale
"""

    return prompt_text_1, prompt_text_2


def generate_trading_decision(multi_tf_data, market_regime, strategy, thresholds, current_session, additional_data,
                              econ_summary):  # 파라미터 추가
    """
    Gemini Pro 모델을 통해 프롬프트를 전달하고, 거래 결정을 받아온다. (청산맵 이미지 포함)
    """
    prompt_part_1, prompt_part_2 = generate_gemini_prompt(multi_tf_data, market_regime, strategy, thresholds,
                                                          current_session, additional_data, econ_summary)

    image_path = "/Users/changpt/Downloads/Liquidation Map.png"  # 청산맵 이미지 경로 (실제 경로로 수정)
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
        match = re.search(r"GO (LONG|SHORT).*?,(.*?)x, *(.*?), *(.*?), *(.*?), *(.*?), *(.*)", response_text,
                          re.DOTALL | re.IGNORECASE)
        if match:
            decision["final_action"] = f"GO {match.group(1).upper()}"
            decision["leverage"] = match.group(2).strip()
            decision["trade_term"] = match.group(3).strip()
            decision["tp_price"] = match.group(4).strip()
            decision["sl_price"] = match.group(5).strip()
            decision["limit_order_price"] = match.group(6).strip()
            decision["rationale"] = match.group(7).strip()

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
                'side': 'sell',
                'amount': amount,
                'price': tp_price,
                'params': {'reduceOnly': True, 'triggerPrice': tp_price, 'stopPrice': tp_price,
                           'takeProfitPrice': tp_price},  # triggerPrice, stopPrice, takeProfitPrice 모두 시도
            },
            {  # 3. Stop Loss (SL) 주문
                'symbol': symbol,
                'type': order_type,
                'side': 'sell',
                'amount': amount,
                'price': sl_price,
                'params': {'reduceOnly': True, 'stopLossPrice': sl_price, 'triggerPrice': sl_price,
                           'stopPrice': sl_price},  # triggerPrice, stopPrice, stopLossPrice 모두 시도
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
    KST 기준 현재 시간을 확인하여 트레이딩 세션 및 주말/미국 공휴일 여부 결정
    """
    now = datetime.now(KST)
    hour = now.hour

    # 주말 여부 확인
    is_weekend = now.weekday() >= 5  # 5: 토요일, 6: 일요일

    # 미국 공휴일 여부 확인
    us_holidays = holidays.US()  # 미국 공휴일 객체 생성
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
        session += "_WEEKEND"  # 주말
    if is_us_holiday:
        session += "_US_HOLIDAY"  # 미국 공휴일

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

            # 거래 후 텔레그램 메시지 전송
            side_emoji = "🟢 매수" if current_side == "LONG" else "🔴 매도"
            message = (
                f"*{side_emoji} 포지션 진입* ({SYMBOL})\n\n"
                f"*레버리지:* {decision['leverage']}\n"
                f"*기간:* {decision['trade_term']}\n"
                f"*진입 가격:* {entry_price:.2f}\n"
                f"*목표 가격 (TP):* {decision['tp_price']}\n"
                f"*손절 가격 (SL):* {decision['sl_price']}\n\n"
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
