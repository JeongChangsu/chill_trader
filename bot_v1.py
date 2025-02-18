import os
import re
import time
import json
import pytz
import glob
import ccxt
import logging
import requests
import pandas_ta

import pandas as pd
import undetected_chromedriver as uc

from PIL import Image  # 이미지 처리
from google import genai
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from workalendar.usa import UnitedStates  # 미국 공휴일 정보
from selenium.webdriver.common.by import By

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants (환경 변수 등)
directory_path = "/Users/changpt/Downloads/"
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
HYPERLIQUID_WALLET_ADDRESS = os.environ.get('HYPE_ADDRESS')
HYPERLIQUID_PRIVATE_KEY = os.environ.get('HYPE_PRIVATE_KEY')
KST = pytz.timezone('Asia/Seoul')
DECISIONS_LOG_FILE = f'/Users/changpt/PycharmProjects/chill_trader/decisions_log.csv'  # 의사결정 로깅 파일

liquidation_map_path = "/Users/changpt/Downloads/Liquidation Map.png"
prefix_to_match = "BTCUSD"

# Telegram variables (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

exchange = ccxt.hyperliquid({
    'walletAddress': HYPERLIQUID_WALLET_ADDRESS,
    'privateKey': HYPERLIQUID_PRIVATE_KEY,
    'options': {
        'defaultType': 'swap',  # 선물 거래를 위한 설정
    },
})


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
    elif "w" in tf:
        weeks = tf.replace("w", "")
        return f"{weeks} week{'s' if weeks != '1' else ''}"
    return tf


def get_driver():
    """
    undetected_chromedriver의 새로운 인스턴스를 생성하여 반환한다.
    """
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options)
    return driver


def fetch_all_chart():
    """
    청산 히트맵 데이터를 CoinAnk 사이트에서 다운로드한다.
    """
    url = "https://www.tradingview.com/chart/?symbol=BITSTAMP%3ABTCUSD"
    timeframes = ['15m', '1h', '4h', '1d']

    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(2)  # Increased wait time

        for tf in timeframes:
            tf_str = timeframe_str_for_chart(tf)

            driver.find_element(By.XPATH, '//div[@id="header-toolbar-intervals"]//button').click()
            time.sleep(1)
            driver.find_element(By.XPATH, f'//span[text()="{tf_str}"]').click()
            time.sleep(2)
            chart_screenshot(driver)
            time.sleep(1)

        logging.info("Chart fetched successfully")
        driver.close()
        driver.quit()

    except Exception as e:
        logging.error(f"Error fetching chart data: {e}")


def chart_screenshot(driver):
    driver.find_element(By.XPATH, '//div[@id="header-toolbar-screenshot"]').click()
    time.sleep(1)
    driver.find_element(By.XPATH, '//span[text()="Download image"]').click()
    time.sleep(1)


def fetch_liquidation_map():
    """
    청산 히트맵 데이터를 CoinAnk 사이트에서 다운로드한다.
    """
    url = "https://coinank.com/liqMapChart"
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(3)  # Increased wait time

        if driver.find_element(By.XPATH, '//span[@class="anticon anticon-camera"]').is_displayed():
            driver.find_element(By.XPATH, '//span[@class="anticon anticon-camera"]').click()
            time.sleep(2)
            driver.quit()

        logging.info("Liquidation map data fetched successfully")
    except Exception as e:
        logging.error(f"Error fetching liquidation map data: {e}")


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
            event_impact_str = event_impact_element.get('data_img_key', 'low')  # 속성에서 중요도 추출, default 'low'
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


def get_ohlcv(exchange, symbol, timeframe, limit):
    """OHLCV 데이터 가져오기"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching OHLCV data: {e}")
        return None


def get_funding_rate(exchange, symbol):
    """펀딩비 가져오기"""
    try:
        funding_rate = exchange.fetch_funding_rate(symbol=symbol)
        return funding_rate['fundingRate']
    except Exception as e:
        logging.error(f"Error fetching funding rate: {e}")
        return None


def get_open_interest(exchange, symbol):
    """미결제약정 가져오기"""
    try:
        open_interest = exchange.fetch_open_interest(symbol)
        if isinstance(open_interest, list):  # 여러 OI가 반환될 경우
            # symbol에 해당하는 OI 찾기
            for oi in open_interest:
                if oi['symbol'] == symbol:
                    return oi['openInterestAmount']
            return None  # symbol에 맞는 OI를 찾지 못한 경우
        else:  # 단일 OI가 반환될 경우
            return open_interest['openInterestAmount']
    except Exception as e:
        logging.error(f"Error fetching open interest: {e}")
        return None


def get_put_call_ratio(exchange, symbol):
    """
    풋/콜 비율 가져오기 (Deribit 예시).
    다른 거래소의 경우, 해당 거래소 API 문서 참조하여 수정 필요.
    """
    try:
        # Deribit API 호출 (BTC-PERPETUAL 옵션 데이터)
        response = requests.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_instrument?instrument_name=BTC-PERPETUAL")
        response.raise_for_status()
        data = response.json()

        # 풋/콜 비율 계산
        total_puts = 0
        total_calls = 0
        for item in data['result']:
            if 'option_type' in item:
                if item['option_type'] == 'put':
                    total_puts += item['open_interest']
                elif item['option_type'] == 'call':
                    total_calls += item['open_interest']

        if total_calls > 0:
            put_call_ratio = total_puts / total_calls
            return put_call_ratio
        else:
            return None

    except Exception as e:
        logging.error(f"Error fetching put/call ratio: {e}")
        return None


def get_implied_volatility(exchange, symbol):
    """
    내재 변동성 가져오기 (Deribit DVOL 지수 사용).
    다른 거래소/데이터 제공자를 사용하는 경우, 해당 API 문서 참조하여 수정 필요.
    """
    try:
        # Deribit API 호출 (DVOL 지수)
        response = requests.get("https://www.deribit.com/api/v2/public/get_index?currency=BTC")
        response.raise_for_status()
        data = response.json()
        return data['result']['DVOL']

    except Exception as e:
        logging.error(f"Error fetching implied volatility: {e}")
        return None


def get_market_data(exchange, symbol):
    """
    Hyperliquid 거래소에서 시장 데이터를 가져온다.
    """
    data = {}

    # OHLCV data (여러 timeframe)
    timeframes = ['15m', '1h', '4h', '1d', '1w']
    for tf in timeframes:
        data[f'ohlcv_{tf}'] = get_ohlcv(exchange, symbol, tf, 100)  # 100개

    # 펀딩비
    data['funding_rate'] = get_funding_rate(exchange, symbol)

    # 미결제약정
    data['open_interest'] = get_open_interest(exchange, symbol)

    return data


def calculate_technical_indicators(df):
    """
    기술적 지표 (MA, MACD, RSI) 계산.
    """
    # 이동평균 (MA)
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()

    # MACD
    macd = df.ta.macd()
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']

    # RSI
    df['RSI'] = df.ta.rsi()

    return df


def get_recent_high_low(df, days):
    """
    최근 N일 동안의 고점/저점 계산.
    """
    # days가 DataFrame의 길이보다 크면, DataFrame의 길이로 조정
    period = min(days, len(df))
    high = df['high'].tail(period).max()
    low = df['low'].tail(period).min()
    return high, low


def calculate_fibonacci_levels(high, low):
    """
    피보나치 되돌림 레벨 계산.
    """
    diff = high - low
    levels = {
        '0.236': low + diff * 0.236,
        '0.382': low + diff * 0.382,
        '0.5': low + diff * 0.5,
        '0.618': low + diff * 0.618,
        '0.786': low + diff * 0.786
    }
    return levels


def is_us_holiday():
    """오늘이 미국 공휴일인지 확인"""
    cal = UnitedStates()
    today = datetime.now(KST).date()
    return cal.is_working_day(today)


def create_prompt(market_data, economic_data_summary, liquidation_map_pil, chart_images, fear_greed_index):
    """
    Gemini 모델에 전달할 프롬프트 생성.
    """
    prompt_parts = []

    # Part 1: System instruction (always first)
    system_instruction = """
You are a world-class, highly successful cryptocurrency trader specializing in Bitcoin.
You make consistent profits and have a deep understanding of market dynamics, technical analysis, and risk management.
You are cautious and prioritize capital preservation.
Your trading decisions are based on a combination of technical and fundamental analysis, as well as market sentiment.
Your time unit is Korean Standard Time (KST).

Analyze the provided data and provide a trading decision with detailed reasoning.

Your output format should be strictly JSON:
{
"action": "GO LONG" or "GO SHORT" or "NO TRADE",
"limit_order_price": "string",
"leverage": "string",
"tp_price": "string",
"sl_price": "string",
"perspective_duration": "string",
"chart_analysis": "string",
"liquidation_map_analysis": "string",
"overall_reasoning": "string"
}

- 'limit_order_price', 'tp_price', and 'sl_price' should be numbers, represented as strings.
- 'perspective_duration' format : "number + unit", ex. 5m, 10m, 30m, 1h, 4h, 1d, 2d ...
- Each analysis (chart, liquidation map, economic events) should be concise, maximum 3 sentences.
- Overall reasoning should clearly state your decision and the combined factors.
- If it is a weekend or US holiday, reflect that in your analysis.
- Leverage: always use 3x ~ 5x.
- Consider price, volume, open interest, funding rates, chart patterns, and Fibonacci levels.
- Always use "fibonacci" when using information from fibonacci.
- "limit_order_price" should be determined by considering support and resistance level from chart and liquidation map.
- Set "tp_price" and "sl_price" considering recent volatility and fibonacci levels.
- Set 'perspective_duration', which represents how long your trading perspective will be maintained, considering current volatility and market conditions.
"""
    chart_guide = """
**Additional Chart Analysis Guide:**
- Identify candle patterns (e.g., hammer, shooting star, engulfing) and interpret their meanings.
- Check for volume spikes/drops and analyze their correlation with price movements.
- Look for golden cross/dead cross occurrences between moving averages.
"""
    liquidation_guide = """
**Additional Liquidation Map Analysis Guide:**
- If liquidation levels are concentrated at a specific price level, it is highly likely to act as strong support/resistance.
- If liquidation levels are skewed in one direction, price movement may accelerate in that direction.
"""

    # Part 2: Market data (text-based)
    market_data_text = f"""
## Current Market Data (BTC/USDC)

**Date:** {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} (KST)
**Weekend/US Holiday:** {is_us_holiday()}
**Fear & Greed Index:** {fear_greed_index}

**Economic Events:**
{economic_data_summary}

**15m OHLCV:**
{market_data['ohlcv_15m'].tail(5).to_string()}

**1h OHLCV:**
{market_data['ohlcv_1h'].tail(5).to_string()}

**4h OHLCV:**
{market_data['ohlcv_4h'].tail(5).to_string()}

**1d OHLCV:**
{market_data['ohlcv_1d'].tail(5).to_string()}

**Funding Rate:** {market_data['funding_rate']}
**Open Interest:** {market_data['open_interest']}
"""

    # Part 3: Technical Indicators (calculated)
    indicator_text = ""
    for tf in ['15m', '1h', '4h', '1d']:
        df = market_data[f'ohlcv_{tf}']
        df = calculate_technical_indicators(df)
        high_30, low_30 = get_recent_high_low(df, 30)  # 30-period high/low
        fib_levels_30 = calculate_fibonacci_levels(high_30, low_30)
        high_90, low_90 = get_recent_high_low(df, 90)  # 90-period high/low
        fib_levels_90 = calculate_fibonacci_levels(high_90, low_90)

        indicator_text += f"""
**{tf} Indicators:**
MA20: {df['MA20'].iloc[-1]:.2f}
MA50: {df['MA50'].iloc[-1]:.2f}
MA200: {df['MA200'].iloc[-1]:.2f}
MACD: {df['MACD'].iloc[-1]:.4f}
MACD_hist: {df['MACD_hist'].iloc[-1]:.4f}
MACD_signal: {df['MACD_signal'].iloc[-1]:.4f}
RSI: {df['RSI'].iloc[-1]:.2f}
30-period High: {high_30:.2f}, Low: {low_30:.2f}
30-period Fibonacci: {fib_levels_30}
90-period High: {high_90:.2f}, Low: {low_90:.2f}
90-period Fibonacci: {fib_levels_90}
"""
    prompt_parts.append(system_instruction)
    prompt_parts.append(market_data_text)
    prompt_parts.append(indicator_text)

    # Part 4: Images (liquidation map, charts) - PIL Image objects
    prompt_parts.append(liquidation_guide)
    prompt_parts.append(liquidation_map_pil)

    prompt_parts.append(chart_guide)
    prompt_parts += chart_images

    return prompt_parts


def analyze_market(prompt_parts):
    """
    Gemini 모델을 사용하여 시장 분석.
    """
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    gemini_client = genai.Client(api_key=google_api_key)

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=prompt_parts
    )

    response_text = re.sub(r"```json\n?(.*?)\n?```", r"\1", response.text, flags=re.DOTALL)

    try:
        # print(response.text)
        return json.loads(response_text)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON response from Gemini: {response.text}")
        return None


def create_hyperliquid_order(symbol, decision, leverage, amount):
    """
    Hyperliquid 거래소에 지정가 주문을 생성한다. (TP/SL 포함)
    """
    try:
        order_type = 'limit'
        side = 'buy' if decision['action'] == 'GO LONG' else 'sell'
        amount = round(float(amount), 5)
        price = float(decision['limit_order_price'])  # 지정가
        btc_amount = round(amount * leverage / price, 5)

        # TP/SL 가격 (문자열 -> 숫자)
        tp_price = float(decision['tp_price'])
        sl_price = float(decision['sl_price'])

        # TP/SL 가격 검증 및 조정
        if decision['action'] == 'GO LONG':
            if tp_price <= price:  # TP가 진입 가격보다 낮거나 같으면
                return None
                # tp_price = price + (price * 0.01)  # 진입 가격보다 1% 높게 설정 (예시)
            if sl_price >= price:  # SL가격이 진입 가격보다 높거나 같으면
                return None
                # sl_price = price - (price * 0.01)
        elif decision['action'] == 'GO SHORT':
            if tp_price >= price:  # TP가 진입 가격보다 높거나 같으면
                return None
                # tp_price = price - (price * 0.01)  # 진입 가격보다 1% 낮게 설정 (예시)
            if sl_price <= price:  # SL가격이 진입 가격보다 낮거나 같으면
                return None
                # sl_price = price + (price * 0.01)

        exchange.set_margin_mode('isolated', symbol, params={'leverage': leverage})

        # TP/SL 주문 side 결정 (진입 방향에 따라 다르게 설정)
        tp_sl_side = 'sell' if decision['action'] == 'GO LONG' else 'buy'  # <-- 조건부 side 설정

        # Hyperliquid는 여러 개의 주문을 하나의 list로 받는다.
        orders = [
            {  # 1. 지정가 매수 주문
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': btc_amount,
                'price': price,
            },
            {  # 2. Take Profit (TP) 주문
                'symbol': symbol,
                'type': order_type,
                'side': tp_sl_side,
                'amount': btc_amount,
                'price': tp_price,
                'params': {'reduceOnly': True, 'triggerPrice': tp_price, 'takeProfitPrice': tp_price},
                # triggerPrice, stopPrice, takeProfitPrice 모두 시도
            },
            {  # 3. Stop Loss (SL) 주문
                'symbol': symbol,
                'type': order_type,
                'side': tp_sl_side,
                'amount': btc_amount,
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
    """Hyperliquid 거래소에서 거래 기간(trade term)이 만료된 포지션을 종료"""
    try:
        position = get_hyperliquid_position()  # 현재 포지션 정보
        orders = exchange.fetch_open_orders(symbol)  # 현재 열려있는 주문 목록
        now_kst = datetime.now(KST)  # 현재 시간 (KST) - 함수 시작 부분에 정의

        if not position:  # 현재 포지션 없으면
            if len(orders) >= 2:  # 지정가 주문, TP, SL (총 3개) 주문이 있으면
                # 주문 생성 시간 가져오기 (orders[0]에 있다고 가정)
                utc_entry_time_str = orders[0]['datetime']
                utc_entry_time = datetime.strptime(utc_entry_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                korea_timezone = pytz.timezone('Asia/Seoul')
                entry_time = pytz.utc.localize(utc_entry_time).astimezone(
                    korea_timezone)  # entry_time 정의 (KST)

                if now_kst >= entry_time + timedelta(minutes=30):  # 주문 후 30분 간 미체결 시 취소
                    order_ids = [order['id'] for order in orders]
                    exchange.cancel_orders(order_ids, symbol)
            return  # 포지션 없으면 함수 종료

        # 포지션 있으면 (trade term 만료 확인)
        # 1. 의사결정 로그 파일에서 trade_term 읽어오기
        df = pd.read_csv(DECISIONS_LOG_FILE)
        if df.empty:  # 로그 파일 비어있으면 리턴
            return

        trade_term = list(df['trade_term'])[-1]  # 가장 최근 trade_term

        # 2. trade_term 파싱 (정규표현식 사용)
        match = re.match(r"(\d+)([mhdw])", trade_term)
        if not match:
            logging.warning(f"Invalid trade_term format: {trade_term}")
            return

        term_value = int(match.group(1))
        term_unit = match.group(2)

        # 3. 포지션 진입 시간 (entry_time) 가져오기
        # (주의) Hyperliquid API에서 포지션 진입 시간 확인 필요!
        #       orders[0]['datetime'] 사용 (가정).  -> position 객체에 있으면 수정.
        if orders:
            utc_entry_time_str = orders[0]['datetime']
            utc_entry_time = datetime.strptime(utc_entry_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            korea_timezone = pytz.timezone('Asia/Seoul')
            entry_time = pytz.utc.localize(utc_entry_time).astimezone(korea_timezone)  # entry_time 정의 (KST)

        # 4. 만료 시간 (expiration_time) 계산
        if term_unit == 'm':
            expiration_time = entry_time + timedelta(minutes=term_value)
        elif term_unit == 'h':
            expiration_time = entry_time + timedelta(hours=term_value)
        elif term_unit == 'd':
            expiration_time = entry_time + timedelta(days=term_value)
        elif term_unit == 'w':
            expiration_time = entry_time + timedelta(weeks=term_value)
        else:
            logging.warning(f"Invalid trade_term unit: {term_unit}")  # 위에서 걸러지지만..
            return

        # 5. 현재 시간 (KST)과 비교하여 만료 여부 확인

        if now_kst >= expiration_time:
            logging.info(f"Closing expired position: {position}")

            # 6. 시장가 주문으로 포지션 종료
            try:
                close_side = 'sell' if position['side'] == 'long' else 'buy'
                current_price = exchange.fetch_ticker('BTC/USDC:USDC')['last']  # 현재 시장가
                closing_order = exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=abs(float(position['contracts'])),  # 포지션 크기 (절대값)
                    price=current_price,  # 시장가 (Hyperliquid에서 market order는 price 필수)
                    params={'reduceOnly': True}  # 포지션 감소 only
                )
                logging.info(f"Position closed: {closing_order}")

                # 텔레그램 메시지 전송 (옵션)
                send_telegram_message(
                    f"*포지션 자동 종료* ({symbol})\n\n"
                    f"*만료 시간:* {expiration_time.strftime('%Y-%m-%d %H:%M:%S (KST)')}\n"
                    f"*사유:* 거래 기간({trade_term}) 만료"
                )

            except Exception as e:
                logging.error(f"Error creating market order to close position: {e}")
                return None  # 포지션 종료 실패

    except Exception as e:
        logging.error(f"Error closing expired positions: {e}")


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


def get_fear_and_greed_index():
    """Alternative.me API에서 공포-탐욕 지수 가져오기"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"  # limit=1 for only the latest value
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        # Check if the data is valid and contains the expected structure
        if data and data['data'] and len(data['data']) > 0:
            value = data['data'][0]['value']
            value_classification = data['data'][0]['value_classification']
            return f"{value} ({value_classification})"
        else:
            logging.warning("Unexpected data structure from fear and greed index API.")
            return "N/A"  # Return a default value indicating unavailability
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching fear and greed index: {e}")
        return "N/A"  # Return a default value in case of an error


def log_decision(decision):
    """
    AI의 의사결정을 CSV 파일에 로깅.
    """
    # CSV 파일 헤더 (첫 실행 시에만 생성)
    if not os.path.exists(DECISIONS_LOG_FILE):
        with open(DECISIONS_LOG_FILE, 'w') as f:
            f.write(
                "timestamp,action,limit_order_price,leverage,tp_price,sl_price,"
                "perspective_duration,chart_analysis,liquidation_map_analysis,"
                "overall_reasoning\n"
            )

    # 의사결정 내용 (따옴표 및 None 값 처리)
    row = [
        datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S'),
        decision.get('action', '').replace('"', '') or '',  # None이면 빈 문자열
        decision.get('limit_order_price', '') or '',  # None이면 빈 문자열, 따옴표 제거
        decision.get('leverage', '') or '',  # None이면 빈 문자열, 따옴표 제거
        decision.get('tp_price', '') or '',  # None이면 빈 문자열, 따옴표 제거
        decision.get('sl_price', '') or '',  # None이면 빈 문자열, 따옴표 제거
        decision.get('perspective_duration', '').replace('"', '') or '',  # None이면 빈 문자열
        decision.get('chart_analysis', '').replace('"', '') or '',  # None이면 빈 문자열
        decision.get('liquidation_map_analysis', '').replace('"', '') or '',  # None이면 빈 문자열
        decision.get('overall_reasoning', '').replace('"', '') or ''  # None이면 빈 문자열
    ]
    # CSV 파일에 추가
    with open(DECISIONS_LOG_FILE, 'a') as f:
        f.write(','.join(str(x) for x in row) + '\n')


def clear_old_images():
    """
    이전 이미지 파일들을 삭제합니다.
    """
    if os.path.exists(liquidation_map_path):
        os.remove(liquidation_map_path)

    # 차트 초기화
    directory_path = "/Users/changpt/Downloads/"
    prefix_to_match = "BTCUSD"
    pattern = os.path.join(directory_path, f"{prefix_to_match}*.png")
    files_to_delete = glob.glob(pattern)
    for chart in files_to_delete:
        os.remove(chart)


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


def format_telegram_message(decision, amount):
    """
    텔레그램 메시지 형식 생성
    """
    action = decision['action']
    if action == 'GO LONG':
        side_emoji = "🟢 매수"
        side_emoji2 = "📈"
    elif action == 'GO SHORT':
        side_emoji = "🔴 매도"
        side_emoji2 = "📉"
    else:
        return "No trade action."

    message = (
        f"*{side_emoji} 포지션 진입* (BTC/USDC)\n\n"
        f"*레버리지:* {decision['leverage']}\n"
        f"*관점 유지 기간:* {decision['perspective_duration']}\n"
        f"*진입 가격:* {float(decision['limit_order_price']):.2f}\n"
        f"*수량:* {amount} USDC\n"
        f"*목표 가격 (TP):* {decision['tp_price']}\n"
        f"*손절 가격 (SL):* {decision['sl_price']}\n"
        "=======================\n"
        f"*{side_emoji2} 분석*\n\n"
        f"*차트 분석*: {decision['chart_analysis']}\n\n"
        f"*청산맵 분석*: {decision['liquidation_map_analysis']}\n\n"
        f"*종합 판단 근거*: {decision['overall_reasoning']}"
    )
    return message


def main():
    """
    주기적으로 실행되며, 시장 분석 및 거래 로직 수행.
    """
    symbol = 'BTC/USDC:USDC'

    # Hyperliquid 거래소 객체 생성
    exchange = ccxt.hyperliquid({
        'walletAddress': HYPERLIQUID_WALLET_ADDRESS,
        'privateKey': HYPERLIQUID_PRIVATE_KEY,
        'options': {
            'defaultType': 'swap',  # 선물 거래
        },
    })

    # 만료된 포지션 정리 (여기서는 30분 미체결 주문 취소만 처리)
    close_expired_positions(symbol)

    # 이전 이미지 파일 삭제
    clear_old_images()

    # 1. 데이터 수집
    market_data = get_market_data(exchange, symbol)
    economic_data = fetch_economic_data()
    fear_greed_index = get_fear_and_greed_index()
    economic_data_summary = parse_economic_data(economic_data)
    fetch_liquidation_map()
    fetch_all_chart()

    # 이미지 파일 PIL 객체로 변환
    liquidation_map_pil = Image.open(liquidation_map_path) if liquidation_map_path else None
    pattern = os.path.join(directory_path, f"{prefix_to_match}*.png")
    chart_paths = glob.glob(pattern)
    chart_list = [Image.open(chart_path) for chart_path in chart_paths]

    # 2. 프롬프트 생성
    prompt_parts = create_prompt(
        market_data, economic_data_summary, liquidation_map_pil,
        chart_list, fear_greed_index
    )

    # 3. 시장 분석 (Gemini)
    decision = analyze_market(prompt_parts)
    if not decision:
        logging.warning("Could not get a valid decision from Gemini, skipping this cycle.")
        return

    # 의사결정 로깅
    log_decision(decision)
    print(decision)

    # 4. 거래 실행 (Hyperliquid)
    # 사용 가능한 잔고 확인 (최소 10 USDC는 남겨둠)
    available_balance = get_hyperliquid_balance() - 10
    if available_balance <= 0:
        logging.info("Insufficient balance to place an order.")
        return

    if decision['action'] != 'NO TRADE':
        # 포지션이 없는 경우에만 주문
        if not get_hyperliquid_position():
            # 결정된 액션이 'NO TRADE'가 아닌 경우에만 주문 생성
            amount = str(available_balance)  # 사용 가능한 잔고 전체 사용
            order_response = create_hyperliquid_order(symbol, decision, int(decision['leverage'].replace("x", "")),
                                                      amount)
            if order_response:
                send_telegram_message(format_telegram_message(decision, amount))
            else:
                send_telegram_message("Failed to place order.")
        else:
            send_telegram_message(f"Position Exists. Order Canceled.")


if __name__ == "__main__":
    main()  # while loop 제거
