# data_fetcher.py
import time
import logging
import asyncio
import aiohttp
import requests

import pandas as pd
import numpy as np
import undetected_chromedriver as uc

from config import KST
from bs4 import BeautifulSoup
from datetime import datetime
from selenium.webdriver.common.by import By
from indicators import calculate_technical_indicators, calculate_volume_divergence, calculate_donchian_channel


async def fetch_ohlcv(exchange, symbol, timeframe, limit=300):
    """
    비동기적으로 OHLCV 데이터를 가져옵니다.
    """
    try:
        # Hyperliquid does not support async.  Use a wrapper if needed.
        loop = asyncio.get_event_loop()
        ohlcv = await loop.run_in_executor(None, exchange.fetch_ohlcv, symbol, timeframe, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(KST)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logging.error(f"Error fetching {symbol} / {timeframe} OHLCV data: {e}")
        return None


async def fetch_order_book(exchange, symbol):
    """비동기적으로 주문서 데이터를 가져옵니다."""
    try:
        # loop = asyncio.get_event_loop()  # 이벤트 루프 가져오기  -> 제거
        # order_book = await loop.run_in_executor(None, exchange.fetch_order_book, symbol) -> 제거
        order_book = exchange.fetch_order_book(symbol)  # 직접 호출
        bid = order_book['bids'][0][0] if order_book['bids'] else None
        ask = order_book['asks'][0][0] if order_book['asks'] else None
        spread = round(ask - bid, 2) if bid and ask else None
        return {"bid": bid, "ask": ask, "spread": spread}
    except Exception as e:
        logging.error(f"Error fetching order book for {symbol}: {e}")
        return {"bid": None, "ask": None, "spread": None}


async def fetch_funding_rate(exchange, symbol):
    """비동기적으로 펀딩비 데이터를 가져옵니다."""
    try:
        # loop = asyncio.get_event_loop()  # 제거
        # funding_info = await loop.run_in_executor(None, exchange.fetch_funding_rate, symbol=symbol) # 제거
        funding_info = exchange.fetch_funding_rate(symbol)  # 직접 호출, symbol= 제거
        return funding_info.get('info', {}).get('funding', 'N/A')

    except Exception as e:
        logging.error(f"Error fetching funding rate for {symbol}: {e}")
        return "N/A"


async def fetch_open_interest(exchange, symbol):
    """비동기적으로 미결제약정 데이터를 가져옵니다."""
    try:
        # loop = asyncio.get_event_loop() # 제거
        # oi_response = await loop.run_in_executor(None, exchange.fetch_open_interest, symbol=symbol) # 제거
        oi_response = exchange.fetch_open_interest(symbol)  # 직접 호출, symbol= 제거

        return oi_response.get('openInterest', 'N/A') if oi_response else 'N/A'
    except Exception as e:
        logging.error(f"Error fetching open interest for {symbol}: {e}")
        return "N/A"


async def fetch_fear_and_greed_index():
    """
    Alternative.me API를 사용하여 Fear & Greed Index를 비동기적으로 가져옵니다.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.alternative.me/fng/?limit=1") as response:
                response.raise_for_status()
                data = await response.json()
                value = data['data'][0]['value'] if 'data' in data and len(data['data']) > 0 else None
                classification = data['data'][0]['value_classification'] if 'data' in data and len(
                    data['data']) > 0 else None
                return classification, value
    except Exception as e:
        logging.error(f"Error fetching Fear & Greed Index: {e}")
        return None, None


def fetch_economic_data():  # async 제거
    """
    Investing.com에서 경제 캘린더 데이터를 가져옵니다.
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
        return response.json()  # JSON 응답을 പ്രതീക്ഷ
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching economic data: {e}")
        return None


def parse_economic_data(json_data):
    """
    경제 캘린더 API 응답(HTML)을 파싱하여 요약 정보를 추출합니다.
    """
    if not json_data or not json_data['data']:
        return "No significant economic events today."

    html_content = json_data['data']
    soup = BeautifulSoup(html_content, 'html.parser')
    event_rows = soup.find_all('tr', class_='js-event-item')

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

            event_time_str = time_cell.text.strip()
            event_currency = currency_cell.text.strip()
            event_impact_element = impact_cell.find('i')
            event_impact_str = event_impact_element.get('data-img_key', 'low').replace('bull', '').lower()
            impact_mapping = {'3': 'high', '2': 'medium', '1': 'low', 'gray': 'low'}
            event_impact = impact_mapping.get(event_impact_str, 'low')
            event_name = event_cell.text.strip()

            event_hour, event_minute = map(int, event_time_str.split(':'))
            event_datetime_kst = now_kst.replace(hour=event_hour, minute=event_minute, second=0, microsecond=0)
            time_diff_minutes = (event_datetime_kst - now_kst).total_seconds() / 60

            if abs(time_diff_minutes) <= 60 * 24:
                event_list.append({
                    'time': event_time_str,
                    'currency': event_currency,
                    'impact': event_impact,
                    'name': event_name,
                    'time_diff': time_diff_minutes
                })

        except (KeyError, ValueError, AttributeError) as e:
            logging.warning(f"Error parsing economic event row: {e}, row data: {row}")
            continue

    if not event_list:
        return "No significant economic events today."

    event_list.sort(key=lambda x: (-{'high': 3, 'medium': 2, 'low': 1}.get(x['impact'], 0), x['time_diff']))
    summary_lines = ["**Upcoming Economic Events (KST):**"]
    for event in event_list:
        time_display = f"in {int(event['time_diff'])} min" if event[
                                                                  'time_diff'] > 0 else f"{abs(int(event['time_diff']))} min ago"
        summary_lines.append(
            f"- `{event['time']}` ({event['currency']}, {event['impact']}): {event['name']} ({time_display})")

    return "\n".join(summary_lines)


def fetch_liquidation_map():
    """
    CoinAnk 사이트에서 청산 히트맵 데이터를 다운로드합니다.
    """
    url = "https://coinank.com/liqMapChart"
    try:
        driver = get_driver()  # 이전에 정의된 get_driver 함수 사용
        driver.get(url)
        time.sleep(5)  # Increased wait time

        if driver.find_element(By.XPATH, '//span[@class="anticon anticon-camera"]').is_displayed():
            driver.find_element(By.XPATH, '//span[@class="anticon anticon-camera"]').click()
            time.sleep(3)
            driver.quit()

        logging.info("Liquidation heatmap data fetched successfully")
    except Exception as e:
        logging.error(f"Error fetching liquidation heatmap data: {e}")


# asyncio를 사용하여 동시에 여러 데이터를 가져오는 함수
async def gather_additional_data(exchange, symbol):
    """
    필요한 모든 추가 데이터를 비동기적으로 가져옵니다.
    """
    order_book, funding_rate, open_interest, fear_and_greed = await asyncio.gather(
        fetch_order_book(exchange, symbol),
        fetch_funding_rate(exchange, symbol),
        fetch_open_interest(exchange, symbol),
        fetch_fear_and_greed_index()
    )
    return {
        "order_book": order_book,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "fear_and_greed_index": fear_and_greed
    }


def get_driver():
    """
    undetected_chromedriver의 새로운 인스턴스를 생성하여 반환한다.
    """
    options = uc.ChromeOptions()
    # Configure options if needed (e.g., headless mode)
    # options.add_argument('--headless') # 필요에 따라 headless 모드 활성화
    driver = uc.Chrome(options=options)
    return driver


async def fetch_multi_tf_data(exchange, symbol, timeframes=None, limit=300):
    """
    여러 타임프레임(예: 5m, 15m, 1h, 4h, 1d)의 OHLCV 데이터를 가져오고
    기술적 지표를 계산하여 요약 정보를 반환한다.
    """
    if timeframes is None:
        timeframes = ["5m", "15m", "1h", "4h", "1d"]
    multi_tf_data = {}
    for tf in timeframes:
        df = await fetch_ohlcv(exchange, symbol, tf, limit)  # await 추가 및 exchange 전달
        if df is None:
            continue
        df = calculate_technical_indicators(df)
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
