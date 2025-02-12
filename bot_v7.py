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
    입력받은 DataFrame에 RSI, SMA, MACD, ATR, OBV, MFI, Bollinger Bands 등
    다양한 기술적 지표를 계산하여 추가한다.
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
# 7. 시장 상태 결정 및 지표 임계값 조정
# =====================================================
def is_high_volatility(multi_tf_data):
    """
    1시간봉의 ATR/현재가 비율을 기준으로 고변동성 여부를 판단한다.
    """
    data = multi_tf_data.get("1h")
    if data and data.get("atr") is not None and data.get("current_price") is not None:
        atr_ratio = data["atr"] / data["current_price"]
        return atr_ratio > 0.02
    return False


def determine_market_regime(multi_tf_data, onchain_data):
    """
    1시간 데이터의 SMA50, SMA200 및 온체인 데이터를 활용하여
    시장의 상태(강세, 약세, 횡보)를 결정한다.
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
    시장 상태에 따라 RSI, MACD, MA 등의 임계값을 동적으로 조정한다.
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
    SMA200과의 편차가 가장 큰 타임프레임을 기본 타임프레임으로 선택한다.
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


# =====================================================
# 8. GPT 프롬프트 생성 및 거래 결정
# =====================================================
def generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data,
                        onchain_data, multi_tf_data, market_regime, thresholds,
                        heatmap_analysis, econ_summary):
    """
    계좌 정보, 다중 타임프레임 데이터, 기술/온체인/심리 데이터 등을 포함한
    XML 기반의 상세 프롬프트를 생성한다.
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

    prompt = f"""<TradeBotPrompt>
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
            <EconomicCalendar>
                {econ_summary}
            </EconomicCalendar>
        </AdditionalData>
    </MarketContext>
    <Indicators>
        <RSI>
            <Guide>
                Oversold ~ {thresholds['rsi_oversold']} / Overbought ~ {thresholds['rsi_overbought']}
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
                Open a new position only if the potential reward is at least 2x the risk, carefully considering liquidation heatmap analysis.
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
</TradeBotPrompt>"""
    return prompt


def generate_trading_decision(wallet_balance, position_info, aggregated_data, extended_data,
                              onchain_data, multi_tf_data, market_regime, thresholds,
                              heatmap_analysis, econ_summary):
    """
    GPT를 통해 위의 프롬프트를 전달하고, 거래 결정을 받아온다.
    """
    prompt = generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data,
                                 onchain_data, multi_tf_data, market_regime, thresholds,
                                 heatmap_analysis, econ_summary)

    developer_prompt = (
        "You are an automated BTC/USDT trading assistant that follows detailed quantitative indicators and strict guidelines "
        "as defined in the provided XML. Do not deviate from the specified output format. "
        "Based on market regime and all provided data including liquidation heatmap analysis, adopt either a trend-following or mean reversion strategy accordingly."
    )
    response = gpt_client.chat.completions.create(
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
    GPT의 응답(콤마 구분 형식)을 파싱하여 거래 결정의 각 필드를 추출한다.
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
def compute_risk_reward(decision, entry_price):
    """
    결정된 거래 가격 및 목표, 손절가를 이용해 위험/보상 비율을 계산한다.
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
    주문서, 거래소 유입량, funding rate, open interest, Fear & Greed Index 등 추가 데이터를 취합한다.
    """
    base_data = fetch_order_book(symbol)
    exchange_inflows = fetch_exchange_inflows()
    funding_rate = fetch_funding_rate(symbol)
    open_interest = fetch_open_interest(symbol)
    fng = fetch_fear_and_greed_index()  # (classification, value)

    extended_data = {
        "order_book_bid": base_data.get("bid", "N/A"),
        "order_book_ask": base_data.get("ask", "N/A"),
        "order_book_spread": base_data.get("spread", "N/A"),
        "exchange_inflows": exchange_inflows,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "fear_and_greed_index": fng
    }
    return extended_data


def get_current_session_kst():
    """
    KST 기준 현재 시간을 확인하여 트레이딩 세션(예: Overnight, Asian, European/US)을 결정한다.
    """
    now = datetime.now(KST)
    hour = now.hour

    if 0 <= hour < 8:
        return "OVERNIGHT_LOW_LIQUIDITY"  # 00:00 ~ 08:00
    elif 8 <= hour < 16:
        return "ASIAN_SESSION"  # 08:00 ~ 16:00
    else:
        return "EUROPEAN_US_SESSION"  # 16:00 ~ 24:00 (런던/미국 겹치는 구간)


def is_news_volatility_period(minutes_before=10, minutes_after=10):
    """
    주요 경제 지표 발표 전후 기간인지 확인하여 변동성 상승 여부를 판단한다.
    """
    major_news_times = [datetime.now(KST).replace(hour=22, minute=30, second=0, microsecond=0)]
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


def parse_economic_data(data):
    """
    가져온 경제 데이터 HTML을 파싱하여 이벤트 정보를 추출한다.
    """
    if data is None:
        return []

    try:
        html_data = data['data']
        soup = BeautifulSoup(html_data, 'html.parser')
        rows = soup.find_all('tr', class_='js-event-item')

        # 반환값: [(time_str, currency, impact_level, event_name, time_diff_minutes), ...]
        # impact_level은 '🔴🔴🔴' 이런 식으로 표시될 수 있음
        # time_diff_minutes: 발표 시간 - 현재 시간 (분 단위)
        result_list = []
        now_kst = datetime.now(KST)

        for row in rows:
            time_str = row.find('td', class_='time').text.strip()
            currency = row.find('td', class_='flagCur').text.strip()
            impact_icons = row.find('td', class_='sentiment').find_all('i')
            impact_level = "🔴" * len(impact_icons)
            event_name = row.find('td', class_='event').text.strip()

            # 실제 발표 시간을 KST 날짜/시간으로 해석
            # investing.com은 기본 UTC or 현지 표시. 위 data에서 timeZone=8이므로 UTC+8일 가능성 있으니 확인 필요
            # 여기서는 간단히 now_kst.date()에 time만 대입
            # time_str 예) "22:30"
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
    """
    econ_list: [(time_str, currency, impact_level, event_name, time_diff_minutes), ...]
    발표 시점이 아직 안 지났으면, 남은 시간을 분으로 표시
    이미 지났으면 '지표 발표 후 x분' 형태 등으로 표시 가능
    """
    if not econ_list:
        return "No major economic events for the day."

    lines = []
    now_kst = datetime.now(KST)
    for item in econ_list:
        time_str, currency, impact, event_name, diff_minutes = item
        if diff_minutes > 0:
            lines.append(f"{time_str} ({currency}, {impact}): {event_name}, about {int(diff_minutes)} min left")
        else:
            lines.append(f"{time_str} ({currency}, {impact}): {event_name}, {abs(int(diff_minutes))} min since release")

    return "\n".join(lines)


def detect_fake_breakout(df, lookback=20, volume_factor=1.5, rsi_threshold=(30, 70), pivot_dist=0.001):
    """
    df: 봉데이터 (이미 기술적 지표 포함)
    lookback: 몇 개 봉을 기준으로 최근 피벗을 잡을지
    volume_factor: 평균 거래량 대비 얼마나 적으면 가짜 돌파로 간주할지
    rsi_threshold: (oversold, overbought) 범위
    pivot_dist: 피벗 돌파 여부를 판단할 때 허용 오차
    -------------
    로직:
    1) 최근 lookback 봉의 최고가, 최저가 구하여 pivot range 생성
    2) 마지막 봉 close가 pivot high 이상(또는 pivot low 이하)인지 → 돌파 시도 판단
    3) 돌파 시점에서:
       - 거래량(마지막봉)이 이전 N봉 평균 대비 volume_factor 미만이라면 가짜 돌파 가능성 ↑
       - RSI가 극단값에 있지 않은데도 갑자기 고가 돌파? 모멘텀 부족 시 가짜 돌파 가능성 ↑
       - 돌파 직후 되돌림(윗꼬리 or 아랫꼬리) 폭이 큰지
    4) 종합적으로 score를 매겨서 일정 이상이면 is_fake_breakout = True
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
    volume_check = (last_vol < prev_20_vol_mean * volume_factor)  # 거래량이 충분하지 않으면 의심
    rsi_check = ((breakout_up and last_rsi < rsi_threshold[1] - 5) or  # 위로 돌파인데 RSI가 너무 낮으면 힘 부족
                 (breakout_down and last_rsi > rsi_threshold[0] + 5))  # 아래로 돌파인데 RSI가 너무 높으면 가짜
    # 되돌림 체크: 종가와 고가/저가 차이 비율
    candle_range = last_candle['high'] - last_candle['low']
    if candle_range == 0:
        retrace_ratio = 0
    else:
        if breakout_up:
            retrace_ratio = (last_candle['high'] - last_candle['close']) / candle_range
        else:
            retrace_ratio = (last_candle['close'] - last_candle['low']) / candle_range
    retrace_check = (retrace_ratio > 0.6)  # 돌파했는데 60% 이상 되돌림 => 윗꼬리/아랫꼬리 심함

    # 가중치 합산
    suspicion_score = 0
    if volume_check:
        suspicion_score += 1
    if rsi_check:
        suspicion_score += 1
    if retrace_check:
        suspicion_score += 1

    # 예: 2점 이상이면 가짜 돌파로 판단
    if suspicion_score >= 2:
        return True
    return False


def analyze_session_open_volatility(df, open_hour=8, open_minute=0, threshold_factor=1.5, window=5):
    """
    df: 전체 봉 데이터(이미 'atr' 계산됨)
    open_hour, open_minute: 세션 오픈 시각(KST) 예) 8:00, 16:00
    threshold_factor: 최근 ATR 비교 시, threshold_factor배 이상의 변동이면 높은 변동성으로 간주
    window: 세션 오픈 이후 몇 개 봉을 관찰할지 (분봉 기준)
    로직:
      1) 세션 오픈 시간~오픈+window분 구간의 가격 변동 폭, ATR 비교
      2) 예) open_dt ~ open_dt + window 분 사이 최대-최소 vs 최근 ATR
    """
    now_kst = datetime.now(KST)
    open_dt = now_kst.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)

    # 세션 오픈 전후 30분 정도만 감지
    diff_minutes = (now_kst - open_dt).total_seconds() / 60.0
    if abs(diff_minutes) > 30:
        return False  # 세션 오픈 30분 이후면 굳이 체크 안함

    # window 분 내에 존재하는 봉 추출
    recent_time = open_dt + timedelta(minutes=window)
    # df 는 timestamp 기준 정렬되어있다고 가정
    subdf = df[(df['timestamp'] >= open_dt) & (df['timestamp'] <= recent_time)]
    if len(subdf) < 1:
        return False

    price_range = subdf['high'].max() - subdf['low'].min()
    # 최근 ATR 평균 (직전 ~20봉)
    # 만약 5분봉이라면, ~100분 정도
    last_atr = df['atr'].iloc[-20:].mean() if len(df) >= 20 else df['atr'].mean()
    # 실제 변동성 체크
    if last_atr and last_atr > 0:
        if price_range / last_atr > threshold_factor:
            return True
    return False


def main():
    logging.info("Trading bot started.")
    wallet_balance = "1000 USDT"
    position_info = "NONE"
    in_position = False
    current_side = None
    entry_price = 0.0

    # 경제 지표 파싱
    econ_data_raw = fetch_economic_data()
    econ_list = parse_economic_data(econ_data_raw)
    econ_summary = get_economic_event_summary(econ_list)

    # 멀티 TF
    mtf = fetch_multi_tf_data(SYMBOL, ["5m", "15m", "1h", "4h", "1d"], limit=300)
    if not mtf or "1h" not in mtf:
        logging.error("Not enough TF data.")
        return

    cprice = mtf["1h"]["current_price"]
    ext_data = fetch_additional_data(SYMBOL)
    onchain_data = fetch_onchain_data(SYMBOL)

    # 세션 파악
    session_name = get_current_session_kst()
    logging.info(f"Current session: {session_name}")

    # 가짜 돌파 감지(정밀)
    # 예: 5분봉 기준
    if "5m" in mtf:
        df_5m = mtf["5m"]["df_full"]
        if detect_fake_breakout(df_5m):
            logging.info("Fake breakout suspicion on 5m. Strategy may skip or filter signals.")

    # 세션 오픈 시 변동성
    # 예: 아시아(8시), 런던(16시)
    # 5m 분봉 데이터 활용
    if "5m" in mtf:
        df_5m = mtf["5m"]["df_full"]
        # 아시아
        if analyze_session_open_volatility(df_5m, 8, 0, threshold_factor=1.5, window=6):
            logging.info("High volatility at Asia open. Potential breakout strategy.")
        # 런던
        if analyze_session_open_volatility(df_5m, 16, 0, threshold_factor=1.5, window=6):
            logging.info("High volatility at London open. Potential breakout strategy.")

    # TF별 bull/bear
    bullish_cnt, bearish_cnt = 0, 0
    directions = {}
    for tf, data in mtf.items():
        sm2 = data["sma200"]
        px = data["current_price"]
        if sm2 and px > sm2:
            bullish_cnt += 1
            directions[tf] = "bullish"
        else:
            bearish_cnt += 1
            directions[tf] = "bearish"
    aggregated_trend = "BULL" if bullish_cnt >= bearish_cnt else "BEAR"
    aggregated_data = {"trend": aggregated_trend, "timeframe_directions": directions}

    primary_tf = choose_primary_timeframe(mtf)
    primary_dir = directions[primary_tf] if primary_tf else None

    # 사전 필터
    if not (bullish_cnt >= 4 or bearish_cnt >= 4):
        if not (
                (primary_dir == "bullish" and bullish_cnt >= 3) or
                (primary_dir == "bearish" and bearish_cnt >= 3)
        ):
            logging.info("No strong consensus. No GPT decision.")
            driver = get_driver()
            driver.quit()
            return

    # Market regime
    regime = determine_market_regime(mtf, onchain_data)
    thresholds = adjust_indicator_thresholds(regime)

    # Liquidation heatmap
    fetch_liquidation_heatmap()
    heatmap_analysis = analyze_liquidation_heatmap()

    # GPT decision
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
            econ_summary=econ_summary
        )
        logging.info(f"GPT raw: {gpt_resp}")
        decision = parse_trading_decision(gpt_resp)
        logging.info(f"Parsed: {decision}")
        log_decision(decision, SYMBOL)
    except Exception as e:
        logging.error(f"Error GPT: {e}")
        return

    rr = compute_risk_reward(decision, cprice)
    rr_text = f"{rr:.2f}" if rr else "N/A"

    if decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
        side = "BUY" if decision["final_action"].upper() == "GO LONG" else "SELL"
        msg = (
            f"{side} SIGNAL\n"
            f"Symbol: {SYMBOL}\n"
            f"Leverage: {decision['leverage']}\n"
            f"R/R: {rr_text}\n"
            f"Term: {decision['trade_term']}\n"
            f"Limit: {decision['limit_order_price']}\n"
            f"TP: {decision['tp_price']} / SL: {decision['sl_price']}\n"
            f"Rationale: {decision['rationale']}"
        )
        send_telegram_message(msg)

    if not in_position:
        if rr and rr >= 2 and decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
            logging.info(f"Open position: {decision['final_action']} @ {cprice}")
            log_open_position(SYMBOL, decision, cprice)
            in_position = True
            current_side = decision["final_action"].split()[-1]
            entry_price = cprice
        else:
            logging.info("No new position (R/R < 2 or not GO LONG/SHORT).")
    else:
        if decision["final_action"].upper() not in ["HOLD LONG", "HOLD SHORT"]:
            logging.info(f"Exiting position @ {cprice}")
            log_closed_position(SYMBOL, entry_price, cprice, current_side)
            in_position = False
        else:
            logging.info("Maintain current position.")


if __name__ == "__main__":
    main()
