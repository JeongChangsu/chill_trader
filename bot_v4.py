#!/usr/bin/env python3
import os
import re
import csv
import time
import ccxt
import requests
import numpy as np
import pandas as pd
import ta
import undetected_chromedriver as uc  # 무료 데이터 크롤링용 stealth driver
import logging
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
# OpenAI API 모듈 (예시)
from openai import OpenAI

# ===========================================
# 로그 설정 (한국어 로그)
# ===========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ===========================================
# 글로벌 상수 및 API 초기화
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

# API 키 및 환경변수 설정
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# 일일 손실 한도 (예: 계좌의 5% 이상 손실 발생 시 거래 중단)
DAILY_LOSS_LIMIT_PERCENT = 5.0
daily_loss = 0.0  # 전역 변수로 매일 초기화 필요 (여기서는 간단히 사용)

# GPT 캐싱 (최근 10분 동일 조건에 대해 캐시)
gpt_cache = {}  # key: f"{market_regime}_{primary_tf}", value: (timestamp, decision)


# ===========================================
# 텔레그램 메시지 전송 함수
# ===========================================
def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        logging.error("텔레그램 봇 토큰 또는 채팅 ID가 설정되지 않았습니다.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            logging.info("텔레그램 메시지 전송 성공")
        else:
            logging.error(f"텔레그램 메시지 전송 실패 (상태 코드: {response.status_code})")
    except Exception as e:
        logging.error(f"텔레그램 메시지 전송 중 오류 발생: {e}")


# ===========================================
# Persistent Driver Setup for Data Crawling
# ===========================================
drivers = {}


def create_driver():
    """새로운 undetected_chromedriver 인스턴스를 생성합니다."""
    options = uc.ChromeOptions()
    # 필요 시 headless 모드 등 옵션 설정 가능
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
# 1. 데이터 수집 및 기술적 지표 계산
# ===========================================
def fetch_ohlcv(symbol, timeframe, limit=300):
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        logging.info(f"{symbol} / {timeframe} OHLCV 데이터 수집 성공")
    except Exception as e:
        logging.error(f"{symbol} / {timeframe} OHLCV 데이터를 가져오는 중 오류 발생: {e}")
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def compute_technical_indicators(df):
    """
    다양한 기술적 지표를 계산합니다.
    추가: ADX, Bollinger Bands 내 가격 위치, OBV slope 계산
    """
    # 모멘텀 및 추세 지표
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['sma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # 변동성 및 거래량 지표
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()

    # Bollinger Bands (20-period, 2 std)
    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()

    # Bollinger Band Position 계산 (문자열 값)
    def bb_position(row):
        if row['close'] > row['bb_upper']:
            return "above"
        elif row['close'] > row['bb_middle']:
            return "upper_mid"
        elif row['close'] >= row['bb_lower']:
            return "lower_mid"
        else:
            return "below"

    df['bb_position'] = df.apply(bb_position, axis=1)

    # ADX (추세 강도)
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    df['adx_neg'] = adx_indicator.adx_neg()
    df['adx_pos'] = adx_indicator.adx_pos()

    # OBV slope 계산: 최근 14기간 OBV 변화량 (단순 차분)
    df['obv_slope'] = df['obv'].diff(14)

    # 이동평균 백분율 차이
    df['ma50_diff'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['ma200_diff'] = (df['close'] - df['sma200']) / df['sma200'] * 100

    logging.info("기술적 지표 계산 완료")
    return df


def fetch_order_book(symbol):
    exchange = ccxt.binance()
    try:
        order_book = exchange.fetch_order_book(symbol)
        bid = order_book['bids'][0][0] if order_book['bids'] else None
        ask = order_book['asks'][0][0] if order_book['asks'] else None
        spread = round(ask - bid, 2) if bid and ask else None
        logging.info(f"{symbol} 주문서 데이터 수집 완료")
    except Exception as e:
        logging.error(f"{symbol} 주문서 데이터를 가져오는 중 오류 발생: {e}")
        bid = ask = spread = None
    return {"bid": bid, "ask": ask, "spread": spread}


def fetch_exchange_inflows():
    """
    무료 소스에서 거래소 유입/유출 데이터를 크롤링 (예시)
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
            logging.info("거래소 유입/유출 데이터 크롤링 성공")
            return net_inflow
    except Exception as e:
        logging.error(f"거래소 순입출금량 크롤링 중 오류 발생: {e}")
        return "N/A"


def fetch_funding_rate(symbol):
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        funding_info = futures_exchange.fetch_funding_rate(symbol=symbol)
        latest_funding = funding_info['info']['lastFundingRate'] if 'info' in funding_info else None
        logging.info(f"{symbol} 펀딩율 데이터 수집 성공")
        return latest_funding
    except Exception as e:
        logging.error(f"{symbol} 펀딩율 데이터를 가져오는 중 오류 발생: {e}")
        return "N/A"


def fetch_open_interest(symbol):
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        symbol_futures = symbol.replace("/", "")
        oi_response = futures_exchange.fetch_open_interest(symbol=symbol_futures)
        open_interest = oi_response['openInterest'] if oi_response and 'openInterest' in oi_response else None
        logging.info(f"{symbol} 미결제약정 데이터 수집 성공")
        return open_interest
    except Exception as e:
        logging.error(f"{symbol} 미결제약정 데이터를 가져오는 중 오류 발생: {e}")
        return "N/A"


def fetch_fear_and_greed_index():
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        data = response.json()
        value = data['data'][0]['value'] if 'data' in data and len(data['data']) > 0 else None
        classification = data['data'][0]['value_classification'] if 'data' in data and len(data['data']) > 0 else None
        logging.info("Fear & Greed Index 데이터 수집 성공")
        return classification, value
    except Exception as e:
        logging.error(f"Fear & Greed Index 데이터를 가져오는 중 오류 발생: {e}")
        return None, None


def fetch_onchain_data(symbol):
    """
    온체인 지표(MVRV, SOPR) 예시 크롤링 (실제 서비스시 API 활용)
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
        logging.info("온체인 데이터 (MVRV, SOPR) 수집 성공")
        return {"mvrv": mvrv, "sopr": sopr}
    except Exception as e:
        logging.error(f"온체인 데이터를 가져오는 중 오류 발생: {e}")
        return {"mvrv": "N/A", "sopr": "N/A"}


def fetch_multi_tf_data(symbol, timeframes=None, limit=300):
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
            "bb_middle": round(latest['bb_middle'], 2) if not np.isnan(latest['bb_middle']) else None,
            "bb_lower": round(latest['bb_lower'], 2) if not np.isnan(latest['bb_lower']) else None,
            "bb_position": latest['bb_position'],
            "macd": round(latest['macd'], 2) if not np.isnan(latest['macd']) else None,
            "macd_signal": round(latest['macd_signal'], 2) if not np.isnan(latest['macd_signal']) else None,
            "atr": round(latest['atr'], 2) if not np.isnan(latest['atr']) else None,
            "obv": round(latest['obv'], 2) if not np.isnan(latest['obv']) else None,
            "obv_slope": round(latest['obv_slope'], 2) if not np.isnan(latest['obv_slope']) else None,
            "mfi": round(latest['mfi'], 2) if not np.isnan(latest['mfi']) else None,
            "adx": round(latest['adx'], 2) if not np.isnan(latest['adx']) else None,
            "timestamp": latest['timestamp']
        }
    logging.info("다중 타임프레임 데이터 및 지표 계산 완료")
    return multi_tf_data


# ===========================================
# 2. 시장 장세 판단 및 전략 파라미터 조정
# ===========================================
def classify_market_regime(multi_tf_data, onchain_data):
    """
    1시간봉 데이터를 기준으로 여러 기술적 지표(가격/SMA, ADX, OBV slope 등)를 활용해 장세를 세분화합니다.
    – 강한 상승: strong_bull
    – 약한 상승: weak_bull
    – 중립: neutral
    – 약한 하락: weak_bear
    – 강한 하락: strong_bear
    추가로 ATR/price 비율이 높으면 high_vol, 낮으면 low_vol 접미사 추가.
    """
    data = multi_tf_data.get("1h")
    if data is None:
        logging.warning("1시간봉 데이터 없음 - neutral 처리")
        return "neutral"
    current_price = data["current_price"]
    sma50 = data["sma50"]
    sma200 = data["sma200"]
    adx = data.get("adx", 20)
    obv_slope = data.get("obv_slope", 0)
    # 가격 대비 SMA 차이 (비율)
    diff50 = (current_price - sma50) / sma50 if sma50 else 0
    diff200 = (current_price - sma200) / sma200 if sma200 else 0

    # 기본 장세 결정 (단순 규칙)
    if diff50 > 0.02 and diff200 > 0.02 and adx > 25 and obv_slope > 0:
        regime = "strong_bull"
    elif diff50 > 0.005 and diff200 > 0.005 and obv_slope > 0:
        regime = "weak_bull"
    elif diff50 < -0.02 and diff200 < -0.02 and adx > 25 and obv_slope < 0:
        regime = "strong_bear"
    elif diff50 < -0.005 and diff200 < -0.005 and obv_slope < 0:
        regime = "weak_bear"
    else:
        regime = "neutral"

    # 온체인 데이터 보정: 만약 MVRV와 SOPR가 모두 1 미만이면 중립으로 보정
    if onchain_data["mvrv"] != "N/A" and onchain_data["sopr"] != "N/A":
        if onchain_data["mvrv"] < 1 and onchain_data["sopr"] < 1:
            logging.info("온체인 지표 극단적 -> 장세 neutral 보정")
            regime = "neutral"

    # 변동성 판단: ATR/현재가격 비율로 high_vol/low_vol 결정
    atr = data.get("atr", 0)
    vol_ratio = atr / current_price if current_price else 0
    vol_tag = "high_vol" if vol_ratio > 0.02 else "low_vol"
    final_regime = f"{regime}_{vol_tag}"
    logging.info(
        f"시장 장세: {final_regime} (diff50: {diff50:.2%}, adx: {adx}, obv_slope: {obv_slope}, vol_ratio: {vol_ratio:.2%})")
    return final_regime


def adjust_trade_parameters(market_regime, primary_tf):
    """
    시장 장세와 주 타임프레임에 따라 TP/SL, 거래 기간, 추천 레버리지 등을 조정합니다.
    """
    # 기본 설정 (퍼센트 단위)
    trade_params = {
        "strong_bull_low_vol": {"tp_pct": 6, "sl_pct": 3.5, "trade_term": "1d", "leverage": "5x"},
        "strong_bull_high_vol": {"tp_pct": 5, "sl_pct": 4, "trade_term": "1d", "leverage": "4x"},
        "weak_bull_low_vol": {"tp_pct": 3, "sl_pct": 2, "trade_term": "6h", "leverage": "3x"},
        "weak_bull_high_vol": {"tp_pct": 2.5, "sl_pct": 2.5, "trade_term": "6h", "leverage": "3x"},
        "neutral_low_vol": {"tp_pct": 1, "sl_pct": 0.5, "trade_term": "intraday", "leverage": "2x"},
        "neutral_high_vol": {"tp_pct": 0.8, "sl_pct": 0.8, "trade_term": "intraday", "leverage": "2x"},
        "weak_bear_low_vol": {"tp_pct": 3, "sl_pct": 2, "trade_term": "6h", "leverage": "3x"},
        "weak_bear_high_vol": {"tp_pct": 2.5, "sl_pct": 2.5, "trade_term": "6h", "leverage": "3x"},
        "strong_bear_low_vol": {"tp_pct": 6, "sl_pct": 3.5, "trade_term": "1d", "leverage": "5x"},
        "strong_bear_high_vol": {"tp_pct": 5, "sl_pct": 4, "trade_term": "1d", "leverage": "4x"}
    }
    # market_regime key가 e.g., "strong_bull_low_vol" 형태로 나옴
    params = trade_params.get(market_regime, {"tp_pct": 1, "sl_pct": 0.5, "trade_term": "intraday", "leverage": "2x"})
    logging.info(f"장세({market_regime})에 따른 거래 파라미터 설정: {params}")
    return params


def choose_primary_timeframe(multi_tf_data):
    """
    SMA200 대비 가격 이격률이 가장 큰 타임프레임을 주 타임프레임으로 선택합니다.
    """
    primary_tf = None
    max_diff = 0
    for tf, data in multi_tf_data.items():
        if data["sma200"]:
            diff = abs(data["current_price"] - data["sma200"]) / data["sma200"]
            if diff > max_diff:
                max_diff = diff
                primary_tf = tf
    logging.info(f"주 타임프레임 선택: {primary_tf} (최대 이격률: {max_diff:.2%})")
    return primary_tf


# ===========================================
# 3. 매매 결정: 알고리즘 & GPT 활용
# ===========================================
def is_signal_clear(multi_tf_data):
    """
    여러 타임프레임에서 동일 방향 신호(4/5 이상)가 확인되면 명확한 신호로 간주합니다.
    """
    bullish_count = 0
    bearish_count = 0
    for tf, data in multi_tf_data.items():
        if data["sma200"] and data["current_price"] > data["sma200"]:
            bullish_count += 1
        else:
            bearish_count += 1
    if bullish_count >= 4 or bearish_count >= 4:
        return True, "BULL" if bullish_count > bearish_count else "BEAR"
    return False, None


def determine_trading_action_by_rules(aggregated_trend, trade_params, current_price):
    """
    명확한 신호가 있을 경우 알고리즘으로 최종 매매 결정을 내립니다.
    """
    decision = {}
    if aggregated_trend == "BULL":
        decision["final_action"] = "GO LONG"
    elif aggregated_trend == "BEAR":
        decision["final_action"] = "GO SHORT"
    else:
        decision["final_action"] = "NO TRADE"

    decision["leverage"] = trade_params["leverage"]
    decision["trade_term"] = trade_params["trade_term"]
    # TP/SL 가격 계산 (단순 % 적용)
    if decision["final_action"] == "GO LONG":
        decision["tp_price"] = round(current_price * (1 + trade_params["tp_pct"] / 100), 2)
        decision["sl_price"] = round(current_price * (1 - trade_params["sl_pct"] / 100), 2)
        decision["limit_order_price"] = round(current_price * 0.995, 2)  # 진입가보다 약간 낮게
    elif decision["final_action"] == "GO SHORT":
        decision["tp_price"] = round(current_price * (1 - trade_params["tp_pct"] / 100), 2)
        decision["sl_price"] = round(current_price * (1 + trade_params["sl_pct"] / 100), 2)
        decision["limit_order_price"] = round(current_price * 1.005, 2)  # 진입가보다 약간 높게
    else:
        decision["tp_price"] = decision["sl_price"] = decision["limit_order_price"] = "N/A"
    decision["rationale"] = "알고리즘 신호에 의해 결정됨 (명확한 다중 타임프레임 합의)."
    return decision


def generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data, onchain_data,
                        multi_tf_data, market_regime, trade_params):
    """
    GPT에게 전달할 XML 형태의 프롬프트 생성 (추가 거래 파라미터 포함)
    """
    # 다중 타임프레임 요약
    multi_tf_summary = ""
    for tf, data in multi_tf_data.items():
        multi_tf_summary += (
            f"{tf} - Price: {data['current_price']}, RSI: {data['rsi']}, "
            f"SMA50: {data['sma50']} (Diff: {data['ma50_diff']}%), "
            f"SMA200: {data['sma200']} (Diff: {data['ma200_diff']}%), "
            f"Bollinger Position: {data['bb_position']}, "
            f"ADX: {data['adx']}, OBV Slope: {data['obv_slope']}\n"
        )
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
        </AdditionalData>
    </MarketContext>
    <Indicators>
        <RSI>
            <Guide>
                In bullish conditions, consider RSI oversold around adjusted thresholds. Analyze divergence and failures.
            </Guide>
        </RSI>
        <MovingAverages>
            <SMA50>
                <Guide>Monitor price relative to SMA50 for support/resistance.</Guide>
            </SMA50>
            <SMA200>
                <Guide>A significant premium/discount versus SMA200 informs bias.</Guide>
            </SMA200>
        </MovingAverages>
        <BollingerBands>
            <Guide>
                Use Bollinger Bands to gauge volatility and determine if price is above/below typical ranges.
            </Guide>
        </BollingerBands>
        <ADX>
            <Guide>
                ADX indicates trend strength; values above 25 suggest a strong trend.
            </Guide>
        </ADX>
        <OBV>
            <Guide>
                OBV slope reflects volume trend supporting price movement.
            </Guide>
        </OBV>
    </Indicators>
    <TradeParameters>
        <TakeProfitPercentage>{trade_params['tp_pct']}%</TakeProfitPercentage>
        <StopLossPercentage>{trade_params['sl_pct']}%</StopLossPercentage>
        <TradeTerm>{trade_params['trade_term']}</TradeTerm>
        <RecommendedLeverage>{trade_params['leverage']}</RecommendedLeverage>
    </TradeParameters>
    <DecisionRules>
        <MultiTimeframe>
            <Guide>
                Trade if at least 4/5 timeframes agree or if the primary timeframe shows a strong signal.
            </Guide>
        </MultiTimeframe>
        <RiskReward>
            <Guide>
                Only enter if potential reward is at least 2x the risk.
            </Guide>
        </RiskReward>
        <StrategySwitch>
            <Guide>
                Adjust strategy (trend-following or mean reversion) based on market regime.
            </Guide>
        </StrategySwitch>
    </DecisionRules>
    <Task>
        Based on the provided account, market context, indicators, and trade parameters, decide the action: GO LONG, GO SHORT, HOLD LONG, HOLD SHORT, or NO TRADE.
        Then output a comma-separated line: Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Brief Rationale.
    </Task>
    <OutputExample>
        GO LONG, 5x, 6h, 11400, 10800, 11300, Majority of indicators show strong bullish momentum.
    </OutputExample>
</TradeBotPrompt>
    """
    return prompt


def generate_trading_decision(wallet_balance, position_info, aggregated_data, extended_data, onchain_data,
                              multi_tf_data, market_regime, trade_params):
    # GPT 캐시 확인: 캐시된 결정이 10분 이내이면 재사용
    primary_tf = aggregated_data.get("primary_tf", "1h")
    cache_key = f"{market_regime}_{primary_tf}"
    now = datetime.utcnow()
    if cache_key in gpt_cache:
        cached_time, cached_decision = gpt_cache[cache_key]
        if now - cached_time < timedelta(minutes=10):
            logging.info("GPT 캐시 결과 재사용")
            return cached_decision
    # GPT 호출 조건: 신호가 애매한 경우에만 호출
    developer_prompt = (
        "You are an automated BTC/USDT trading assistant. Follow the provided XML guidelines strictly and output a single comma-separated line as instructed."
    )
    prompt = generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data, onchain_data,
                                 multi_tf_data, market_regime, trade_params)
    response = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    decision_text = response.choices[0].message.content
    # 캐시에 저장
    gpt_cache[cache_key] = (now, decision_text)
    return decision_text


def parse_trading_decision(response_text):
    """
    GPT의 응답을 파싱합니다.
    Expected output: Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Brief Rationale
    """
    response_text = response_text.strip()
    parts = [part.strip() for part in re.split(r'\s*,\s*', response_text)]
    if len(parts) < 7:
        raise ValueError("응답이 불완전합니다. 최소 7개의 콤마로 구분된 필드가 필요합니다.")
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
# 4. 포지션 로깅 및 계산 함수
# ===========================================
def compute_risk_reward(decision, entry_price):
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
        logging.info(f"위험/보상 비율: {rr_ratio:.2f}")
        return rr_ratio
    except Exception as e:
        logging.error(f"위험/보상 비율 계산 중 오류: {e}")
        return None


def compute_position_size(account_balance, risk_percent, entry_price, sl_price):
    """
    포지션 사이징 계산: 위험 금액 = 계좌잔고 * risk_percent, 단위 위험 = |entry_price - sl_price|
    """
    try:
        risk_amount = account_balance * (risk_percent / 100)
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit == 0:
            return 0
        quantity = risk_amount / risk_per_unit
        return quantity
    except Exception as e:
        logging.error(f"포지션 사이징 계산 중 오류: {e}")
        return 0


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
    logging.info("매매 결정 로그 기록됨.")


def log_open_position(symbol, decision, entry_price, position_size):
    file_exists = os.path.isfile(OPEN_POSITIONS_FILE)
    with open(OPEN_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["timestamp", "symbol", "action", "entry_price", "position_size", "leverage", "trade_term", "tp_price",
                 "sl_price",
                 "limit_order_price", "rationale"])
        writer.writerow([datetime.utcnow(), symbol, decision["final_action"], entry_price, position_size,
                         decision["leverage"], decision["trade_term"], decision["tp_price"], decision["sl_price"],
                         decision["limit_order_price"], decision["rationale"]])
    logging.info(f"{symbol} 포지션 개시 로그 (진입가: {entry_price}, 사이즈: {position_size}) 기록됨.")


def log_closed_position(symbol, entry_price, exit_price, trade_side):
    profit = (exit_price - entry_price) if trade_side.upper() == "LONG" else (entry_price - exit_price)
    file_exists = os.path.isfile(CLOSED_POSITIONS_FILE)
    with open(CLOSED_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "trade_side", "entry_price", "exit_price", "profit"])
        writer.writerow([datetime.utcnow(), symbol, trade_side, entry_price, exit_price, profit])
    logging.info(f"{symbol} 포지션 종료 (수익: {profit}) 로그 기록됨.")


# ===========================================
# 5. 메인 트레이딩 로직 및 자금관리
# ===========================================
def main():
    global daily_loss
    logging.info("트레이딩 봇 실행 시작")

    # 예시 계좌 잔고 (실제 운용시 실시간 조회)
    wallet_balance_str = "1000 USDT"
    wallet_balance = 1000.0  # USDT 기준
    position_info = "NONE"
    in_position = False
    current_position_side = None
    current_position_entry_price = 0.0
    current_position_size = 0.0

    # 데이터 수집: 다중 타임프레임
    multi_tf_data = fetch_multi_tf_data(SYMBOL, timeframes=["5m", "15m", "1h", "4h", "1d"], limit=300)
    if not multi_tf_data or "1h" not in multi_tf_data:
        logging.error("필요한 데이터 수집 실패")
        return

    current_price = multi_tf_data["1h"]["current_price"]
    logging.info(f"현재 1시간봉 가격: {current_price}")

    # 확장 데이터 (주문서, 펀딩율, 미결제약정, 거래소 유입, Fear & Greed, 온체인)
    base_data = fetch_order_book(SYMBOL)
    exchange_inflows = fetch_exchange_inflows()
    funding_rate = fetch_funding_rate(SYMBOL)
    open_interest = fetch_open_interest(SYMBOL)
    fng = fetch_fear_and_greed_index()  # (classification, value)
    onchain_data = fetch_onchain_data(SYMBOL)
    extended_data = {
        "order_book_bid": base_data.get("bid", "N/A"),
        "order_book_ask": base_data.get("ask", "N/A"),
        "order_book_spread": base_data.get("spread", "N/A"),
        "exchange_inflows": exchange_inflows,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "fear_and_greed_index": fng
    }
    logging.info("확장 데이터 수집 완료.")

    # 타임프레임별 추세 집계
    bullish_count = 0
    bearish_count = 0
    tf_directions = {}
    for tf, data in multi_tf_data.items():
        if data["sma200"] and data["current_price"] > data["sma200"]:
            bullish_count += 1
            tf_directions[tf] = "bullish"
        else:
            bearish_count += 1
            tf_directions[tf] = "bearish"
    aggregated_trend = "BULL" if bullish_count >= bearish_count else "BEAR"
    aggregated_data = {"trend": aggregated_trend, "timeframe_directions": tf_directions}
    logging.info(f"타임프레임 신호: {tf_directions} (집계: {aggregated_trend})")

    # 주 타임프레임 결정
    primary_tf = choose_primary_timeframe(multi_tf_data)
    aggregated_data["primary_tf"] = primary_tf
    primary_direction = tf_directions.get(primary_tf, None)
    logging.info(f"주 타임프레임: {primary_tf} (방향: {primary_direction})")

    # 명확한 신호 여부 확인 (알고리즘으로 결정 가능한 경우)
    signal_clear, agg_trend = is_signal_clear(multi_tf_data)

    # 시장 장세 판단 (여러 지표 통합)
    market_regime = classify_market_regime(multi_tf_data, onchain_data)
    # 장세에 따른 거래 파라미터 설정
    trade_params = adjust_trade_parameters(market_regime, primary_tf)

    # 만약 명확한 신호가 있다면 알고리즘 결정, 아니면 GPT 호출
    if signal_clear:
        logging.info("명확한 신호 감지 - 알고리즘 결정 사용")
        decision = determine_trading_action_by_rules(agg_trend, trade_params, current_price)
    else:
        try:
            gpt_response = generate_trading_decision(wallet_balance_str, position_info, aggregated_data,
                                                     extended_data, onchain_data, multi_tf_data,
                                                     market_regime, trade_params)
            logging.info("GPT 원시 응답:")
            logging.info(gpt_response)
            decision = parse_trading_decision(gpt_response)
        except Exception as e:
            logging.error(f"GPT 결정 생성/파싱 오류: {e}")
            return

    log_decision(decision, SYMBOL)

    # 위험/보상 비율 계산
    rr = compute_risk_reward(decision, current_price)
    rr_text = f"{rr:.2f}" if rr is not None else "N/A"

    # 포지션 진입 전, 일일 손실 한도 초과 여부 확인
    if daily_loss >= (wallet_balance * DAILY_LOSS_LIMIT_PERCENT / 100):
        logging.warning("일일 손실 한도 초과 - 추가 거래 중단")
        send_telegram_message("일일 손실 한도 초과로 오늘은 거래하지 않습니다.")
        return

    # 진입 조건: 위험/보상 비율 2 이상 및 명확한 매수/매도 신호
    if not in_position:
        if rr is not None and rr >= 2 and decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
            # 동적 포지션 사이징 (예: 계좌 1% 위험)
            risk_percent = 1.0
            pos_size = compute_position_size(wallet_balance, risk_percent, current_price, float(decision["sl_price"]))
            logging.info(f"매매 신호 포착: {decision['final_action']} at {current_price} (포지션 사이즈: {pos_size:.4f})")
            log_open_position(SYMBOL, decision, current_price, pos_size)
            in_position = True
            current_position_side = decision["final_action"].split()[-1]  # LONG 또는 SHORT
            current_position_entry_price = current_price
            current_position_size = pos_size

            # 진입 알림 (자세한 정보 포함)
            action_text = "매수" if decision["final_action"].upper() == "GO LONG" else "매도"
            entry_msg = (f"{action_text} 진입 신호!\n"
                         f"심볼: {SYMBOL}\n진입가: {current_price}\n"
                         f"포지션 사이즈: {pos_size:.4f}\n"
                         f"TP: {decision['tp_price']}, SL: {decision['sl_price']}\n"
                         f"거래 기간: {decision['trade_term']}\n"
                         f"레버리지: {decision['leverage']}\n"
                         f"세부내용: {decision['rationale']}")
            send_telegram_message(entry_msg)
        else:
            logging.info("위험/보상 기준 미충족 또는 매매 신호 없음 - 진입하지 않음.")
    else:
        # 포지션이 열려 있을 경우: 알고리즘에 따라 포지션 관리 (여기서는 단순 청산)
        # 예시: HOLD 상태가 아니라면 청산
        if decision["final_action"].upper() not in ["HOLD LONG", "HOLD SHORT"]:
            logging.info(f"{SYMBOL} 포지션 청산: 현재 가격 {current_price}")
            log_closed_position(SYMBOL, current_position_entry_price, current_price, current_position_side)
            # 포지션 종료 시 손익 계산 후 일일 손실에 반영 (여기서는 단순 합산)
            profit = (current_price - current_position_entry_price) if current_position_side.upper() == "LONG" else (
                        current_position_entry_price - current_price)
            daily_loss += -profit if profit < 0 else 0  # 손실이면 더함
            in_position = False
            exit_msg = (f"포지션 청산!\n심볼: {SYMBOL}\n청산가: {current_price}\n"
                        f"수익: {profit:.2f}\n현재 일일 누적 손실: {daily_loss:.2f}")
            send_telegram_message(exit_msg)
        else:
            logging.info("현재 포지션 유지 중.")


if __name__ == "__main__":
    main()
