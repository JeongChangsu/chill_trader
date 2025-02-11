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
from datetime import datetime
from selenium.webdriver.common.by import By
from openai import OpenAI

# ===========================================
# 로그 설정 (모든 로그를 한국어로 출력)
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

# OPENAI_API_KEY는 환경변수에 설정되어 있어야 합니다.
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# 텔레그램 메시지 전송을 위한 환경변수 (텔레그램 봇 토큰과 채팅 ID)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

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
    """
    Binance에서 OHLCV 데이터를 ccxt를 통해 가져오고 pandas DataFrame으로 반환합니다.
    """
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
    다양한 기술적 지표를 계산하여 DataFrame에 추가합니다.
    포함 지표: RSI, SMA20, SMA50, SMA200, MACD, ATR, OBV, MFI, Bollinger Bands, 이동평균 차이(%) 등.
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

    # 이동평균 백분율 차이
    df['ma50_diff'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['ma200_diff'] = (df['close'] - df['sma200']) / df['sma200'] * 100

    logging.info("기술적 지표 계산 완료")
    return df

def fetch_order_book(symbol):
    """
    Binance의 주문서 데이터(최우선 bid, ask, 스프레드)를 반환합니다.
    """
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

# ===========================================
# 1-2. 확장 데이터 수집 (크롤링, 온체인 데이터 등)
# ===========================================
def fetch_exchange_inflows():
    """
    무료 소스(예: CryptoQuant)에서 거래소 유입/유출 데이터를 크롤링합니다.
    실제 사이트 구조에 맞게 XPath를 수정해야 합니다.
    """
    url = "https://cryptoquant.com/asset/btc/chart/exchange-flows"  # 예시 URL
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
    """
    Binance Futures 데이터를 사용하여 펀딩율을 반환합니다.
    """
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
    """
    Binance Futures 데이터를 사용하여 미결제약정을 반환합니다.
    """
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
    """
    Alternative.me에서 Fear & Greed Index를 가져옵니다.
    """
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
    온체인 지표 (예: MVRV, SOPR)를 가져옵니다.
    실제 운영 시 Glassnode, CryptoQuant, Coin Metrics 등의 API를 활용할 수 있습니다.
    여기서는 예시로 더미 데이터를 사용합니다.
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
    """
    다중 타임프레임의 OHLCV 데이터와 기술적 지표를 계산하여 반환합니다.
    반환형은 타임프레임별 최신 값에 대한 요약 dict입니다.
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
            "timestamp": latest['timestamp']
        }
    logging.info("다중 타임프레임 데이터 및 지표 계산 완료")
    return multi_tf_data

# ===========================================
# 2. 시장 장세 판단 및 동적 지표 기준 조정
# ===========================================
def determine_market_regime(multi_tf_data, onchain_data):
    """
    1시간봉 데이터를 기준으로 SMA50, SMA200와의 관계 및 가격의 이격 정도를 보고
    - 가격이 SMA들 근처(예: ±1% 이내)이면 neutral (중립)
    - 둘 다 위에 있으면 bull, 둘 다 아래면 bear,
    - 그렇지 않으면 neutral로 판단합니다.

    추가로 온체인 데이터(MVRV, SOPR)가 극단적(예: MVRV, SOPR < 1)인 경우도 neutral로 보정합니다.
    """
    data = multi_tf_data.get("1h")
    if data is None:
        logging.warning("1시간봉 데이터가 없어 중립(m-neutral)으로 판정")
        return "neutral"
    current_price = data["current_price"]
    sma50 = data["sma50"]
    sma200 = data["sma200"]
    if sma50 is None or sma200 is None:
        base_regime = "neutral"
    else:
        # ±1% 내에 있으면 중립으로 판단
        if abs(current_price - sma50) / sma50 < 0.01 and abs(current_price - sma200) / sma200 < 0.01:
            base_regime = "neutral"
        elif current_price > sma50 and current_price > sma200:
            base_regime = "bull"
        elif current_price < sma50 and current_price < sma200:
            base_regime = "bear"
        else:
            base_regime = "neutral"

    # 온체인 데이터 보정 (예시: MVRV와 SOPR 모두 1 미만이면 지나치게 과매도 상태 -> 중립으로 판단)
    if onchain_data["mvrv"] != "N/A" and onchain_data["sopr"] != "N/A":
        if onchain_data["mvrv"] < 1 and onchain_data["sopr"] < 1:
            logging.info("온체인 지표가 극단적이어서 시장 장세를 중립으로 보정")
            return "neutral"
    return base_regime

def adjust_indicator_thresholds(market_regime):
    """
    시장 장세에 따라 RSI, MACD, MA 관련 기준을 동적으로 설정합니다.
    """
    if market_regime == "bull":
        thresholds = {
            "rsi_oversold": 45,
            "rsi_overbought": 80,
            "macd_comment": "강한 상승장에서는 가벼운 하락 MACD 크로스는 무시하고, 깊은 조정 시 진입 고려.",
            "ma_comment": "50MA 지지선에서 반등 시 매수 고려."
        }
    elif market_regime == "bear":
        thresholds = {
            "rsi_oversold": 20,
            "rsi_overbought": 55,
            "macd_comment": "하락장에서는 가벼운 상승 MACD 크로스는 무시하고, 모멘텀 확인 시 진입 고려.",
            "ma_comment": "50MA 저항선에서 반등 시 매도 고려."
        }
    else:  # neutral
        thresholds = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_comment": "중립장에서는 MACD 크로스를 보조 신호로 활용.",
            "ma_comment": "50MA 주변에서 평균 회귀 전략 적용."
        }
    logging.info(f"시장 장세({market_regime})에 따른 지표 기준 설정 완료")
    return thresholds

def choose_primary_timeframe(multi_tf_data):
    """
    알고리즘적으로 주 타임프레임(primary timeframe)을 선택합니다.
    여기서는 SMA200 대비 가격의 이격률(절대값 비율)이 가장 큰 타임프레임을
    강한 추세 신호가 있는 것으로 보고 주 타임프레임으로 선택합니다.
    """
    primary_tf = None
    max_diff = 0
    for tf, data in multi_tf_data.items():
        if data["sma200"] is not None:
            diff = abs(data["current_price"] - data["sma200"]) / data["sma200"]
            if diff > max_diff:
                max_diff = diff
                primary_tf = tf
    logging.info(f"주 타임프레임으로 '{primary_tf}' 선택 (SMA200 대비 최대 이격률: {max_diff:.2%})")
    return primary_tf

# ===========================================
# 3. GPT 프롬프트 생성 및 매매 결정
# ===========================================
def generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data, onchain_data, multi_tf_data,
                        market_regime, thresholds):
    """
    GPT에게 전달할 XML 형태의 상세 프롬프트를 생성합니다.
    여기에는 계좌정보, 다중 타임프레임 요약, 기술적/온체인/심리 데이터, 그리고 전략 전환 지침이 포함됩니다.
    """
    # 다중 타임프레임 요약 생성
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

    # Fear & Greed Index (클래스, 값)
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
                In bullish conditions, consider RSI oversold around {thresholds['rsi_oversold']} (instead of 30) and overbought near {thresholds['rsi_overbought']}. Analyze divergence and failure swings.
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
                Only open a new position if the potential reward is at least 2x the risk.
            </Guide>
        </RiskReward>
        <StrategySwitch>
            <Guide>
                Based on market regime: adopt a trend-following strategy in bullish conditions, a mean reversion approach in neutral or bearish conditions.
                Consider technical, on-chain, and sentiment data to decide the optimal strategy.
            </Guide>
        </StrategySwitch>
    </DecisionRules>
    <Task>
        Based on the provided account info, market context, indicators (technical and on-chain), and dynamic decision rules, decide whether to GO LONG, GO SHORT, HOLD LONG, HOLD SHORT, or NO TRADE.
        Then, recommend a leverage multiplier (e.g., 3x, 5x, etc.), a trade term, a take profit price, a stop loss price, and a specific limit order price.
        Output your decision as a single comma-separated line with the following fields:
        Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Brief Rationale.
    </Task>
    <OutputExample>
        GO LONG, 5x, 6h, 11400, 10800, 11300, Majority of timeframes show bullish momentum with low volatility and adjusted RSI indicating a dip entry.
    </OutputExample>
</TradeBotPrompt>
    """
    return prompt

def generate_trading_decision(wallet_balance, position_info, aggregated_data, extended_data, onchain_data,
                              multi_tf_data, market_regime, thresholds):
    prompt = generate_gpt_prompt(wallet_balance, position_info, aggregated_data, extended_data, onchain_data,
                                 multi_tf_data, market_regime, thresholds)
    developer_prompt = (
        "You are an automated BTC/USDT trading assistant that follows detailed quantitative indicators and strict guidelines "
        "as defined in the provided XML. Do not deviate from the specified output format. "
        "Based on market regime, adopt either a trend-following or mean reversion strategy accordingly. "
        "Consider all provided technical, on-chain, and sentiment data in your decision."
    )
    response = client.chat.completions.create(
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
    GPT의 응답을 파싱합니다.
    예상 출력 형식 (콤마 구분): Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Brief Rationale
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
# 4. 포지션 로깅 함수들
# ===========================================
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
    logging.info("매매 결정이 로그 파일에 기록되었습니다.")

def log_open_position(symbol, decision, entry_price):
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
    logging.info(f"{symbol} 포지션 개시 로그 기록됨 (진입가: {entry_price}).")

def log_closed_position(symbol, entry_price, exit_price, trade_side):
    profit = (exit_price - entry_price) if trade_side.upper() == "LONG" else (entry_price - exit_price)
    file_exists = os.path.isfile(CLOSED_POSITIONS_FILE)
    with open(CLOSED_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "trade_side", "entry_price", "exit_price", "profit"])
        writer.writerow([datetime.utcnow(), symbol, trade_side, entry_price, exit_price, profit])
    logging.info(f"{symbol} 포지션 종료 로그 기록됨 (수익: {profit}).")

# ===========================================
# 5. 포지션 관리 및 메인 트레이딩 로직
# ===========================================
def compute_risk_reward(decision, entry_price):
    """
    절대 가격을 기준으로 위험/보상 비율을 계산합니다.
    LONG: (TP Price - entry_price) / (entry_price - SL Price)
    SHORT: (entry_price - TP Price) / (SL Price - entry_price)
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
        logging.info(f"위험/보상 비율 계산 결과: {rr_ratio:.2f}")
        return rr_ratio
    except Exception as e:
        logging.error(f"위험/보상 비율 계산 중 오류 발생: {e}")
        return None

def fetch_additional_data(symbol):
    """
    기본 데이터와 확장 데이터를 병합하여 반환합니다.
    """
    base_data = fetch_order_book(symbol)
    exchange_inflows = fetch_exchange_inflows()
    funding_rate = fetch_funding_rate(symbol)
    open_interest = fetch_open_interest(symbol)
    fear_and_greed = fetch_fear_and_greed_index()  # (classification, value)
    onchain = fetch_onchain_data(symbol)

    extended_data = {
        "order_book_bid": base_data.get("bid", "N/A"),
        "order_book_ask": base_data.get("ask", "N/A"),
        "order_book_spread": base_data.get("spread", "N/A"),
        "exchange_inflows": exchange_inflows,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "fear_and_greed_index": fear_and_greed  # Tuple (classification, value)
    }
    logging.info("확장 데이터 및 온체인 데이터 수집 완료")
    return extended_data, onchain

def main():
    logging.info("트레이딩 봇 실행 시작")
    wallet_balance = "1000 USDT"
    position_info = "NONE"  # 초기 포지션 정보
    in_position = False
    current_position_side = None
    current_position_entry_price = 0.0

    # 다중 타임프레임 데이터 수집 (5m, 15m, 1h, 4h, 1d)
    multi_tf_data = fetch_multi_tf_data(SYMBOL, timeframes=["5m", "15m", "1h", "4h", "1d"], limit=300)
    if not multi_tf_data or "1h" not in multi_tf_data:
        logging.error("데이터를 충분히 수집하지 못했습니다.")
        return

    # 1시간봉 기준 현재 가격
    current_price = multi_tf_data["1h"]["current_price"]
    logging.info(f"현재 1시간봉 가격: {current_price}")

    # 확장 데이터 (주문서, 펀딩율, 미결제약정, 거래소 유입/유출, Fear & Greed Index, 온체인 데이터)
    extended_data, onchain_data = fetch_additional_data(SYMBOL)

    # 각 타임프레임별 bullish/bearish 신호 (SMA200 기준) 계산
    bullish_count = 0
    bearish_count = 0
    tf_directions = {}
    for tf, data in multi_tf_data.items():
        if data["sma200"] is not None and data["current_price"] > data["sma200"]:
            bullish_count += 1
            tf_directions[tf] = "bullish"
        else:
            bearish_count += 1
            tf_directions[tf] = "bearish"
    aggregated_trend = "BULL" if bullish_count >= bearish_count else "BEAR"
    aggregated_data = {"trend": aggregated_trend, "timeframe_directions": tf_directions}
    logging.info(f"타임프레임별 신호: {tf_directions} (집계: {aggregated_trend})")

    # 주 타임프레임 선택 (알고리즘적으로 SMA200 대비 이격률이 가장 큰 타임프레임)
    primary_tf = choose_primary_timeframe(multi_tf_data)
    primary_direction = tf_directions.get(primary_tf, None)
    logging.info(f"주 타임프레임: {primary_tf} (방향: {primary_direction})")

    # 합의(consensus) 조건 확인: 기본적으로 4/5 합의 또는 주 타임프레임 신호 기준
    if not (bullish_count >= 4 or bearish_count >= 4):
        if primary_direction is None or not ((primary_direction == "bullish" and bullish_count >= 3) or
                                              (primary_direction == "bearish" and bearish_count >= 3)):
            logging.info("명확한 시장 신호가 확인되지 않아 매매를 권장하지 않습니다.")
            return

    # 시장 장세 결정 (기술적 데이터와 온체인 데이터를 모두 반영)
    market_regime = determine_market_regime(multi_tf_data, onchain_data)  # "bull", "bear", "neutral"
    thresholds = adjust_indicator_thresholds(market_regime)
    logging.info(f"시장 장세 판정 결과: {market_regime}")

    # GPT 프롬프트를 이용한 매매 결정 생성
    try:
        gpt_response = generate_trading_decision(wallet_balance, position_info, aggregated_data,
                                                 extended_data, onchain_data, multi_tf_data,
                                                 market_regime, thresholds)
        logging.info("GPT 원시 응답:")
        logging.info(gpt_response)
        decision = parse_trading_decision(gpt_response)
        logging.info("파싱된 매매 결정:")
        logging.info(decision)
        log_decision(decision, SYMBOL)
    except Exception as e:
        logging.error(f"GPT 매매 결정 생성 및 파싱 중 오류 발생: {e}")
        return

    # GPT 응답 후 위험/보상 비율 계산 및 텔레그램 알림 전송 (매매 실행 여부와 관계없이)
    rr = compute_risk_reward(decision, current_price)
    rr_text = f"{rr:.2f}" if rr is not None else "N/A"
    alert_msg = (f"GPT 매매 결정 알림:\n"
                 f"심볼: {SYMBOL}\n"
                 f"최종 액션: {decision['final_action']}\n"
                 f"위험/보상 비율: {rr_text}\n"
                 f"상세: {decision['rationale']}")
    send_telegram_message(alert_msg)

    # 포지션 관리: 위험/보상 비율이 2 이상이어야 하며, GO LONG 또는 GO SHORT 명령일 때만 포지션 진입
    if not in_position:
        if rr is not None and rr >= 2 and decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
            logging.info(f"유리한 매매 설정 발견. 포지션 진입: {decision['final_action']} at {current_price}")
            log_open_position(SYMBOL, decision, current_price)
            in_position = True
            current_position_side = decision["final_action"].split()[-1]  # LONG 또는 SHORT
            current_position_entry_price = current_price

            # 포지션 진입 시, long 또는 short 신호에 대해 텔레그램 메시지 전송
            action = "매수" if decision["final_action"].upper() == "GO LONG" else "매도"
            entry_msg = (f"{action} 진입 신호!\n심볼: {SYMBOL}\n진입가: {current_price}\n"
                         f"상세: {decision['rationale']}")
            send_telegram_message(entry_msg)
        else:
            logging.info("위험/보상 비율 기준에 맞는 매매 설정을 찾지 못했습니다.")
    else:
        # 포지션이 열려 있는 경우, HOLD LONG/HOLD SHORT 외의 명령이면 청산
        if decision["final_action"].upper() not in ["HOLD LONG", "HOLD SHORT"]:
            logging.info(f"{SYMBOL} 포지션 청산: 현재 가격 {current_price}")
            log_closed_position(SYMBOL, current_position_entry_price, current_price, current_position_side)
            in_position = False
        else:
            logging.info("현재 포지션 유지 중.")

if __name__ == "__main__":
    main()
