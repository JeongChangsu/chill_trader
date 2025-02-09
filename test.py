import os
import re
import time
import csv
import requests
import pandas as pd
import numpy as np
import ccxt
import ta
from datetime import datetime
from openai import OpenAI

# GPT 클라이언트 초기화 (환경변수에 OPENAI_API_KEY가 있어야 함)
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# 글로벌 설정
SYMBOL = "BTC/USDT"
DECISIONS_LOG_FILE = "trading_decisions.csv"
OPEN_POSITIONS_FILE = "open_positions.csv"
CLOSED_POSITIONS_FILE = "closed_positions.csv"
TIMEFRAMES = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}  # 다중 타임프레임 분석 (필요 시 추가)


##############################################
# 1. 데이터 수집 및 지표 계산 함수들
##############################################

def fetch_ohlcv(symbol, timeframe, limit=300):
    """
    ccxt를 이용해 Binance에서 OHLCV 데이터를 가져오고 DataFrame으로 반환
    """
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching OHLCV for {symbol} on {timeframe}: {e}")
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def compute_technical_indicators(df):
    """
    df에 여러 기술적 지표를 계산 후 컬럼 추가
    """
    # RSI (14)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # 단기, 중기, 장기 이동평균: 20, 50, 200
    df['sma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # ATR (14)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

    # MFI (14)
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()

    return df


def fetch_order_book(symbol):
    """
    Binance에서 스팟 오더북 데이터를 가져옴
    """
    exchange = ccxt.binance()
    try:
        order_book = exchange.fetch_order_book(symbol)
        bid = order_book['bids'][0][0] if order_book['bids'] else None
        ask = order_book['asks'][0][0] if order_book['asks'] else None
        spread = round(ask - bid, 2) if bid and ask else None
    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")
        bid = ask = spread = None
    return {"bid": bid, "ask": ask, "spread": spread}


def fetch_additional_data(symbol):
    """
    추가 데이터: 펀딩률, 미결제약정
    Binance Futures API 사용
    """
    try:
        # Futures 데이터 호출
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        symbol_futures = symbol.replace("/", "")
        latest_funding = futures_exchange.fetch_funding_rate(symbol=symbol)
        oi_response = futures_exchange.fetch_open_interest(symbol=symbol_futures)
        open_interest = oi_response['openInterest'] if oi_response and 'openInterest' in oi_response else None
    except Exception as e:
        print(f"Error fetching futures data for {symbol}: {e}")
        latest_funding = None
        open_interest = None

    order_book_data = fetch_order_book(symbol)
    additional_data = {
        "funding_rate": latest_funding,
        "open_interest": open_interest,
        "order_book_bid": order_book_data["bid"],
        "order_book_ask": order_book_data["ask"],
        "order_book_spread": order_book_data["spread"]
    }
    return additional_data


def fetch_multi_tf_data(symbol, timeframes=["1h", "4h", "1d"], limit=300):
    """
    여러 타임프레임에서 데이터 및 지표를 계산
    """
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
            "macd": round(latest['macd'], 2) if not np.isnan(latest['macd']) else None,
            "macd_signal": round(latest['macd_signal'], 2) if not np.isnan(latest['macd_signal']) else None,
            "atr": round(latest['atr'], 2) if not np.isnan(latest['atr']) else None,
            "obv": round(latest['obv'], 2) if not np.isnan(latest['obv']) else None,
            "mfi": round(latest['mfi'], 2) if not np.isnan(latest['mfi']) else None,
            "timestamp": latest['timestamp']
        }
    return multi_tf_data


##############################################
# 2. GPT 프롬프트 생성 및 의사결정 함수
##############################################

def generate_gpt_prompt(wallet_balance, position_info, aggregated_data, additional_data, multi_tf_data):
    """
    GPT에게 최종 트레이딩 결정을 요청하기 위한 구체적인 프롬프트 생성
    각 지표에 대해 정량적 기준과 가중치를 포함함.
    """
    # 각 타임프레임에 대한 상세 요약 작성
    multi_tf_summary = ""
    for tf, data in multi_tf_data.items():
        multi_tf_summary += (
            f"{tf} - Price: {data['current_price']}, RSI: {data['rsi']}, "
            f"SMA20: {data['sma20']}, SMA50: {data['sma50']}, SMA200: {data['sma200']}, "
            f"MACD: {data['macd']} (Signal: {data['macd_signal']}), ATR: {data['atr']}, "
            f"OBV: {data['obv']}, MFI: {data['mfi']}\n"
        )

    # 가격 상대 지표 비교 예시 (수치 예시, 실제 백테스트로 조정 필요)
    # - RSI: 0~30이면 과매도, 70~100이면 과매수, 가중치 1.0
    # - MFI: 0~20 과매도, 80~100 과매수, 가중치 0.8
    # - MACD: MACD > Signal이면 긍정적, 가중치 1.0; 그 반대면 부정적
    # - OBV: 최근 5캔들 추세 확인, 상승이면 긍정, 하락이면 부정, 가중치 0.7
    # - SMA20, SMA50, SMA200: 가격이 각 SMA보다 높으면 긍정, 낮으면 부정, 가중치 각각 0.5, 0.7, 1.0
    # - ATR: 변동성이 낮으면 (ATR/Price < 0.005) 레버리지 상향 (최대 10x), 높으면 (ATR/Price > 0.02) 레버리지 하향 (최소 1x)

    prompt = f"""
You are an automated BTC/USDT trading assistant that analyzes real-time multi-timeframe data along with additional market metrics.
Below are the details of the current market situation and account status.

ACCOUNT:
- Wallet Balance: {wallet_balance}
- Current Position: {position_info} 

MARKET DATA:
{multi_tf_summary}

ADDITIONAL DATA:
- Funding Rate: {additional_data.get('funding_rate', 'N/A')}
- Open Interest: {additional_data.get('open_interest', 'N/A')}
- Order Book: Bid = {additional_data.get('order_book_bid', 'N/A')}, Ask = {additional_data.get('order_book_ask', 'N/A')}, Spread = {additional_data.get('order_book_spread', 'N/A')}

Guidelines for decision-making (use quantitative indicators and assign weights as provided):
1. RSI:
   - If RSI is between 0 and 30, consider it oversold (weight 1.0 for long).
   - If RSI is between 70 and 100, consider it overbought (weight 1.0 for short).
2. MFI:
   - If MFI is between 0 and 20, consider it oversold (weight 0.8 for long).
   - If MFI is between 80 and 100, consider it overbought (weight 0.8 for short).
3. MACD:
   - If MACD > Signal, consider it bullish (weight 1.0 for long).
   - If MACD < Signal, consider it bearish (weight 1.0 for short).
4. OBV:
   - Evaluate OBV trend over the last 5 candles. If rising, add a weight of 0.7 for long; if falling, add 0.7 for short.
5. Moving Averages:
   - If current price > SMA20, add weight 0.5 for long; if < SMA20, add weight 0.5 for short.
   - If current price > SMA50, add weight 0.7 for long; if < SMA50, add weight 0.7 for short.
   - If current price > SMA200, add weight 1.0 for long; if < SMA200, add weight 1.0 for short.
6. ATR and Volatility:
   - Calculate ATR as a percentage of price. If ATR/Price < 0.005, market is quiet, and higher leverage (up to 10x) may be recommended for long positions.
   - If ATR/Price > 0.02, market is volatile, so lower leverage (down to 1x) is suggested.
7. Position Management:
   - If already in a position, first decide whether to exit based on trend reversal or reaching stop-loss/take-profit criteria.
   - If no position, evaluate opening a new one only if the aggregated weighted scores indicate a clear long or short signal.
8. Risk/Reward:
   - Only open a new position if the potential reward is at least twice the risk (2:1).
9. Provide a final decision in a single comma-separated line with the following fields:
   Final Action (GO LONG, GO SHORT, HOLD LONG, HOLD SHORT, or NO TRADE),
   Recommended Leverage (e.g., 3x, 5x, etc.),
   Trade Term (e.g., intraday, 6h, 1d, 1w),
   Take Profit Level (as a percentage change, e.g., +12%),
   Stop Loss Level (as a percentage change, e.g., -5%),
   Limit Order Price (specific price for limit entry),
   Brief Rationale.

Ensure the decision is based on the quantified signals above and their assigned weights.
Respond strictly with the final decision in the required comma-separated format.
"""
    return prompt


def generate_trading_decision(wallet_balance, position_info, aggregated_data, additional_data, multi_tf_data):
    prompt = generate_gpt_prompt(wallet_balance, position_info, aggregated_data, additional_data, multi_tf_data)
    developer_prompt = "You are an automated BTC/USDT trading assistant that strictly follows quantitative indicators and weights for decision-making."
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
    출력 형식:
    Final Action, Recommended Leverage, Trade Term, TP Level, SL Level, Limit Order Price, Brief Rationale
    """
    response_text = response_text.strip()
    parts = [part.strip() for part in re.split(r'\s*,\s*', response_text)]
    if len(parts) < 7:
        raise ValueError("Incomplete response. Expected at least 7 comma-separated fields.")
    decision = {
        "final_action": parts[0],
        "leverage": parts[1],
        "trade_term": parts[2],
        "tp_level": parts[3],
        "sl_level": parts[4],
        "limit_order_price": parts[5],
        "rationale": ", ".join(parts[6:])
    }
    return decision


##############################################
# 3. 포지션 로깅 함수들
##############################################

def log_decision(decision, symbol):
    file_exists = os.path.isfile(DECISIONS_LOG_FILE)
    with open(DECISIONS_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "final_action", "leverage", "trade_term", "tp_level", "sl_level",
                             "limit_order_price", "rationale"])
        writer.writerow(
            [datetime.utcnow(), symbol, decision["final_action"], decision["leverage"], decision["trade_term"],
             decision["tp_level"], decision["sl_level"], decision["limit_order_price"], decision["rationale"]])


def log_open_position(symbol, decision, entry_price):
    file_exists = os.path.isfile(OPEN_POSITIONS_FILE)
    with open(OPEN_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["timestamp", "symbol", "action", "entry_price", "leverage", "trade_term", "tp_level", "sl_level",
                 "limit_order_price", "rationale"])
        writer.writerow([datetime.utcnow(), symbol, decision["final_action"], entry_price, decision["leverage"],
                         decision["trade_term"],
                         decision["tp_level"], decision["sl_level"], decision["limit_order_price"],
                         decision["rationale"]])


def log_closed_position(symbol, entry_price, exit_price, trade_side):
    profit = (exit_price - entry_price) if trade_side.upper() == "LONG" else (entry_price - exit_price)
    file_exists = os.path.isfile(CLOSED_POSITIONS_FILE)
    with open(CLOSED_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "trade_side", "entry_price", "exit_price", "profit"])
        writer.writerow([datetime.utcnow(), symbol, trade_side, entry_price, exit_price, profit])


##############################################
# 4. 포지션 관리 및 메인 로직
##############################################

def compute_risk_reward(decision, current_price, trade_side):
    """
    TP와 SL 퍼센트를 기반으로 위험대비 수익비 계산
    """
    try:
        tp_pct = float(re.sub(r'[%+]', '', decision["tp_level"]))
        sl_pct = float(re.sub(r'[%+-]', '', decision["sl_level"]))
        risk = abs(sl_pct)
        reward = abs(tp_pct)
        if risk == 0:
            return None
        return reward / risk
    except Exception as e:
        print("Error computing risk-reward:", e)
        return None


def main():
    wallet_balance = "10000 USDT"
    # 현재 포지션 정보: 예) "NONE" 또는 "LONG from 20000" 또는 "SHORT from 21000"
    position_info = "NONE"  # 초기에는 포지션 없음
    in_position = False
    current_position_side = None
    current_position_entry_price = 0.0

    # BTC/USDT 전용 데이터 수집 (1시간봉 중심, 보조 4h, 1d)
    multi_tf_data = fetch_multi_tf_data(SYMBOL, timeframes=["1h", "4h", "1d"], limit=300)
    if not multi_tf_data or "1h" not in multi_tf_data:
        print("Not enough data fetched.")
        return

    # 기본적으로 1시간봉 데이터를 기준으로 현재 가격 결정
    current_price = multi_tf_data["1h"]["current_price"]
    additional_data = fetch_additional_data(SYMBOL)

    # aggregated_data: 여러 타임프레임의 시장 추세 종합 판단 (예: 가격이 SMA200 위에 있는지 등)
    # 간단 예시로 1시간, 4시간, 1일 모두에서 가격이 SMA200 위에 있으면 bull, 아니면 bear
    bull_count = 0
    bear_count = 0
    for tf, data in multi_tf_data.items():
        if data["sma200"] is not None and data["current_price"] > data["sma200"]:
            bull_count += 1
        else:
            bear_count += 1
    aggregated_trend = "BULL" if bull_count >= bear_count else "BEAR"
    aggregated_data = {"trend": aggregated_trend}

    # GPT 프롬프트 기반 의사결정
    try:
        gpt_response = generate_trading_decision(wallet_balance, position_info, aggregated_data, additional_data,
                                                 multi_tf_data)
        print("Raw GPT Response:")
        print(gpt_response)
        decision = parse_trading_decision(gpt_response)
        print("Parsed Decision:")
        print(decision)
        log_decision(decision, SYMBOL)
    except Exception as e:
        print("Error generating/parsing GPT decision:", e)
        return

    # 포지션 관리: 현재 포지션이 없으면 신규 진입 판단, 있으면 종료 조건 판단
    if not in_position:
        # 신규 포지션 진입 시 Risk/Reward 비율이 2 이상이어야 함
        rr = compute_risk_reward(decision, current_price, decision["final_action"])
        if rr is not None and rr >= 2 and decision["final_action"].upper() in ["GO LONG", "GO SHORT"]:
            print(f"Opening new position: {decision['final_action']} at {current_price}")
            log_open_position(SYMBOL, decision, current_price)
            in_position = True
            current_position_side = decision["final_action"].split()[-1]  # "LONG" or "SHORT"
            current_position_entry_price = current_price
        else:
            print("No favorable trade setup found.")
    else:
        # 포지션 관리: 기존 포지션 종료 여부 판단
        # (여기서는 단순 예시로, GPT가 HOLD 명령이 아닌 경우 포지션 종료로 판단)
        if decision["final_action"].upper() not in ["HOLD LONG", "HOLD SHORT"]:
            print(f"Closing position for {SYMBOL} at price {current_price}")
            log_closed_position(SYMBOL, current_position_entry_price, current_price, current_position_side)
            in_position = False
        else:
            print("Holding current position.")


if __name__ == "__main__":
    main()
