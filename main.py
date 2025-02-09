import ta
import re
import os
import ccxt
import time
import pandas as pd
import csv
from datetime import datetime
from openai import OpenAI

# GPT 클라이언트 초기화 (환경변수에 OPENAI_API_KEY가 있어야 함)
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# 글로벌 설정
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
DECISIONS_LOG_FILE = "trading_decisions.csv"
OPEN_POSITIONS_FILE = "open_positions.csv"
CLOSED_POSITIONS_FILE = "closed_positions.csv"

##############################################
# 1. 시장 데이터 및 추가 데이터 받아오기
##############################################

def fetch_market_data(symbol, timeframe, limit=300):
    """
    Binance에서 지정한 심볼과 타임프레임의 OHLC 데이터를 받아와서,
    기술적 지표(RSI, SMA50, SMA200, MACD, ATR 등)를 계산한 후
    필요한 값을 딕셔너리 형태로 반환합니다.
    """
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching data for {symbol} on {timeframe}: {e}")
        return None

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # 기술적 지표 계산
    try:
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi_indicator.rsi()
        sma50_indicator = ta.trend.SMAIndicator(close=df['close'], window=50)
        sma200_indicator = ta.trend.SMAIndicator(close=df['close'], window=200)
        df['sma50'] = sma50_indicator.sma_indicator()
        df['sma200'] = sma200_indicator.sma_indicator()
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['signal'] = macd.macd_signal()
        atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()
    except Exception as e:
        print(f"Error computing indicators for {symbol} on {timeframe}: {e}")
        return None

    latest = df.iloc[-1]
    data = {
        "timeframe": timeframe,
        "rsi_value": round(latest['rsi'], 2) if not pd.isna(latest['rsi']) else None,
        "ma50_value": round(latest['sma50'], 2) if not pd.isna(latest['sma50']) else None,
        "ma200_value": round(latest['sma200'], 2) if not pd.isna(latest['sma200']) else None,
        "macd_line": round(latest['macd'], 2) if not pd.isna(latest['macd']) else None,
        "signal_line": round(latest['signal'], 2) if not pd.isna(latest['signal']) else None,
        "current_volume": round(latest['volume'], 2),
        "average_volume": round(df['volume'].mean(), 2),
        "volatility_index": round(latest['atr'], 2) if not pd.isna(latest['atr']) else None,
        "current_price": round(latest['close'], 2)
    }
    support = round(df['low'].min(), 2)
    resistance = round(df['high'].max(), 2)
    data["support_resistance_info"] = f"Support around {support}, Resistance around {resistance}"

    # 단순 HTF 추세: 현재 가격이 SMA200 이상이면 bullish, 아니면 bearish
    if data["ma200_value"] is not None and data["current_price"] > data["ma200_value"]:
        data["market_trend"] = "bull"
    else:
        data["market_trend"] = "bear"

    # Divergence 정보(추후 고도화 가능)
    data["divergence_info"] = "None detected"
    return data

def fetch_additional_data(symbol):
    """
    펀딩비, 미결제약정, 오더북 정보를 포함한 추가 데이터를 가져옵니다.
    """
    # Binance Futures 엔드포인트 사용 (심볼은 "BTC/USDT" -> "BTCUSDT" 형태)
    try:
        futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        symbol_futures = symbol.replace("/", "")
        funding_response = futures_exchange.fetch_funding_rate(symbol=symbol_futures)
        latest_funding = funding_response[-1]['fundingRate'] if funding_response else None
        oi_response = futures_exchange.fetch_open_interest(symbol=symbol_futures)
        open_interest = oi_response['openInterest'] if oi_response and 'openInterest' in oi_response else None
    except Exception as e:
        print(f"Error fetching futures data for {symbol}: {e}")
        latest_funding = None
        open_interest = None

    # 오더북 정보 (Spot)
    exchange = ccxt.binance()
    try:
        order_book = exchange.fetch_order_book(symbol)
        bid = order_book['bids'][0][0] if order_book['bids'] else None
        ask = order_book['asks'][0][0] if order_book['asks'] else None
        spread = round(ask - bid, 2) if bid and ask else None
    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")
        bid = ask = spread = None

    return {
        "funding_rate": latest_funding,
        "open_interest": open_interest,
        "order_book_bid": bid,
        "order_book_ask": ask,
        "order_book_spread": spread
    }

def fetch_multi_tf_market_data(symbol, timeframes=TIMEFRAMES, limit=300):
    """
    지정된 다중 타임프레임에 대해 시장 데이터를 받아옵니다.
    """
    multi_tf_data = {}
    for tf in timeframes:
        data = fetch_market_data(symbol, tf, limit)
        if data:
            multi_tf_data[tf] = data
        time.sleep(0.5)  # 레이트 제한 회피
    return multi_tf_data

def aggregate_market_trend(multi_tf_data):
    """
    여러 타임프레임의 데이터를 바탕으로 전반적인 시장 추세(bull or bear)를 산출합니다.
    """
    bull_count = 0
    bear_count = 0
    for tf, data in multi_tf_data.items():
        if data is None:
            continue
        if data.get("ma200_value") is not None and data.get("current_price") > data.get("ma200_value"):
            bull_count += 1
        else:
            bear_count += 1
    return "bull" if bull_count >= bear_count else "bear"

def generate_multi_tf_summary(multi_tf_data):
    """
    각 타임프레임의 주요 지표 요약 텍스트를 생성합니다.
    """
    summary_lines = []
    for tf, data in multi_tf_data.items():
        if data:
            line = (f"{tf}: Price={data['current_price']}, RSI={data['rsi_value']}, 50MA={data['ma50_value']}, "
                    f"200MA={data['ma200_value']}, Trend={data['market_trend']}, Support/Res={data['support_resistance_info']}")
            summary_lines.append(line)
    return "\n".join(summary_lines)

##############################################
# 2. GPT 프롬프트와 자동매매 결정 함수 (멀티타임프레임 및 추가 데이터 반영)
##############################################

def generate_trading_decision(
        wallet_balance,
        current_position_asset,
        current_position_side,
        current_position_entry_price,
        current_position_size,
        aggregated_market_trend,
        multi_tf_summary,
        additional_data,
        divergence_info,
):
    """
    GPT 프롬프트를 통해 다중 타임프레임 및 추가 데이터를 기반으로 최종 의사결정을 생성합니다.
    최종 출력은 아래 7가지 필드:
    Final Action, Recommended Leverage, Trade Term, TP Level, SL Level, Limit Order Price, Brief Rationale
    TP와 SL은 현재 가격 대비 퍼센트(예: "+12%", "-5%")로 답변해주세요.
    """
    developer_prompt = (
        "You are an automated cryptocurrency trading assistant that analyzes multiple coins using real-time multi-timeframe market data "
        "and additional metrics such as funding rate, open interest, and order book information. You also consider liquidation maps and exit strategies."
    )
    user_prompt = f"""Based on the guidelines below, generate a final decision including the following parameters: 
- Final Action: "LONG", "SHORT", "CLOSE", or "HOLD"
- Recommended Leverage (e.g., "3x", "5x", etc.)
- Trade Term (e.g., "intraday", "6h", "1d", "1w")
- Take Profit (TP) level (as a percentage change from the current price, e.g., "+12%")
- Stop Loss (SL) level (as a percentage change from the current price, e.g., "-5%")
- Limit Order Price for entry (the price at which to place a limit order)
- Brief Rationale

-----------------------------------------------------
ACCOUNT OVERVIEW:
- Current wallet balance: {wallet_balance}
- Current positions:
  - Asset: {current_position_asset}
  - Side: {current_position_side}
  - Entry price: {current_position_entry_price}
  - Size: {current_position_size}

-----------------------------------------------------
MARKET CONTEXT:
- Aggregated market trend (from multiple timeframes): {aggregated_market_trend}
- Additional Data:
  - Funding Rate: {additional_data.get('funding_rate', 'N/A')}
  - Open Interest: {additional_data.get('open_interest', 'N/A')}
  - Order Book: Bid = {additional_data.get('order_book_bid', 'N/A')}, Ask = {additional_data.get('order_book_ask', 'N/A')}, Spread = {additional_data.get('order_book_spread', 'N/A')}
- Divergence info: {divergence_info}

-----------------------------------------------------
MULTI-TIMEFRAME ANALYSIS:
{multi_tf_summary}

-----------------------------------------------------
LEVERAGE, TP/SL & TRADE TERM GUIDELINES:
1. Leverage: Recommend higher leverage (e.g., 3x-5x) in strong bullish/bearish conditions and lower leverage (e.g., 1x-2x) in uncertain markets.
2. Trade Term: For short-term opportunities specify "intraday" or "6h"; for longer trends specify "1d" or "1w".
3. TP/SL: Provide TP and SL levels as percentages relative to current price ensuring at least a 1:2 risk-reward ratio.
4. Entry: Recommend a limit order price for entry based on technical analysis and key support/resistance levels.
5. Include liquidation map considerations if applicable.

-----------------------------------------------------
OUTPUT REQUIREMENTS:
- Provide the final decision in a single comma-separated line with the following fields in order:
  Final Action, Recommended Leverage, Trade Term, TP Level, SL Level, Limit Order Price, Brief Rationale
- Do not include any internal chain-of-thought or additional commentary.
- Ensure your output is clear, direct, and strictly follows the guidelines above.

END
"""
    response = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def parse_trading_decision(response_text):
    """
    예상 출력 형식 (한 줄, 콤마로 구분):
    Final Action, Recommended Leverage, Trade Term, TP Level, SL Level, Limit Order Price, Brief Rationale
    예) "LONG, 3x, 6h, +12%, -5%, 21000, RSI is oversold and price is above key moving averages"
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
# 3. 다중 코인 분석, 최적 포지션 선정, 로깅 및 포지션 모니터링
##############################################

def get_top_volume_coins(limit=5):
    """
    Binance의 USDT 거래쌍 중 거래량이 높은 상위 N개의 코인 심볼을 반환합니다.
    """
    exchange = ccxt.binance()
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        print("Error fetching tickers:", e)
        return []
    usdt_tickers = {symbol: data for symbol, data in tickers.items() if symbol.endswith("/USDT")}
    sorted_symbols = sorted(usdt_tickers.keys(), key=lambda s: usdt_tickers[s].get("quoteVolume", 0), reverse=True)
    return sorted_symbols[:limit]

def compute_risk_reward(decision, current_price, trade_side):
    """
    TP와 SL 퍼센트(예: "+12%", "-5%")를 기반으로 위험대비 수익비를 계산합니다.
    """
    try:
        tp_pct = float(re.sub(r'[%+]', '', decision["tp_level"]))
        sl_pct = float(re.sub(r'[%+-]', '', decision["sl_level"]))
        # LONG/SHORT 모두 절대값 기준으로 계산
        if trade_side.upper() == "LONG":
            risk = abs(sl_pct)
            reward = abs(tp_pct)
        elif trade_side.upper() == "SHORT":
            risk = abs(sl_pct)
            reward = abs(tp_pct)
        else:
            return None
        if risk == 0:
            return None
        return reward / risk
    except Exception as e:
        print("Error computing risk-reward:", e)
        return None

def log_decision(decision, symbol):
    """
    의사결과 데이터를 CSV 파일에 기록합니다.
    """
    file_exists = os.path.isfile(DECISIONS_LOG_FILE)
    with open(DECISIONS_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "final_action", "leverage", "trade_term", "tp_level", "sl_level", "limit_order_price", "rationale"])
        writer.writerow([datetime.utcnow(), symbol, decision["final_action"], decision["leverage"], decision["trade_term"],
                         decision["tp_level"], decision["sl_level"], decision["limit_order_price"], decision["rationale"]])

def log_open_position(symbol, decision, entry_price):
    """
    오픈된 포지션 정보를 CSV 파일에 기록합니다.
    """
    file_exists = os.path.isfile(OPEN_POSITIONS_FILE)
    with open(OPEN_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "entry_price", "leverage", "trade_term", "tp_level", "sl_level", "limit_order_price", "rationale"])
        writer.writerow([datetime.utcnow(), symbol, decision["final_action"], entry_price, decision["leverage"], decision["trade_term"],
                         decision["tp_level"], decision["sl_level"], decision["limit_order_price"], decision["rationale"]])

def log_closed_position(symbol, entry_price, exit_price, trade_side):
    """
    청산된 포지션과 수익률을 CSV 파일에 기록합니다.
    """
    profit = (exit_price - entry_price) if trade_side.upper() == "LONG" else (entry_price - exit_price)
    file_exists = os.path.isfile(CLOSED_POSITIONS_FILE)
    with open(CLOSED_POSITIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "trade_side", "entry_price", "exit_price", "profit"])
        writer.writerow([datetime.utcnow(), symbol, trade_side, entry_price, exit_price, profit])

##############################################
# 4. 메인 로직: 다중 코인 분석 후 최적 포지션 선정
##############################################

def main():
    # 계좌 정보 (예시)
    wallet_balance = "5000 USDT"
    current_position_asset = "None"   # 현재 포지션 없을 경우
    current_position_side = "NONE"
    current_position_entry_price = "0"
    current_position_size = "0"

    top_coins = get_top_volume_coins(limit=2)
    print("Top volume coins:", top_coins)
    best_coin = None
    best_decision = None
    best_rr = -1
    best_current_price = None

    # 각 상위 코인별로 분석
    for symbol in top_coins:
        print(f"\nAnalyzing {symbol}...")
        multi_tf_data = fetch_multi_tf_market_data(symbol)
        if not multi_tf_data:
            continue
        aggregated_trend = aggregate_market_trend(multi_tf_data)
        multi_tf_summary = generate_multi_tf_summary(multi_tf_data)
        additional_data = fetch_additional_data(symbol)
        divergence_info = multi_tf_data.get("1h", {}).get("divergence_info", "N/A")
        try:
            gpt_response = generate_trading_decision(
                wallet_balance=wallet_balance,
                current_position_asset=current_position_asset,
                current_position_side=current_position_side,
                current_position_entry_price=current_position_entry_price,
                current_position_size=current_position_size,
                aggregated_market_trend=aggregated_trend,
                multi_tf_summary=multi_tf_summary,
                additional_data=additional_data,
                divergence_info=divergence_info,
            )
            print("Raw GPT Response:")
            print(gpt_response)
            decision = parse_trading_decision(gpt_response)
            print("Parsed Decision:")
            print(decision)
            log_decision(decision, symbol)
        except Exception as e:
            print(f"Error generating/parsing decision for {symbol}: {e}")
            continue

        # 1시간봉 기준 현재 가격 사용
        current_price = multi_tf_data.get("1h", {}).get("current_price")
        if current_price is None:
            continue
        rr = compute_risk_reward(decision, current_price, decision["final_action"])
        if rr is not None:
            print(f"Risk-Reward ratio for {symbol}: {rr:.2f}")
            if rr > best_rr and decision["final_action"].upper() in ["LONG", "SHORT"]:
                best_rr = rr
                best_coin = symbol
                best_decision = decision
                best_current_price = current_price

    if best_coin:
        print(f"\nSelected coin for trade: {best_coin} with risk-reward ratio: {best_rr:.2f}")
        # 지정가 주문 추천 가격 (GPT에서 제안한 가격 사용)
        limit_order_price = best_decision["limit_order_price"]
        print(f"Placing limit order for {best_coin} at price: {limit_order_price}")
        log_open_position(best_coin, best_decision, best_current_price)
    else:
        print("No suitable trade found based on current analysis.")

if __name__ == "__main__":
    main()
