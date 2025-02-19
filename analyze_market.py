import ccxt
import pandas as pd
import pandas_ta as ta
import json
import logging
from datetime import datetime, timedelta

# 로깅 설정 (PEP8 스타일)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_market_data(exchange_id: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    거래소에서 특정 암호화폐의 OHLCV 데이터를 가져오는 함수.

    Args:
        exchange_id (str): 거래소 ID (예: 'binance').
        symbol (str): 암호화폐 심볼 (예: 'BTC/USDT').
        timeframe (str): 캔들 시간 프레임 (예: '4h', '1d').

    Returns:
        pd.DataFrame: OHLCV 데이터를 담은 Pandas DataFrame.
                      컬럼: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                      인덱스: timestamp (datetime object)
    """
    try:
        exchange = getattr(ccxt, exchange_id)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Seoul')  # UTC to KST
        df.set_index('timestamp', inplace=True)
        logging.info(f"Successfully fetched {symbol} {timeframe} data from {exchange_id}")
        return df
    except Exception as e:
        logging.error(f"Error fetching {symbol} {timeframe} data from {exchange_id}: {e}")
        return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 데이터프레임에 기술적 지표를 추가하는 함수.

    Args:
        df (pd.DataFrame): OHLCV 데이터프레임 (fetch_market_data 함수 결과).

    Returns:
        pd.DataFrame: 기술적 지표가 추가된 Pandas DataFrame.
                      추가 지표: MA50, MA200, RSI, MACD_hist
    """
    try:
        df['MA50'] = ta.sma(df['close'], length=50)
        df['MA200'] = ta.sma(df['close'], length=200)
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast_period=12, slow_period=26, signal_period=9)
        df['MACD_hist'] = macd['MACDh_12_26_9']  # MACD Histogram
        logging.info("Successfully calculated technical indicators.")
        return df
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return df


def analyze_market_situation(df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
    """
    4시간봉 및 일봉 데이터프레임을 기반으로 시장 상황을 분석하는 함수.

    Args:
        df_4h (pd.DataFrame): 4시간봉 데이터프레임 (기술 지표 포함).
        df_1d (pd.DataFrame): 일봉 데이터프레임 (기술 지표 포함).

    Returns:
        dict: 시장 상황 분석 결과 (JSON 포맷).
              예시: {"market_situation_category": "Bull Market", "confidence_level": "High", "supporting_rationale": "..."}
    """
    market_situation = "Range-Bound Market"  # 기본값
    confidence_level = "Medium"
    rationale_points = []

    # 4시간봉 분석
    ma50_4h = df_4h['MA50'].iloc[-1]
    ma200_4h = df_4h['MA200'].iloc[-1]
    rsi_4h = df_4h['RSI'].iloc[-1]
    macd_hist_4h = df_4h['MACD_hist'].iloc[-1]
    close_4h = df_4h['close'].iloc[-1]

    # 1일봉 분석
    ma50_1d = df_1d['MA50'].iloc[-1]
    ma200_1d = df_1d['MA200'].iloc[-1]
    rsi_1d = df_1d['RSI'].iloc[-1]
    macd_hist_1d = df_1d['MACD_hist'].iloc[-1]
    close_1d = df_1d['close'].iloc[-1]

    # 불장 판단 조건 (Bull Market Conditions)
    if (ma50_4h > ma200_4h and close_4h > ma200_4h and rsi_4h > 50 and macd_hist_4h > 0) and \
            (ma50_1d > ma200_1d and close_1d > ma200_1d and rsi_1d > 50 and macd_hist_1d > 0):
        market_situation = "Bull Market"
        confidence_level = "High"
        rationale_points.append("Strong Bull Market signals across both 4h and 1d timeframes.")
        if ma50_4h > ma200_4h: rationale_points.append("4h MA50 > MA200 (Golden Cross)")
        if close_4h > ma200_4h: rationale_points.append("4h Price above MA200")
        if rsi_4h > 60: rationale_points.append("4h RSI > 60 (Bullish Momentum)")  # RSI 60으로 상향
        if macd_hist_4h > 0: rationale_points.append("4h MACD Histogram Positive")
        if ma50_1d > ma200_1d: rationale_points.append("1d MA50 > MA200 (Golden Cross)")
        if close_1d > ma200_1d: rationale_points.append("1d Price above MA200")
        if rsi_1d > 60: rationale_points.append("1d RSI > 60 (Bullish Momentum)")  # RSI 60으로 상향
        if macd_hist_1d > 0: rationale_points.append("1d MACD Histogram Positive")


    # 베어장 판단 조건 (Bear Market Conditions)
    elif (ma50_4h < ma200_4h and close_4h < ma200_4h and rsi_4h < 50 and macd_hist_4h < 0) and \
            (ma50_1d < ma200_1d and close_1d < ma200_1d and rsi_1d < 50 and macd_hist_1d < 0):
        market_situation = "Bear Market"
        confidence_level = "High"
        rationale_points.append("Strong Bear Market signals across both 4h and 1d timeframes.")
        if ma50_4h < ma200_4h: rationale_points.append("4h MA50 < MA200 (Death Cross)")
        if close_4h < ma200_4h: rationale_points.append("4h Price below MA200")
        if rsi_4h < 40: rationale_points.append("4h RSI < 40 (Bearish Momentum)")  # RSI 40으로 하향
        if macd_hist_4h < 0: rationale_points.append("4h MACD Histogram Negative")
        if ma50_1d < ma200_1d: rationale_points.append("1d MA50 < MA200 (Death Cross)")
        if close_1d < ma200_1d: rationale_points.append("1d Price below MA200")
        if rsi_1d < 40: rationale_points.append("1d RSI < 40 (Bearish Momentum)")  # RSI 40으로 하향
        if macd_hist_1d < 0: rationale_points.append("1d MACD Histogram Negative")

    # 박스권 판단 조건 (Range-Bound Market Conditions) - (불장/베어장 조건에 해당하지 않는 경우)
    else:
        market_situation = "Range-Bound Market"
        confidence_level = "Medium"
        rationale_points.append("Mixed signals, indicating a Range-Bound Market.")
        if not (ma50_4h > ma200_4h and close_4h > ma200_4h and rsi_4h > 50 and macd_hist_4h > 0):
            rationale_points.append(
                "4h timeframe showing mixed or neutral signals (MA, RSI, MACD not strongly bullish).")
        if not (ma50_1d > ma200_1d and close_1d > ma200_1d and rsi_1d > 50 and macd_hist_1d > 0):
            rationale_points.append(
                "1d timeframe showing mixed or neutral signals (MA, RSI, MACD not strongly bullish).")
        if not (ma50_4h < ma200_4h and close_4h < ma200_4h and rsi_4h < 50 and macd_hist_4h < 0):
            rationale_points.append("4h timeframe not showing strong bearish signals.")
        if not (ma50_1d < ma200_1d and close_1d < ma200_1d and rsi_1d < 50 and macd_hist_1d < 0):
            rationale_points.append("1d timeframe not showing strong bearish signals.")

    analysis_result = {
        "market_situation_category": market_situation,
        "confidence_level": confidence_level,
        "supporting_rationale": ", ".join(rationale_points)
    }

    logging.info(f"Market Situation Analysis Result: {analysis_result}")
    return analysis_result


def main():
    """
    1단계 시장 상황 분석을 수행하는 메인 함수.
    """
    symbol = 'BTC/USDT'
    exchange_id = 'binance'
    timeframe_4h = '4h'
    timeframe_1d = '1d'

    logging.info("Starting Market Situation Analysis...")

    # 1. 데이터 Fetching (Data Fetching)
    df_4h = fetch_market_data(exchange_id, symbol, timeframe_4h)
    df_1d = fetch_market_data(exchange_id, symbol, timeframe_1d)

    if df_4h.empty or df_1d.empty:
        logging.warning("Data fetching failed. Exiting.")
        return

    # 2. 기술 지표 계산 (Indicator Calculation)
    df_4h_with_indicators = calculate_indicators(df_4h)
    df_1d_with_indicators = calculate_indicators(df_1d)

    # 3. 시장 상황 분석 (Market Situation Analysis)
    if not df_4h_with_indicators.empty and not df_1d_with_indicators.empty:
        analysis_result = analyze_market_situation(df_4h_with_indicators, df_1d_with_indicators)

        # 4. JSON 출력 (JSON Output)
        json_output = json.dumps(analysis_result, indent=4, ensure_ascii=False)  # ensure_ascii=False for 한글 (Korean)
        print(json_output)

        logging.info("Market Situation Analysis Completed.")
    else:
        logging.warning("Indicator calculation failed. Market situation analysis aborted.")


if __name__ == "__main__":
    main()
