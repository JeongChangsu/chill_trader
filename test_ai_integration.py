# test_ai_integration.py

import asyncio
from data_acquisition import DataAcquisition
from technical_indicators import TechnicalIndicators
from pattern_recognition import PatternRecognition
from market_context_analysis import MarketContextAnalysis
from trigger_condition_check import TriggerConditions
from ai_integration import GeminiIntegration
import os
import time
from functools import partial

api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')


async def process_data(timeframe, ohlcv_data, data_acquisition):
    """
    Processes OHLCV data for a given timeframe.

    Args:
        timeframe (str): The timeframe (e.g., '1m', '5m', '1h', '4h', '1d').
        ohlcv_data (pd.DataFrame): The OHLCV data for the timeframe.
        data_acquisition : data_acquisition 객체
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing data for timeframe: {timeframe}")

    # 2. 기술적 지표 계산
    indicators = TechnicalIndicators({timeframe: ohlcv_data})
    try:
        all_indicators = indicators.calculate_all(timeframe, {
            'sma': 20,
            'ema': 20,
            'wma': 20,
            'adx': 14,
            'sar_acceleration': 0.02,
            'sar_maximum': 0.2,
            'supertrend_period': 10,
            'supertrend_multiplier': 3,
            'bollinger_period': 20,
            'bollinger_stddev': 2,
            'atr': 14,
            'keltner_period': 20,
            'keltner_multiplier': 2,
            'donchian_period': 20,
            'mfi': 14,
            'cmf': 20,
            'rsi': 14,
            'stochastic_fastk': 14,
            'stochastic_slowk': 3,
            'stochastic_slowd': 3,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'cci': 20,
        })
    except Exception as e:
        print("indicator calculate error")
        return

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Indicators: {all_indicators}")  # 지표

    # 3. 패턴 인식
    patterns = PatternRecognition({timeframe: ohlcv_data})
    detected_patterns = patterns.get_all_patterns(timeframe)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Patterns: {detected_patterns}")

    # 4. 시장 상황 분석
    market_analysis = MarketContextAnalysis(indicators, patterns, timeframe)  # Pass 'timeframe'
    market_context = market_analysis.analyze()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Market Context: {market_context}")

    # 5. 트리거 조건 검사
    trigger_conditions = TriggerConditions(indicators, patterns, timeframe, market_context)  # Pass 'timeframe'
    trigger_type = trigger_conditions.check_conditions()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trigger Type: {trigger_type}")

    # 6. AI 통합 (Gemini) - 필요한 경우
    if trigger_type:
        google_api_key = os.environ.get('GOOGLE_API_KEY')
        if not google_api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            return

        gemini_integration = GeminiIntegration(google_api_key)
        ai_decision = gemini_integration.get_ai_decision(
            market_context=market_context,
            indicators=all_indicators,
            patterns=detected_patterns
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] AI Decision: {ai_decision}")


async def main():
    # 1. 데이터 수집
    symbol = 'BTC/USDT'
    exchange_id = 'binance'
    data_acquisition = DataAcquisition(exchange_id, symbol, api_key, api_secret)
    await data_acquisition.load_markets()

    # 지원되는 모든 타임프레임 정의
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w']

    # 각 타임프레임에 대한 콜백 함수 등록 및 watch_ohlcv 작업 생성
    for timeframe in timeframes:
        callback_with_args = partial(process_data, timeframe, data_acquisition=data_acquisition)
        data_acquisition.set_callback(timeframe, callback_with_args)
        asyncio.create_task(data_acquisition.watch_ohlcv(timeframe))

    start_time = time.time()
    print("실시간 데이터 수집 및 분석 시작 (1시간 동안 실행)...")

    # 메인 루프 (1시간 동안 실행 또는 다른 종료 조건)
    while time.time() - start_time <= 3600:
        await asyncio.sleep(5)  # 5초 대기

    await data_acquisition.close()
    print("실시간 데이터 수집 및 분석 종료.")


if __name__ == '__main__':
    asyncio.run(main())
