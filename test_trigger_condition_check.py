# test_trigger_condition_check.py

import os
import asyncio
from data_acquisition import DataAcquisition
from technical_indicators import TechnicalIndicators
from pattern_recognition import PatternRecognition
from market_context_analysis import MarketContextAnalysis
from trigger_condition_check import TriggerConditions

api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')


async def main():
    # 1. 데이터 수집
    symbol = 'BTC/USDT'
    exchange_id = 'binance'
    data_acquisition = DataAcquisition(exchange_id, symbol, api_key, api_secret)
    await data_acquisition.load_markets()
    ohlcv_1h = await data_acquisition.fetch_ohlcv(timeframe='1h', limit=500)
    ohlcv_4h = await data_acquisition.fetch_ohlcv(timeframe='4h', limit=500)
    await data_acquisition.close()

    # 2. 기술적 지표 계산
    indicators = TechnicalIndicators({'1h': ohlcv_1h, '4h': ohlcv_4h})

    # 3. 패턴 인식
    patterns = PatternRecognition({'1h': ohlcv_1h})

    # 4. 시장 상황 분석
    market_analysis = MarketContextAnalysis(indicators, patterns, '1h')
    market_context = market_analysis.analyze()

    # 5. 트리거 조건 검사
    trigger_conditions = TriggerConditions(indicators, patterns, '1h', market_context)  # 수정된 부분
    trigger_type = trigger_conditions.check_conditions()

    print("Market Context:", market_context)
    print("Trigger Type:", trigger_type)


if __name__ == '__main__':
    asyncio.run(main())
