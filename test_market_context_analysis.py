# test_market_context_analysis.py

import os
import asyncio
from data_acquisition import DataAcquisition
from technical_indicators import TechnicalIndicators
from pattern_recognition import PatternRecognition
from market_context_analysis import MarketContextAnalysis

api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')


async def main():
    # 1. 데이터 수집
    symbol = 'BTC/USDT'
    exchange_id = 'binance'

    data_acquisition = DataAcquisition(exchange_id, symbol, api_key, api_secret)
    await data_acquisition.load_markets()
    ohlcv_1h = await data_acquisition.fetch_ohlcv(timeframe='1h', limit=200)  # Fetch more data
    await data_acquisition.close()

    # 2. 기술적 지표 계산
    indicators = TechnicalIndicators({'1h': ohlcv_1h})

    # 3. 패턴 인식
    patterns = PatternRecognition({'1h': ohlcv_1h})

    # 4. 시장 상황 분석 (온체인/심리 데이터는 예시로 None 전달)
    market_analysis = MarketContextAnalysis(indicators, patterns, '1h', onchain_data=None, sentiment_data=None)
    result = market_analysis.analyze()

    print("Market Context Analysis Result:\n", result)


if __name__ == '__main__':
    asyncio.run(main())
