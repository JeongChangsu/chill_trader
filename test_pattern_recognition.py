# test_pattern_recognition.py

import os
import asyncio
from data_acquisition import DataAcquisition
from pattern_recognition import PatternRecognition

api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')


async def main():
    # 1. 데이터 수집 (Data Acquisition)
    symbol = 'BTC/USDT'
    exchange_id = 'binance'

    data_acquisition = DataAcquisition(exchange_id, symbol, api_key, api_secret)
    await data_acquisition.load_markets()
    # 1시간 봉 데이터
    ohlcv_1h = await data_acquisition.fetch_ohlcv(timeframe='1h', limit=200)
    await data_acquisition.close()
    # 2. 패턴 인식 (Pattern Recognition)
    pattern_recognition = PatternRecognition({'1h': ohlcv_1h})

    # 캔들 패턴
    candle_patterns = pattern_recognition.recognize_candle_patterns('1h')
    print("Candlestick Patterns:\n", candle_patterns)

    # 차트 패턴
    all_patterns = pattern_recognition.get_all_patterns('1h')
    print("All Detected Patterns:\n", all_patterns)


if __name__ == '__main__':
    asyncio.run(main())
