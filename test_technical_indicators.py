# test_technical_indicators.py (continued)

import os
import asyncio
from data_acquisition import DataAcquisition  # Assuming data_acquisition.py is in the same directory
from technical_indicators import TechnicalIndicators

api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')


async def main():
    # 1. 데이터 수집 (Data Acquisition)
    symbol = 'BTC/USDT'
    exchange_id = 'binance'

    data_acquisition = DataAcquisition(exchange_id, symbol, api_key, api_secret)
    await data_acquisition.load_markets()

    # 예시: 1시간봉 데이터 가져오기
    ohlcv_1h = await data_acquisition.fetch_ohlcv(timeframe='1h', limit=100)
    print("Fetched 1h OHLCV data:\n", ohlcv_1h)

    # 2. 기술적 지표 계산 (Technical Indicator Calculation)
    indicators = TechnicalIndicators({'1h': ohlcv_1h})  # Pass the dictionary

    # 모든 지표 계산 (periods 딕셔너리 사용)
    periods = {
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
    }
    all_indicators = indicators.calculate_all('1h', periods)
    print("Calculated indicators for 1h timeframe:\n", all_indicators)

    # 개별 지표 계산 예시
    sma_20 = indicators.calculate_sma('1h', 20)
    print("SMA (20):\n", sma_20)

    ema_50 = indicators.calculate_ema('1h', 50)
    print("EMA (50):\n", ema_50)

    rsi_14 = indicators.calculate_rsi('1h', 14)
    print("RSI (14):\n", rsi_14)

    macd, macdsignal, macdhist = indicators.calculate_macd('1h', 12, 26, 9)
    print("MACD:\n", macd)
    print("MACD Signal:\n", macdsignal)
    print("MACD Histogram:\n", macdhist)

    upperband, middleband, lowerband = indicators.calculate_bollinger_bands('1h', 20, 2)
    print("Bollinger Bands (Upper):\n", upperband)
    print("Bollinger Bands (Middle):\n", middleband)
    print("Bollinger Bands (Lower):\n", lowerband)

    supertrend = indicators.calculate_supertrend('1h', 7, 3)
    print("Supertrend(7,3): \n", supertrend)

    await data_acquisition.close()  # data_acquisition 객체 close


if __name__ == '__main__':
    asyncio.run(main())
