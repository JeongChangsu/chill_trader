import ccxt.pro as ccxtpro
import asyncio
import pandas as pd
import logging
from datetime import datetime

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading.log"), logging.StreamHandler()]
)


class DataCollector:
    def __init__(self, symbol='BTC/USDT', timeframes=None):
        if timeframes is None:
            timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.exchange = ccxtpro.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
        self.symbol = symbol
        self.timeframes = timeframes
        self.candle_data = {tf: pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            for tf in timeframes}
        self.is_running = True

    async def fetch_real_time_data(self):
        while self.is_running:
            try:
                tasks = [self._fetch_tf_data(tf) for tf in self.timeframes]
                await asyncio.gather(*tasks)
            except Exception as e:
                logging.error(f"Critical error in fetch_real_time_data: {e}")
                await asyncio.sleep(5)  # 재연결 대기

    async def _fetch_tf_data(self, timeframe):
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                ohlcv = await self.exchange.watch_ohlcv(self.symbol, timeframe)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                self.candle_data[timeframe] = pd.concat([self.candle_data[timeframe], df]).tail(200)
                logging.info(f"{timeframe} 데이터 수집 성공: Open={df['open'].iloc[-1]}, Close={df['close'].iloc[-1]}, "
                             f"Volume={df['volume'].iloc[-1]}, Timestamp={df['timestamp'].iloc[-1]}")
                retry_count = 0  # 성공 시 재시도 카운트 초기화
            except Exception as e:
                retry_count += 1
                logging.warning(f"{timeframe} 데이터 수집 실패 ({retry_count}/{max_retries}): {e}")
                await asyncio.sleep(2 ** retry_count)  # 지수 백오프


async def main():
    collector = DataCollector()
    await collector.fetch_real_time_data()


if __name__ == "__main__":
    asyncio.run(main())
