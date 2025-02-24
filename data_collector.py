import ccxt.pro
import asyncio
import pandas as pd
import logging
from datetime import datetime

# 상세 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("data_collector.log"), logging.StreamHandler()]
)


class DataCollector:
    def __init__(self, exchange_name='binance', symbol='BTC/USDT', timeframes=['5m', '15m', '1h', '4h', '1d']):
        self.exchange = getattr(ccxt.pro, exchange_name)({'enableRateLimit': True})
        self.symbol = symbol
        self.timeframes = timeframes
        self.candle_data = {tf: [] for tf in timeframes}
        self.running = True

    async def fetch_real_time_data(self):
        while self.running:
            try:
                await self.exchange.watch_ohlcv(self.symbol, self.timeframes[0])  # 초기 연결 확인
                tasks = [
                    self._fetch_ohlcv_for_timeframe(tf) for tf in self.timeframes
                ]
                await asyncio.gather(*tasks)
            except Exception as e:
                logging.error(f"Connection error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def _fetch_ohlcv_for_timeframe(self, timeframe):
        while self.running:
            try:
                ohlcv = await self.exchange.watch_ohlcv(self.symbol, timeframe)
                if ohlcv:
                    latest_candle = ohlcv[-1]
                    self.candle_data[timeframe].append(latest_candle)
                    # 상세 로그: 캔들 데이터 기록
                    logging.info(
                        f"[{timeframe}] New candle - Time: {datetime.fromtimestamp(latest_candle[0] / 1000)}, "
                        f"Open: {latest_candle[1]}, High: {latest_candle[2]}, Low: {latest_candle[3]}, "
                        f"Close: {latest_candle[4]}, Volume: {latest_candle[5]}"
                    )
                    # 최대 1000개 캔들 유지
                    if len(self.candle_data[timeframe]) > 1000:
                        self.candle_data[timeframe].pop(0)
                await asyncio.sleep(0.1)  # CPU 부하 감소
            except Exception as e:
                logging.error(f"[{timeframe}] Error fetching data: {e}")
                await asyncio.sleep(2)  # 오류 발생 시 재시도

    def stop(self):
        self.running = False


async def main():
    collector = DataCollector()
    await collector.fetch_real_time_data()


if __name__ == "__main__":
    asyncio.run(main())
