# test_data_acquisition.py

import os
import asyncio
from data_acquisition import DataAcquisition  # Import the DataAcquisition class

api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')


async def main():
    # Replace with your actual API key and secret
    symbol = 'BTC/USDT'
    exchange_id = 'binance'  # Use 'binanceusdm' for USD-M Futures

    data_acquisition = DataAcquisition(exchange_id, symbol, api_key, api_secret)

    # Load markets
    await data_acquisition.load_markets()
    print("Markets loaded.")

    # Fetch historical OHLCV data
    ohlcv_df = await data_acquisition.fetch_ohlcv(timeframe='1h', limit=10)
    print("Historical OHLCV data (1h):")
    print(ohlcv_df)

    # Create tasks for streaming data
    watch_ohlcv_task = asyncio.create_task(data_acquisition.watch_ohlcv(timeframe='1m'))
    # watch_trades_task = asyncio.create_task(data_acquisition.watch_trades())
    # watch_order_book_task = asyncio.create_task(data_acquisition.watch_order_book())

    # Run the tasks concurrently
    await asyncio.gather(watch_ohlcv_task)
    # await asyncio.gather(watch_ohlcv_task, watch_trades_task, watch_order_book_task)

    await data_acquisition.close()


if __name__ == '__main__':
    asyncio.run(main())
