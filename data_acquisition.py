# data_acquisition.py (수정)

import ccxt.pro
import asyncio
import pandas as pd


class DataAcquisition:
    def __init__(self, exchange_id, symbol, api_key, api_secret):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.exchange = getattr(ccxt.pro, exchange_id)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            },
        })
        self.ohlcv_data = {}
        self.callbacks = {}  # 콜백 함수들을 저장할 딕셔너리

    async def load_markets(self):
        await self.exchange.load_markets()

    async def fetch_ohlcv(self, timeframe='1m', since=None, limit=None, params={}):
        """Fetches historical OHLCV data."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe, since, limit, params)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            self.ohlcv_data[timeframe] = df
            return df
        except ccxt.NetworkError as e:
            print(f'Network error: {e}')
        except ccxt.ExchangeError as e:
            print(f'Exchange error: {e}')
        except Exception as e:
            print(f'Error fetching OHLCV: {e}')

    def set_callback(self, timeframe, callback):
        """
        Registers a callback function to be called when new OHLCV data is available.

        Args:
            timeframe (str): The timeframe to watch.
            callback (function): The callback function.  It should accept a Pandas DataFrame as input.
        """
        if timeframe not in self.callbacks:
            self.callbacks[timeframe] = []
        self.callbacks[timeframe].append(callback)

    async def watch_ohlcv(self, timeframe='1m'):
        """Streams real-time OHLCV data and calls registered callbacks."""
        while True:  # 무한 루프 (실시간 스트리밍)
            try:
                candles = await self.exchange.watch_ohlcv(self.symbol, timeframe)
                if timeframe not in self.ohlcv_data:
                    await self.fetch_ohlcv(timeframe)  # 초기 데이터 로드

                if candles:
                    last_candle = candles[-1]
                    df = self.ohlcv_data[timeframe]
                    last_df_candle = df.iloc[-1]

                    if last_df_candle.name == pd.to_datetime(last_candle[0], unit='ms'):
                        # Update the last candle
                        df.loc[last_df_candle.name, 'open'] = last_candle[1]
                        df.loc[last_df_candle.name, 'high'] = max(last_df_candle['high'], last_candle[2])
                        df.loc[last_df_candle.name, 'low'] = min(last_df_candle['low'], last_candle[3])
                        df.loc[last_df_candle.name, 'close'] = last_candle[4]
                        df.loc[last_df_candle.name, 'volume'] = last_candle[5]
                    else:
                        # Create a new candle
                        new_candle = pd.DataFrame([[last_candle[0], last_candle[1], last_candle[2], last_candle[3],
                                                    last_candle[4], last_candle[5]]],
                                                  columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        new_candle['datetime'] = pd.to_datetime(new_candle['timestamp'], unit='ms')
                        new_candle.set_index('datetime', inplace=True)
                        self.ohlcv_data[timeframe] = pd.concat([df, new_candle])

                    # Call registered callbacks
                    if timeframe in self.callbacks:
                        for callback in self.callbacks[timeframe]:
                            await callback(self.ohlcv_data[timeframe])  # 수정된 부분

            except ccxt.NetworkError as e:
                print(f'Network error: {e}')
            except ccxt.ExchangeError as e:
                print(f'Exchange error: {e}')
            except Exception as e:
                print(f'Error watching OHLCV: {e}')

    async def watch_trades(self):
        """Streams real-time trade data using websockets."""
        while True:
            try:
                trades = await self.exchange.watch_trades(self.symbol)
                # Process new trades
                for trade in trades:
                    print(f"New trade: {trade['side']} {trade['amount']} @ {trade['price']}")
            except ccxt.NetworkError as e:
                print(f'Network error: {e}')
            except ccxt.ExchangeError as e:
                print(f'Exchange error: {e}')
            except Exception as e:
                print(f'Error watching trades: {e}')

    async def watch_order_book(self):
        """Streams real-time order book data using websockets."""
        while True:
            try:
                order_book = await self.exchange.watch_order_book(self.symbol)
                # Process order book updates (bids, asks)
                print(
                    f"Order book: Bids={order_book['bids'][:5]}, Asks={order_book['asks'][:5]}")  # Show top 5 bids and asks
            except ccxt.NetworkError as e:
                print(f'Network error: {e}')
            except ccxt.ExchangeError as e:
                print(f'Exchange error: {e}')
            except Exception as e:
                print(f'Error watching order book: {e}')

    async def close(self):
        if self.exchange:
            await self.exchange.close()
