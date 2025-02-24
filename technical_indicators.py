# technical_indicators.py

import talib
import numpy as np
import pandas as pd


class TechnicalIndicators:
    def __init__(self, ohlcv_data):
        """
        Initializes the TechnicalIndicators class with OHLCV data.

        Args:
            ohlcv_data (dict): A dictionary where keys are timeframes (e.g., '1m', '5m', '1h')
                               and values are Pandas DataFrames containing OHLCV data.
        """
        self.ohlcv_data = ohlcv_data

    def calculate_sma(self, timeframe, period):
        """Calculates Simple Moving Average (SMA)."""
        close_prices = self.ohlcv_data[timeframe]['close']
        sma = talib.SMA(close_prices, timeperiod=period)
        return sma

    def calculate_ema(self, timeframe, period):
        """Calculates Exponential Moving Average (EMA)."""
        close_prices = self.ohlcv_data[timeframe]['close']
        ema = talib.EMA(close_prices, timeperiod=period)
        return ema

    def calculate_wma(self, timeframe, period):
        """Calculates Weighted Moving Average (WMA)."""
        close_prices = self.ohlcv_data[timeframe]['close']
        wma = talib.WMA(close_prices, timeperiod=period)
        return wma

    def calculate_adx(self, timeframe, period):
        """Calculates Average Directional Index (ADX)."""
        high_prices = self.ohlcv_data[timeframe]['high']
        low_prices = self.ohlcv_data[timeframe]['low']
        close_prices = self.ohlcv_data[timeframe]['close']
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
        return adx

    def calculate_parabolic_sar(self, timeframe, acceleration=0.02, maximum=0.2):
        """Calculates Parabolic SAR."""
        high_prices = self.ohlcv_data[timeframe]['high']
        low_prices = self.ohlcv_data[timeframe]['low']
        sar = talib.SAR(high_prices, low_prices, acceleration=acceleration, maximum=maximum)
        return sar

    def calculate_supertrend(self, timeframe, period, multiplier):
        """
        Calculates Supertrend.

        Custom implementation, as TA-Lib doesn't have a built-in Supertrend function.
        """
        df = self.ohlcv_data[timeframe].copy()  # Work on a copy to avoid modifying the original

        # Calculate ATR
        df['tr'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['atr'] = talib.SMA(df['tr'], timeperiod=period)  # Using SMA for ATR calculation, as in the reference
        # df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period) # Can be replaced with ATR

        # Calculate basic upper and lower bands
        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['atr']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['atr']

        # Calculate final upper and lower bands
        df['final_ub'] = df['basic_ub'].copy()
        df['final_lb'] = df['basic_lb'].copy()

        for i in range(period, len(df)):
            if df['basic_ub'].iloc[i] < df['final_ub'].iloc[i - 1] or df['close'].iloc[i - 1] > df['final_ub'].iloc[
                i - 1]:
                df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
            else:
                df.loc[df.index[i], 'final_ub'] = df['final_ub'].iloc[i - 1]

            if df['basic_lb'].iloc[i] > df['final_lb'].iloc[i - 1] or df['close'].iloc[i - 1] < df['final_lb'].iloc[
                i - 1]:
                df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
            else:
                df.loc[df.index[i], 'final_lb'] = df['final_lb'].iloc[i - 1]

        # Calculate Supertrend
        df['supertrend'] = 0.0
        df['supertrend'] = df['supertrend'].astype(float)
        for i in range(period, len(df)):
            if df['close'].iloc[i] <= df['final_ub'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_ub'].iloc[i]
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_lb'].iloc[i]

        return df['supertrend']

    def calculate_bollinger_bands(self, timeframe, period, stddev):
        """Calculates Bollinger Bands."""
        close_prices = self.ohlcv_data[timeframe]['close']
        upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=period, nbdevup=stddev, nbdevdn=stddev,
                                                        matype=0)  # matype=0 for SMA
        return upperband, middleband, lowerband

    def calculate_atr(self, timeframe, period):
        """Calculates Average True Range (ATR)."""
        high_prices = self.ohlcv_data[timeframe]['high']
        low_prices = self.ohlcv_data[timeframe]['low']
        close_prices = self.ohlcv_data[timeframe]['close']
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
        return atr

    def calculate_keltner_channels(self, timeframe, period, multiplier):
        """
        Calculates Keltner Channels.

        Custom implementation, as TA-Lib doesn't have a built-in Keltner Channels function.
        """

        df = self.ohlcv_data[timeframe].copy()

        # Calculate EMA
        ema = self.calculate_ema(timeframe, period)  # Use the EMA function we defined
        # Calculate ATR
        atr = self.calculate_atr(timeframe, period)

        # Calculate upper and lower bands
        upper_band = ema + (multiplier * atr)
        lower_band = ema - (multiplier * atr)

        return upper_band, ema, lower_band

    def calculate_donchian_channels(self, timeframe, period):
        """
        Calculates Donchian Channels.

        Custom implementation, as TA-Lib doesn't have a built-in Donchian Channels function.
        """

        df = self.ohlcv_data[timeframe].copy()

        # Calculate upper and lower bands
        upper_band = df['high'].rolling(window=period).max()
        lower_band = df['low'].rolling(window=period).min()

        # Calculate middle band
        middle_band = (upper_band + lower_band) / 2

        return upper_band, middle_band, lower_band

    def calculate_vwap(self, timeframe):
        """Calculates Volume Weighted Average Price (VWAP)."""
        df = self.ohlcv_data[timeframe].copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tpv'] = df['typical_price'] * df['volume']
        df['cum_tpv'] = df['tpv'].cumsum()
        df['cum_volume'] = df['volume'].cumsum()
        vwap = df['cum_tpv'] / df['cum_volume']
        return vwap

    def calculate_obv(self, timeframe):
        """Calculates On Balance Volume (OBV)."""
        close_prices = self.ohlcv_data[timeframe]['close']
        volume = self.ohlcv_data[timeframe]['volume']
        obv = talib.OBV(close_prices, volume)
        return obv

    def calculate_vpt(self, timeframe):
        """
        Calculates Volume-Price Trend (VPT).
        Custom implementation
        """
        df = self.ohlcv_data[timeframe].copy()
        df['price_change'] = df['close'].diff()
        df['vpt'] = (df['volume'] * (df['price_change'] / df['close'].shift(1))).fillna(0).cumsum()
        return df['vpt']

    def calculate_mfi(self, timeframe, period):
        """Calculates Money Flow Index (MFI)."""
        high_prices = self.ohlcv_data[timeframe]['high']
        low_prices = self.ohlcv_data[timeframe]['low']
        close_prices = self.ohlcv_data[timeframe]['close']
        volume = self.ohlcv_data[timeframe]['volume']
        mfi = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=period)
        return mfi

    def calculate_cmf(self, timeframe, period):
        """
        Calculates Chaikin Money Flow (CMF).
        Custom Implementation
        """
        df = self.ohlcv_data[timeframe].copy()

        # Calculate Money Flow Multiplier
        df['mfm'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['mfm'] = df['mfm'].fillna(0)  # Handle potential NaN values

        # Calculate Money Flow Volume
        df['mfv'] = df['mfm'] * df['volume']

        # Calculate CMF
        cmf = df['mfv'].rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

        return cmf

    def calculate_rsi(self, timeframe, period):
        """Calculates Relative Strength Index (RSI)."""
        close_prices = self.ohlcv_data[timeframe]['close']
        rsi = talib.RSI(close_prices, timeperiod=period)
        return rsi

    def calculate_stochastic_oscillator(self, timeframe, fastk_period, slowk_period, slowd_period):
        """Calculates Stochastic Oscillator."""
        high_prices = self.ohlcv_data[timeframe]['high']
        low_prices = self.ohlcv_data[timeframe]['low']
        close_prices = self.ohlcv_data[timeframe]['close']
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=fastk_period,
                                   slowk_period=slowk_period, slowk_matype=0, slowd_period=slowd_period,
                                   slowd_matype=0)  # matype=0 for SMA
        return slowk, slowd

    def calculate_macd(self, timeframe, fastperiod, slowperiod, signalperiod):
        """Calculates Moving Average Convergence Divergence (MACD)."""
        close_prices = self.ohlcv_data[timeframe]['close']
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=fastperiod, slowperiod=slowperiod,
                                                signalperiod=signalperiod)
        return macd, macdsignal, macdhist

    def calculate_cci(self, timeframe, period):
        """Calculates Commodity Channel Index (CCI)."""
        high_prices = self.ohlcv_data[timeframe]['high']
        low_prices = self.ohlcv_data[timeframe]['low']
        close_prices = self.ohlcv_data[timeframe]['close']
        cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=period)
        return cci

    def calculate_all(self, timeframe, periods):
        """
        Calculates all supported indicators for a given timeframe.

        Args:
            timeframe (str): The timeframe for which to calculate indicators (e.g., '1m', '1h').
            periods (dict): A dictionary specifying the periods for each indicator.

        Returns:
            dict: A dictionary containing the calculated indicators.
        """

        indicators = {}

        # Trend Indicators
        indicators['sma'] = self.calculate_sma(timeframe, periods.get('sma', 20))
        indicators['ema'] = self.calculate_ema(timeframe, periods.get('ema', 20))
        indicators['wma'] = self.calculate_wma(timeframe, periods.get('wma', 20))
        indicators['adx'] = self.calculate_adx(timeframe, periods.get('adx', 14))
        indicators['parabolic_sar'] = self.calculate_parabolic_sar(timeframe,
                                                                   acceleration=periods.get('sar_acceleration', 0.02),
                                                                   maximum=periods.get('sar_maximum', 0.2))
        indicators['supertrend'] = self.calculate_supertrend(timeframe, periods.get('supertrend_period', 10),
                                                             periods.get('supertrend_multiplier', 3))

        # Volatility Indicators
        indicators['bollinger_bands'] = self.calculate_bollinger_bands(timeframe, periods.get('bollinger_period', 20),
                                                                       periods.get('bollinger_stddev', 2))
        indicators['atr'] = self.calculate_atr(timeframe, periods.get('atr', 14))
        indicators['keltner_channels'] = self.calculate_keltner_channels(timeframe, periods.get('keltner_period', 20),
                                                                         periods.get('keltner_multiplier', 2))
        indicators['donchian_channels'] = self.calculate_donchian_channels(timeframe,
                                                                           periods.get('donchian_period', 20))

        # Volume Indicators
        indicators['vwap'] = self.calculate_vwap(timeframe)
        indicators['obv'] = self.calculate_obv(timeframe)
        indicators['vpt'] = self.calculate_vpt(timeframe)
        indicators['mfi'] = self.calculate_mfi(timeframe, periods.get('mfi', 14))
        indicators['cmf'] = self.calculate_cmf(timeframe, periods.get('cmf', 20))

        # Momentum Indicators
        indicators['rsi'] = self.calculate_rsi(timeframe, periods.get('rsi', 14))
        indicators['stochastic'] = self.calculate_stochastic_oscillator(timeframe, periods.get('stochastic_fastk', 14),
                                                                        periods.get('stochastic_slowk', 3),
                                                                        periods.get('stochastic_slowd', 3))
        indicators['macd'] = self.calculate_macd(timeframe, periods.get('macd_fast', 12), periods.get('macd_slow', 26),
                                                 periods.get('macd_signal', 9))
        indicators['cci'] = self.calculate_cci(timeframe, periods.get('cci', 20))

        return indicators
