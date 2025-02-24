# market_context_analysis.py
import pandas as pd


class MarketContextAnalysis:
    def __init__(self, indicators, patterns, timeframe, onchain_data=None, sentiment_data=None):
        """
        Initializes the MarketContextAnalysis class.

        Args:
            indicators (TechnicalIndicators): TechnicalIndicators object.
            patterns (PatternRecognition): PatternRecognition object.
            timeframe (str): The timeframe for analysis (e.g., '1m', '1h').
            onchain_data (dict, optional): On-chain data. Defaults to None.
            sentiment_data (dict, optional): Market sentiment data. Defaults to None.
        """
        self.indicators = indicators
        self.patterns = patterns
        self.timeframe = timeframe
        self.onchain_data = onchain_data
        self.sentiment_data = sentiment_data
        self.data = self.indicators.ohlcv_data[self.timeframe]  # Get OHLCV data

    def analyze_trend(self):
        """Analyzes the current trend strength."""

        if self.data.empty or len(self.data) < 50:  # Check if enough data
            return "Neutral"

        # 1. Moving Averages (Short-term, Long-term)
        sma_short = self.indicators.calculate_sma(self.timeframe, 20)
        sma_long = self.indicators.calculate_sma(self.timeframe, 50)

        # Check for valid SMA values
        if sma_short is None or sma_long is None or sma_short.iloc[-1] is None or sma_long.iloc[-1] is None:
            return "Neutral"

        # 2. ADX (Average Directional Index - Trend Strength)
        adx = self.indicators.calculate_adx(self.timeframe, 14)
        if adx is None or adx.iloc[-1] is None:
            adx_value = 0  # Default value
        else:
            adx_value = adx.iloc[-1]

        # 3. MACD (Moving Average Convergence Divergence)
        macd, macdsignal, macdhist = self.indicators.calculate_macd(self.timeframe, 12, 26, 9)
        if macd is None or macdsignal is None or macdhist is None or macd.iloc[-1] is None or macdsignal.iloc[
            -1] is None or macdhist.iloc[-1] is None:
            macd_above_signal = False
            macd_hist_positive = False
        else:
            macd_above_signal = macd.iloc[-1] > macdsignal.iloc[-1]
            macd_hist_positive = macdhist.iloc[-1] > 0

        # 4. Supertrend
        supertrend = self.indicators.calculate_supertrend(self.timeframe, 7, 3)
        if supertrend is None or supertrend.iloc[-1] is None:
            supertrend_value = self.data['close'].iloc[-1]
        else:
            supertrend_value = supertrend.iloc[-1]

        # Trend Determination Logic (Rules)
        if (
                sma_short.iloc[-1] > sma_long.iloc[-1]
                and self.data['close'].iloc[-1] > supertrend_value
                and macd_above_signal
                and macd_hist_positive
                and adx_value > 25
        ):
            return "Strong Bullish"
        elif (
                sma_short.iloc[-1] > sma_long.iloc[-1]
                and macd_above_signal
                and adx_value > 20  # Lower threshold for weak trend
        ):
            return "Weak Bullish"

        elif (
                sma_short.iloc[-1] < sma_long.iloc[-1]
                and self.data['close'].iloc[-1] < supertrend_value
                and not macd_above_signal
                and not macd_hist_positive
                and adx_value > 25
        ):
            return "Strong Bearish"

        elif (
                sma_short.iloc[-1] < sma_long.iloc[-1]
                and not macd_above_signal
                and adx_value > 20
        ):
            return "Weak Bearish"

        else:
            return "Neutral"

    def analyze_volatility(self):
        """Analyzes the current volatility."""
        if self.data.empty or len(self.data) < 20:
            return "Medium"  # Default
        # 1. Bollinger Bands Width
        upperband, middleband, lowerband = self.indicators.calculate_bollinger_bands(self.timeframe, 20, 2)
        if upperband is None or middleband is None or lowerband is None or upperband.iloc[-1] is None or \
                middleband.iloc[-1] is None or lowerband.iloc[-1] is None:
            bb_width = 0.01  # Default
        else:
            bb_width = (upperband.iloc[-1] - lowerband.iloc[-1]) / middleband.iloc[-1]

        # 2. ATR (Average True Range)
        atr = self.indicators.calculate_atr(self.timeframe, 14)
        if atr is None or atr.iloc[-1] is None:
            atr_value = 0.01  # Default
        else:
            atr_value = atr.iloc[-1] / self.data['close'].iloc[-1]  # Normalize ATR by price

        # Volatility Determination Logic
        if bb_width > 0.05 or atr_value > 0.03:  # Example thresholds
            return "High"
        elif bb_width < 0.02 or atr_value < 0.01:
            return "Low"
        else:
            return "Medium"

    def analyze_volume(self):
        """Analyzes the current volume."""

        if self.data.empty or len(self.data) < 20:
            return "Average"

        # 1. Compare current volume to a short-term moving average of volume
        volume_sma = self.data['volume'].rolling(window=5).mean()  # Short-term MA
        if volume_sma.iloc[-1] is None:
            return "Average"

        current_volume = self.data['volume'].iloc[-1]
        if current_volume > volume_sma.iloc[-1] * 2:  # Example: 2x the average
            return "Surge"
        elif current_volume > volume_sma.iloc[-1] * 1.2:
            return "Above Average"
        elif current_volume < volume_sma.iloc[-1] * 0.8:
            return "Below Average"
        elif current_volume < volume_sma.iloc[-1] * 0.5:
            return "Plunge"
        else:
            return "Average"

    def is_sideways(self, threshold_percent=0.03):
        """
        Checks if the market is sideways (ranging).

        Args:
          threshold_percent (float): Percentage threshold for price movement to be considered sideways.
        """
        if self.data.empty or len(self.data) < 20:
            return False

        # Calculate the range of price movement over a lookback period
        lookback_period = 20  # Example period.
        high_max = self.data['high'].iloc[-lookback_period:].max()
        low_min = self.data['low'].iloc[-lookback_period:].min()
        price_range = (high_max - low_min) / self.data['close'].iloc[-1]

        # Check if the price range is within the threshold
        if price_range < threshold_percent:
            return True
        else:
            return False

    def analyze(self):
        """Performs the complete market context analysis."""
        trend = self.analyze_trend()
        volatility = self.analyze_volatility()
        volume = self.analyze_volume()
        sideways = self.is_sideways()

        # Combine analysis results into a dictionary
        analysis_result = {
            'trend': trend,
            'volatility': volatility,
            'volume': volume,
            'sideways': sideways,
            'patterns': self.patterns.get_all_patterns(self.timeframe),  # Include detected patterns
            # Add on-chain and sentiment data if available
        }
        if self.onchain_data:
            analysis_result['onchain'] = self.onchain_data
        if self.sentiment_data:
            analysis_result['sentiment'] = self.sentiment_data

        return analysis_result
