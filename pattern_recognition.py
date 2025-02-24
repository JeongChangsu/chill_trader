# pattern_recognition.py

import talib
import numpy as np
import pandas as pd


class PatternRecognition:
    def __init__(self, ohlcv_data):
        """
        Initializes the PatternRecognition class with OHLCV data.

        Args:
            ohlcv_data (dict): A dictionary where keys are timeframes (e.g., '1m', '5m', '1h')
                               and values are Pandas DataFrames containing OHLCV data.
        """
        self.ohlcv_data = ohlcv_data

    def recognize_candle_patterns(self, timeframe):
        """Recognizes candlestick patterns using TA-Lib."""
        df = self.ohlcv_data[timeframe]
        if df.empty or len(df) < 100:  # Ensure enough data for calculations
            return {}

        patterns = {
            'hammer': talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']),
            'inverted_hammer': talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close']),
            'hanging_man': talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close']),
            'shooting_star': talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']),
            'bullish_engulfing': talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']),
            'bearish_engulfing': talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']),
            'bullish_belt_hold': talib.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close']),  # Bullish only
            'bearish_belt_hold': talib.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close']),  # Bearish only
            'doji': talib.CDLDOJI(df['open'], df['high'], df['low'], df['close']),
            'spinning_top': talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close']),
            'morning_star': talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close']),
            'evening_star': talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close']),
            'bullish_counterattack': talib.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close']),  # Bullish
            'bearish_counterattack': talib.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close']),  # Bearish
            'piercing_pattern': talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close']),
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close']),
            'three_white_soldiers': talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close']),
            'three_black_crows': talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close']),
            'rising_three_methods': talib.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close']),
            # Bullish
            'falling_three_methods': talib.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close']),
            # Bearish
        }

        # Filter and categorize patterns based on their values (100: bullish, -100: bearish, 0: no pattern)
        filtered_patterns = {}
        for pattern_name, pattern_values in patterns.items():
            # Bullish patterns
            if pattern_name in ['hammer', 'inverted_hammer', 'bullish_engulfing', 'bullish_belt_hold',
                                'morning_star', 'bullish_counterattack', 'piercing_pattern',
                                'three_white_soldiers', 'rising_three_methods']:
                if pattern_values.iloc[-1] > 0:  # Check only the last value
                    filtered_patterns[pattern_name] = "bullish"  # Signal
            # Bearish patterns
            elif pattern_name in ['hanging_man', 'shooting_star', 'bearish_engulfing', 'bearish_belt_hold',
                                  'evening_star', 'bearish_counterattack', 'dark_cloud_cover',
                                  'three_black_crows', 'falling_three_methods']:
                if pattern_values.iloc[-1] < 0:  # Check only the last value
                    filtered_patterns[pattern_name] = "bearish"  # Signal
            # Neutral patterns (Doji, Spinning Top) - Only check for non-zero (presence)
            elif pattern_name in ['doji', 'spinning_top']:
                if pattern_values.iloc[-1] != 0:  # Check only the last value
                    filtered_patterns[pattern_name] = 'neutral'

        return filtered_patterns

    def detect_double_top(self, timeframe, threshold_percent=0.01):
        """Detects Double Top chart pattern."""

        df = self.ohlcv_data[timeframe].copy()
        if df.empty or len(df) < 50:
            return None

        # 1. Find local maxima
        df['local_max'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        maxima_indices = df[df['local_max']].index

        if len(maxima_indices) < 2:
            return None

        # 2. Get the last two maxima
        last_max_index = maxima_indices[-1]
        second_last_max_index = maxima_indices[-2]

        # 3. Find the lowest point between the two maxima
        min_between = df.loc[second_last_max_index:last_max_index]['low'].min()

        # 4. Check if the two maxima are within the threshold
        max1 = df.loc[second_last_max_index]['high']
        max2 = df.loc[last_max_index]['high']

        if abs(max1 - max2) / ((max1 + max2) / 2) < threshold_percent:  # Check if two peaks' prices are similar
            # 5. Check if the second max is lower than the first (optional, for stricter pattern)
            if max2 < max1 * (1 + threshold_percent):
                # 6. Confirmation: Check if the price breaks below the minimum point between the two maxima
                if df['close'].iloc[-1] < min_between:
                    return {
                        'type': 'double_top',
                        'first_top': second_last_max_index,
                        'second_top': last_max_index,
                        'confirmation': df.index[-1],
                        'strength': abs(max1 - max2) / ((max1 + max2) / 2)  # Example: Strength based on the difference
                    }
        return None

    def detect_double_bottom(self, timeframe, threshold_percent=0.01):
        """Detects Double Bottom chart pattern."""
        df = self.ohlcv_data[timeframe].copy()
        if df.empty or len(df) < 50:
            return None

        # 1. Find local minima
        df['local_min'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        minima_indices = df[df['local_min']].index

        if len(minima_indices) < 2:
            return None

        # 2. Get the last two minima
        last_min_index = minima_indices[-1]
        second_last_min_index = minima_indices[-2]

        # 3. Find the highest point between the two minima
        max_between = df.loc[second_last_min_index:last_min_index]['high'].max()

        # 4. Check if the two minima are within the threshold
        min1 = df.loc[second_last_min_index]['low']
        min2 = df.loc[last_min_index]['low']

        if abs(min1 - min2) / ((min1 + min2) / 2) < threshold_percent:
            if min2 > min1 * (1 - threshold_percent):
                # 5. Confirmation: Check if the price breaks above the maximum point between the two minima
                if df['close'].iloc[-1] > max_between:
                    return {
                        'type': 'double_bottom',
                        'first_bottom': second_last_min_index,
                        'second_bottom': last_min_index,
                        'confirmation': df.index[-1],
                        'strength': abs(min1 - min2) / ((min1 + min2) / 2)  # Strength based on similarity of bottoms

                    }
        return None

    def detect_head_and_shoulders(self, timeframe, threshold_percent=0.03):
        """
        Detects Head and Shoulders chart pattern (Inverted H&S detection is separate).

        This is a simplified example, and real-world H&S detection is more complex.
        Consider using a machine learning approach for more robust detection.
        """
        df = self.ohlcv_data[timeframe].copy()

        if df.empty or len(df) < 100:  # Need enough data
            return None

        # 1. Find local maxima and minima
        df['local_max'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['local_min'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        maxima_indices = df[df['local_max']].index
        minima_indices = df[df['local_min']].index

        if len(maxima_indices) < 3 or len(minima_indices) < 2:  # Need at least 3 peaks and 2 troughs
            return None

        # 2. Check for potential Head and Shoulders formation (simplified)
        # Look for the last 3 maxima and 2 minima.

        # Note: The following logic is an extreme simplification for demonstration.
        # A real implementation would need more sophisticated checks and validations.

        maxima = df.loc[maxima_indices[-3:]]['high'].values  # Last three peaks
        minima = df.loc[minima_indices[-2:]]['low'].values  # Last two troughs

        # Check for Head and Shoulders (peak 2 is the highest)
        if maxima[1] > maxima[0] * (1 + threshold_percent) and maxima[1] > maxima[2] * (1 + threshold_percent) and \
                minima[0] < maxima[0] * (1 + threshold_percent) and minima[1] < maxima[2] * (1 + threshold_percent) and \
                abs(maxima[0] - maxima[2]) / ((maxima[0] + maxima[2]) / 2) < threshold_percent:

            neckline_slope = (minima[1] - minima[0]) / (minima_indices[-1].timestamp() - minima_indices[-2].timestamp())
            neckline_intercept = minima[1] - neckline_slope * minima_indices[-1].timestamp()

            # Confirmation: Check if current price crosses the neckline.
            neckline_price_at_current = neckline_slope * df.index[-1].timestamp() + neckline_intercept

            if df['close'].iloc[-1] < neckline_price_at_current:
                return {
                    'type': 'head_and_shoulders',
                    'left_shoulder': maxima_indices[-3],
                    'head': maxima_indices[-2],
                    'right_shoulder': maxima_indices[-1],
                    'neckline_break': df.index[-1],
                    'strength': (maxima[1] - (maxima[0] + maxima[2]) / 2) / maxima[1]
                }
        return None

    def detect_inverted_head_and_shoulders(self, timeframe, threshold_percent=0.03):
        """Detects Inverted Head and Shoulders chart pattern."""
        df = self.ohlcv_data[timeframe].copy()

        if df.empty or len(df) < 100:  # Need enough data
            return None

        # 1. Find local maxima and minima
        df['local_max'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['local_min'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        maxima_indices = df[df['local_max']].index
        minima_indices = df[df['local_min']].index

        if len(maxima_indices) < 2 or len(minima_indices) < 3:  # Need at least 2 peaks and 3 troughs
            return None

        # Check for Inverted Head and Shoulders (trough 2 is the lowest)
        minima = df.loc[minima_indices[-3:]]['low'].values
        maxima = df.loc[maxima_indices[-2:]]['high'].values

        if minima[1] < minima[0] * (1 - threshold_percent) and minima[1] < minima[2] * (1 - threshold_percent) and \
                maxima[0] > minima[0] * (1 - threshold_percent) and maxima[1] > minima[2] * (1 - threshold_percent) and \
                abs(minima[0] - minima[2]) / ((minima[0] + minima[2]) / 2) < threshold_percent:

            neckline_slope = (maxima[1] - maxima[0]) / (maxima_indices[-1].timestamp() - maxima_indices[-2].timestamp())
            neckline_intercept = maxima[1] - neckline_slope * maxima_indices[-1].timestamp()

            # Confirmation: Check if the current price crosses the neckline (upward breakout).
            neckline_price_at_current = neckline_slope * df.index[-1].timestamp() + neckline_intercept

            if df['close'].iloc[-1] > neckline_price_at_current:
                return {
                    'type': 'inverted_head_and_shoulders',
                    'left_shoulder': minima_indices[-3],
                    'head': minima_indices[-2],
                    'right_shoulder': minima_indices[-1],
                    'neckline_break': df.index[-1],
                    'strength': ((minima[0] + minima[2]) / 2 - minima[1]) / minima[1]  # Strength based on head depth
                }
        return None

    def detect_triangle(self, timeframe, threshold_percent=0.03):
        """
        Detects Triangle chart patterns (Symmetrical, Ascending, Descending).
        This is a simplified version. Real-world triangle detection requires more complex logic.
        """
        df = self.ohlcv_data[timeframe].copy()
        if df.empty or len(df) < 50:
            return None

        # 1. Find significant highs and lows (using rolling window)
        window_size = 10  # Adjust this parameter as needed.
        df['roll_max'] = df['high'].rolling(window=window_size, center=True).max()
        df['roll_min'] = df['low'].rolling(window=window_size, center=True).min()

        # Find points where the price touches the rolling max/min (potential pivot points)
        df['high_pivot'] = (df['high'] >= df['roll_max'])
        df['low_pivot'] = (df['low'] <= df['roll_min'])

        high_pivots = df[df['high_pivot']].index
        low_pivots = df[df['low_pivot']].index

        if len(high_pivots) < 2 or len(low_pivots) < 2:
            return None

        # 2. Fit trendlines to the last few high and low pivots
        #    (Simplified: using only the last 2 points for each trendline)

        # Upper trendline (connecting highs)
        high_slope = (df.loc[high_pivots[-1]]['high'] - df.loc[high_pivots[-2]]['high']) / (
                high_pivots[-1].timestamp() - high_pivots[-2].timestamp())
        high_intercept = df.loc[high_pivots[-1]]['high'] - high_slope * high_pivots[-1].timestamp()

        # Lower trendline (connecting lows)
        low_slope = (df.loc[low_pivots[-1]]['low'] - df.loc[low_pivots[-2]]['low']) / (
                low_pivots[-1].timestamp() - low_pivots[-2].timestamp())
        low_intercept = df.loc[low_pivots[-1]]['low'] - low_slope * low_pivots[-1].timestamp()

        # 3. Classify triangle type based on trendline slopes
        triangle_type = None
        if abs(high_slope) < threshold_percent and low_slope > threshold_percent:  # Adjust thresholds as needed
            triangle_type = 'ascending'
        elif abs(low_slope) < threshold_percent and high_slope < -threshold_percent:
            triangle_type = 'descending'
        elif high_slope < -threshold_percent and low_slope > threshold_percent:
            triangle_type = 'symmetrical'

        if triangle_type:
            # 4. Check for breakout (confirmation)
            upper_trendline_value = high_slope * df.index[-1].timestamp() + high_intercept
            lower_trendline_value = low_slope * df.index[-1].timestamp() + low_intercept

            if df['close'].iloc[-1] > upper_trendline_value:
                breakout = 'bullish'
            elif df['close'].iloc[-1] < lower_trendline_value:
                breakout = 'bearish'
            else:
                breakout = None

            if breakout:
                return {
                    'type': f'{triangle_type}_triangle',
                    'upper_trendline': (high_pivots[-2], high_pivots[-1]),  # Return the last two pivot points
                    'lower_trendline': (low_pivots[-2], low_pivots[-1]),
                    'breakout': breakout,  # 'bullish' or 'bearish'
                    'breakout_time': df.index[-1]
                }
        return None

    def detect_wedge(self, timeframe, threshold_percent=0.02):
        """
        Detects Rising and Falling Wedge patterns.
        This is a simplified implementation and should be refined further for real-world use.
        """
        df = self.ohlcv_data[timeframe].copy()
        if df.empty or len(df) < 50:
            return None

        # 1. Find significant highs and lows
        window_size = 10
        df['roll_max'] = df['high'].rolling(window=window_size, center=True).max()
        df['roll_min'] = df['low'].rolling(window=window_size, center=True).min()
        df['high_pivot'] = (df['high'] >= df['roll_max'])
        df['low_pivot'] = (df['low'] <= df['roll_min'])
        high_pivots = df[df['high_pivot']].index
        low_pivots = df[df['low_pivot']].index

        if len(high_pivots) < 2 or len(low_pivots) < 2:
            return None

        # 2. Fit trendlines
        high_slope = (df.loc[high_pivots[-1]]['high'] - df.loc[high_pivots[-2]]['high']) / (
                high_pivots[-1].timestamp() - high_pivots[-2].timestamp())
        high_intercept = df.loc[high_pivots[-1]]['high'] - high_slope * high_pivots[-1].timestamp()
        low_slope = (df.loc[low_pivots[-1]]['low'] - df.loc[low_pivots[-2]]['low']) / (
                low_pivots[-1].timestamp() - low_pivots[-2].timestamp())
        low_intercept = df.loc[low_pivots[-1]]['low'] - low_slope * low_pivots[-1].timestamp()

        # 3. Check for converging trendlines (wedge)
        if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
            wedge_type = 'falling'  # Falling Wedge (Bullish)
        elif high_slope < 0 and low_slope < 0 and high_slope < low_slope:
            wedge_type = 'rising'  # Rising Wedge (Bearish)
        else:
            return None

        # 4. Check for breakout
        upper_trendline_value = high_slope * df.index[-1].timestamp() + high_intercept
        lower_trendline_value = low_slope * df.index[-1].timestamp() + low_intercept

        breakout = None
        if wedge_type == 'falling':
            if df['close'].iloc[-1] > upper_trendline_value:
                breakout = 'bullish'
        elif wedge_type == 'rising':
            if df['close'].iloc[-1] < lower_trendline_value:
                breakout = 'bearish'
        if breakout:
            return {
                'type': f'{wedge_type}_wedge',
                'upper_trendline': (high_pivots[-2], high_pivots[-1]),  # Return last two pivot points
                'lower_trendline': (low_pivots[-2], low_pivots[-1]),
                'breakout': breakout,
                'breakout_time': df.index[-1]
            }

        return None

    def detect_flag_and_pennant(self, timeframe, threshold_percent=0.05):
        """Detects Flag and Pennant patterns (Bullish and Bearish)."""

        # Flags and Pennants are short-term continuation patterns.
        # This is a simplified version; real-world detection is more complex
        df = self.ohlcv_data[timeframe].copy()

        if df.empty or len(df) < 30:
            return None

        # 1. Identify a strong trend (pole)
        #    (Simplified: using a significant price change over a short period)
        pole_length = 10  # Length of the "pole"
        price_change = df['close'].iloc[-1] - df['close'].iloc[-pole_length]

        if abs(price_change) / df['close'].iloc[-pole_length] < threshold_percent:
            return None  # Not a strong enough trend

        trend = "bullish" if price_change > 0 else "bearish"

        # 2. Look for consolidation (flag or pennant) after the trend
        consolidation_length = 5  # Minimum length for consolidation period

        if len(df) < pole_length + consolidation_length:
            return None

        consolidation_data = df.iloc[-consolidation_length:]

        high_max = consolidation_data['high'].max()
        high_min = consolidation_data['high'].min()
        low_max = consolidation_data['low'].max()
        low_min = consolidation_data['low'].min()

        # Check if consolidation range is relatively tight
        consolidation_range_high = (high_max - high_min) / high_min
        consolidation_range_low = (low_max - low_min) / low_min

        if consolidation_range_high > threshold_percent * 2 or consolidation_range_low > threshold_percent * 2:
            return None

        # 3. Determine Flag or Pennant (simplified)
        # Flag:  Highs and lows move roughly parallel (within a channel)
        # Pennant: Highs and lows converge (form a small triangle)

        high_slope = (consolidation_data['high'].iloc[-1] - consolidation_data['high'].iloc[0]) / consolidation_length
        low_slope = (consolidation_data['low'].iloc[-1] - consolidation_data['low'].iloc[0]) / consolidation_length

        pattern_type = None
        if abs(high_slope - low_slope) < 0.005:
            pattern_type = "flag"  # Nearly parallel
        elif (high_slope < 0 and low_slope > 0):
            pattern_type = "pennant"
        else:
            return None

        # 4. Check for breakout
        breakout = None
        if trend == "bullish":
            if df['close'].iloc[-1] > high_max:
                breakout = 'bullish'
        elif trend == "bearish":
            if df['close'].iloc[-1] < low_min:
                breakout = 'bearish'

        if breakout:
            return {
                'type': f'{trend}_{pattern_type}',
                'pole_start': df.index[-pole_length - 1],
                'consolidation_start': df.index[-consolidation_length],
                'breakout': breakout,
                'breakout_time': df.index[-1]
            }

        return None

    def get_all_patterns(self, timeframe):
        """
        Detects all supported patterns for a given timeframe.
        Returns:
            dict: A dictionary containing the detected patterns.
        """

        all_patterns = {}
        candle_patterns = self.recognize_candle_patterns(timeframe)

        if candle_patterns:
            all_patterns.update(candle_patterns)

        # Chart patterns
        double_top = self.detect_double_top(timeframe)
        if double_top:
            all_patterns['double_top'] = double_top

        double_bottom = self.detect_double_bottom(timeframe)
        if double_bottom:
            all_patterns['double_bottom'] = double_bottom

        head_and_shoulders = self.detect_head_and_shoulders(timeframe)
        if head_and_shoulders:
            all_patterns['head_and_shoulders'] = head_and_shoulders

        inverted_head_and_shoulders = self.detect_inverted_head_and_shoulders(timeframe)
        if inverted_head_and_shoulders:
            all_patterns['inverted_head_and_shoulders'] = inverted_head_and_shoulders

        triangle = self.detect_triangle(timeframe)
        if triangle:
            all_patterns['triangle'] = triangle

        wedge = self.detect_wedge(timeframe)
        if wedge:
            all_patterns['wedge'] = wedge

        flag_pennant = self.detect_flag_and_pennant(timeframe)
        if flag_pennant:
            all_patterns['flag_and_pennant'] = flag_pennant

        return all_patterns
