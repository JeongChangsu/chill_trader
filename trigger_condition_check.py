# trigger_condition_check.py

from technical_indicators import TechnicalIndicators
from market_context_analysis import MarketContextAnalysis


class TriggerConditions:
    def __init__(self, indicators, patterns, timeframe, market_context):
        """
        Initializes the TriggerConditions class.

        Args:
            indicators (TechnicalIndicators): TechnicalIndicators object.
            patterns (PatternRecognition): PatternRecognition object.
            timeframe (str): The timeframe for analysis (e.g., '1m', '1h').
            market_context (dict): The market context analysis result from MarketContextAnalysis.
        """
        self.indicators = indicators
        self.patterns = patterns
        self.timeframe = timeframe
        self.market_context = market_context  # Store the entire market context
        self.data = self.indicators.ohlcv_data[self.timeframe]

    def _get_trend_duration(self, threshold=20):
        """Calculates the duration of the current trend (in number of candles)."""
        # Uses ADX to estimate trend duration.  A more sophisticated approach
        # could use multiple indicators and timeframes.
        adx = self.indicators.calculate_adx(self.timeframe, 14)
        if adx is None or len(adx) < 2:  # adx가 none이거나, 길이가 2 미만이면
            return 0

        duration = 0
        for i in range(len(adx) - 1, -1, -1):  # Iterate backwards
            if adx.iloc[i] >= threshold:
                duration += 1
            else:
                break  # Stop when ADX falls below the threshold
        return duration

    def _get_sideways_duration(self, threshold_percent=0.03, min_duration=10):
        """
        시간이 흐름에 따라, 가격 변동을 계산하여, 횡보의 기간(몇개의 캔들)을 구한다.
        """
        if len(self.data) < min_duration:
            return 0

        duration = 0
        for i in range(len(self.data) - 1, len(self.data) - 1 - min_duration, -1):
            max_high = self.data['high'].iloc[i - min_duration + 1: i + 1].max()
            min_low = self.data['low'].iloc[i - min_duration + 1: i + 1].min()
            price_range = (max_high - min_low) / self.data['close'].iloc[i]
            if price_range < threshold_percent:
                duration += 1
            else:
                break
        return duration

    def check_long_entry(self):
        """Checks for long entry trigger conditions."""
        # --- Multi-Timeframe Analysis (Example: 1h and 4h) ---
        if self.timeframe != '1h':  # 메인 timeframe이 1시간이 아닐 경우, 리턴
            return False

        # 4시간봉 데이터 가져오기
        try:
            ohlcv_4h = self.indicators.ohlcv_data['4h']
        except:
            return False
        if ohlcv_4h.empty or len(ohlcv_4h) < 50:
            return False
        indicators_4h = TechnicalIndicators({'4h': ohlcv_4h})

        # 1시간봉 추세
        trend_1h = self.market_context['trend']
        # 4시간봉 추세
        trend_4h_analysis = MarketContextAnalysis(indicators_4h, None, '4h').analyze_trend()

        # 1. Strong Bullish Trend on Multiple Timeframes
        if not (trend_1h in ['Strong Bullish', "Weak Bullish"] and trend_4h_analysis in [
            'Strong Bullish', 'Weak Bullish']):
            return False

        # 2. Trend Duration
        trend_duration = self._get_trend_duration()
        if trend_duration < 3:  # Minimum trend duration (in candles)
            return False

        # 3. Volume Confirmation
        if self.market_context['volume'] not in ['Surge', 'Above Average']:
            return False

        # 4. Pattern Confirmation (Bullish Patterns)
        if not ('hammer' in self.market_context['patterns'] or
                'bullish_engulfing' in self.market_context['patterns'] or
                'morning_star' in self.market_context['patterns'] or
                'bullish_counterattack' in self.market_context['patterns'] or
                'piercing_pattern' in self.market_context['patterns'] or
                'three_white_soldiers' in self.market_context['patterns'] or
                'rising_three_methods' in self.market_context['patterns'] or
                'double_bottom' in self.market_context['patterns'] or
                'inverted_head_and_shoulders' in self.market_context['patterns']
        ):
            return False

        # 5. Volatility
        if self.market_context['volatility'] != 'High':
            return False

        # (Future) 6.  On-chain/Sentiment (Placeholder)
        # if self.market_context.get('onchain') and self.market_context['onchain'].get('whale_activity') != 'buying': # 예시
        #    return False
        # if self.market_context.get('sentiment') and self.market_context['sentiment'].get('fear_greed_index') < 40: # 예시
        #   return False
        return True

    def check_short_entry(self):
        """Checks for short entry trigger conditions."""
        # --- Multi-Timeframe Analysis (Example: 1h and 4h) ---
        if self.timeframe != '1h':  # 메인 timeframe이 1시간이 아닐 경우, 리턴
            return False

        # 4시간봉 데이터 가져오기
        try:
            ohlcv_4h = self.indicators.ohlcv_data['4h']
        except:
            return False
        if ohlcv_4h.empty or len(ohlcv_4h) < 50:
            return False
        indicators_4h = TechnicalIndicators({'4h': ohlcv_4h})

        # 1시간봉 추세
        trend_1h = self.market_context['trend']
        # 4시간봉 추세
        trend_4h_analysis = MarketContextAnalysis(indicators_4h, None, '4h').analyze_trend()

        # 1. Strong Bearish Trend on Multiple Timeframes
        if not (trend_1h in ["Strong Bearish", "Weak Bearish"] and trend_4h_analysis in [
            "Strong Bearish", "Weak Bearish"]):
            return False

        # 2. Trend Duration
        trend_duration = self._get_trend_duration()
        if trend_duration < 3:  # Minimum trend duration (in candles)
            return False

        # 3. Volume Confirmation
        if self.market_context['volume'] not in ['Surge', 'Above Average']:
            return False

        # 4. Pattern Confirmation (Bearish Patterns)
        if not ('hanging_man' in self.market_context['patterns'] or
                'bearish_engulfing' in self.market_context['patterns'] or
                'evening_star' in self.market_context['patterns'] or
                'bearish_counterattack' in self.market_context['patterns'] or
                'dark_cloud_cover' in self.market_context['patterns'] or
                'three_black_crows' in self.market_context['patterns'] or
                'falling_three_methods' in self.market_context['patterns'] or
                'double_top' in self.market_context['patterns'] or
                'head_and_shoulders' in self.market_context['patterns']
        ):
            return False

        # 5. Volatility
        if self.market_context['volatility'] != 'High':
            return False

        # (Future) 6. On-chain/Sentiment
        # ...

        return True

    def check_sideways_breakout(self):
        """
        Checks for sideways breakout trigger conditions, including refined logic
        for triangle, wedge, and flag/pennant breakouts.
        """
        if not self.market_context['sideways']:
            return None

        sideways_duration = self._get_sideways_duration()
        if sideways_duration < 10:  # Minimum sideways duration (in candles)
            return None

        # Prioritize triangle breakouts, then wedges, then flags/pennants
        if 'triangle' in self.market_context['patterns']:
            if self.market_context['patterns']['triangle']['breakout'] == 'bullish':
                return "sideways_breakout_long"  # Bullish Triangle Breakout
            elif self.market_context['patterns']['triangle']['breakout'] == 'bearish':
                return "sideways_breakout_short"  # Bearish Triangle Breakout

        if 'wedge' in self.market_context['patterns']:
            if self.market_context['patterns']['wedge']['breakout'] == 'bullish':
                return "sideways_breakout_long"  # Bullish Wedge Breakout
            elif self.market_context['patterns']['wedge']['breakout'] == 'bearish':
                return "sideways_breakout_short"  # Bearish Wedge Breakout

        if 'flag_and_pennant' in self.market_context['patterns']:
            if self.market_context['patterns']['flag_and_pennant']['breakout'] == 'bullish':
                return "sideways_breakout_long"
            elif self.market_context['patterns']['flag_and_pennant']['breakout'] == 'bearish':
                return "sideways_breakout_short"

        return None  # No breakout detected

    def check_conditions(self):
        if self.check_long_entry():
            return "long_entry"
        elif self.check_short_entry():
            return "short_entry"
        elif self.check_sideways_breakout():
            return self.check_sideways_breakout()  # Returns "sideways_breakout_long" or "sideways_breakout_short"
        else:
            return None
