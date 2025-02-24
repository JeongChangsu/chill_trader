import talib
import numpy as np
import logging

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("indicator_calculator.log"), logging.StreamHandler()]
)


class IndicatorCalculator:
    @staticmethod
    def calculate_ichimoku(high, low, close):
        """정확한 Ichimoku Cloud 계산"""
        if len(high) < 52 or len(low) < 52 or len(close) < 26:
            return None
        tenkan = (np.max(high[-9:], axis=0) + np.min(low[-9:], axis=0)) / 2
        kijun = (np.max(high[-26:], axis=0) + np.min(low[-26:], axis=0)) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (np.max(high[-52:], axis=0) + np.min(low[-52:], axis=0)) / 2
        chikou = close[-26]
        logging.info(
            f"Ichimoku - Tenkan: {tenkan:.2f}, Kijun: {kijun:.2f}, Senkou A: {senkou_a:.2f}, Senkou B: {senkou_b:.2f}")
        return {'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a, 'senkou_b': senkou_b, 'chikou': chikou}

    @staticmethod
    def calculate_volume_profile(prices, volumes, bins=50):
        """Volume Profile: 가격 구간별 거래량 합산"""
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
        logging.info(f"Volume Profile - Top price level: {bin_edges[0]:.2f}, Volume: {hist[0]:.2f}")
        return {'price_levels': bin_edges[:-1], 'volume': hist}

    def calculate_indicators(self, candle_data):
        indicators = {}
        for tf, data in candle_data.items():
            if len(data) < 52:  # 최소 데이터 요구사항
                logging.warning(f"[{tf}] Insufficient data for indicators: {len(data)} candles")
                continue
            close = np.array([candle[4] for candle in data], dtype=float)
            high = np.array([candle[2] for candle in data], dtype=float)
            low = np.array([candle[3] for candle in data], dtype=float)
            volume = np.array([candle[5] for candle in data], dtype=float)

            indicators[tf] = {
                'MA5': talib.SMA(close, timeperiod=5)[-1],
                'MA20': talib.SMA(close, timeperiod=20)[-1],
                'RSI': talib.RSI(close, timeperiod=14)[-1],
                'Ichimoku': self.calculate_ichimoku(high, low, close),
                'Volume_Profile': self.calculate_volume_profile(close, volume)
            }
            logging.info(
                f"[{tf}] Indicators calculated - MA5: {indicators[tf]['MA5']:.2f}, RSI: {indicators[tf]['RSI']:.2f}")
        return indicators
