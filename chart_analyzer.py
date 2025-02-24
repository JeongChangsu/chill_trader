from sklearn import linear_model
import numpy as np
import talib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chart_analyzer.log"), logging.StreamHandler()]
)


class ChartAnalyzer:
    @staticmethod
    def detect_trendlines(points):
        if len(points) < 10:  # 최소 포인트 수
            logging.warning("Insufficient points for trendline detection")
            return None
        x = np.array([p[0] for p in points]).reshape(-1, 1)
        y = np.array([p[1] for p in points])
        model = linear_model.RANSACRegressor(max_trials=1000, min_samples=5)
        model.fit(x, y)
        slope, intercept = model.estimator_.coef_[0], model.estimator_.intercept_
        logging.info(f"Trendline detected - Slope: {slope:.4f}, Intercept: {intercept:.2f}")
        return {'slope': slope, 'intercept': intercept}

    def analyze_charts(self, candle_data, indicators):
        analysis = {}
        for tf, data in candle_data.items():
            if len(data) < 52:
                logging.warning(f"[{tf}] Insufficient data for analysis")
                continue
            open_p = np.array([candle[1] for candle in data], dtype=float)
            high = np.array([candle[2] for candle in data], dtype=float)
            low = np.array([candle[3] for candle in data], dtype=float)
            close = np.array([candle[4] for candle in data], dtype=float)
            volume = np.array([candle[5] for candle in data], dtype=float)

            # 캔들 패턴 탐지
            patterns = {
                'Hammer': talib.CDLHAMMER(open_p, high, low, close)[-1],
                'Engulfing': talib.CDLENGULFING(open_p, high, low, close)[-1],
                'Doji': talib.CDLDOJI(open_p, high, low, close)[-1]
            }
            detected_patterns = {k: v for k, v in patterns.items() if v != 0}
            if detected_patterns:
                logging.info(f"[{tf}] Detected patterns: {detected_patterns}")

            # 거래량 기반 주요 지점 탐지
            avg_volume = np.mean(volume)
            significant_points = [
                (i, high[i] if high[i] > close[i] else low[i])
                for i in range(len(volume)) if volume[i] > avg_volume * 1.5
            ]

            analysis[tf] = {
                'patterns': detected_patterns,
                'trendlines': self.detect_trendlines(significant_points),
                'support_resistance': indicators[tf]['Volume_Profile']['price_levels'][:2].tolist()
            }
        return analysis
