import talib
import numpy as np
from sklearn.linear_model import RANSACRegressor
import logging


def detect_candle_patterns(df):
    patterns = {
        'Hammer': talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']),
        'Engulfing': talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']),
        'Doji': talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    }
    detected = {k: v.iloc[-1] for k, v in patterns.items() if v.iloc[-1] != 0}
    if detected:
        logging.info(f"캔들 패턴 탐지: {detected}")
    return detected


def detect_trendlines(df):
    highs = df['high'][df['high'].rolling(window=5, center=True).max() == df['high']]
    lows = df['low'][df['low'].rolling(window=5, center=True).min() == df['low']]
    if len(lows) > 5:
        X = np.array(lows.index).reshape(-1, 1)
        y = lows.values
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        slope, intercept = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
        logging.info(f"추세선 탐지: 기울기={slope:.2f}, 절편={intercept:.2f}")
        return {'slope': slope, 'intercept': intercept}
    return None


def analyze_charts(candle_data):
    analysis = {}
    for tf, df in candle_data.items():
        if df.empty:
            continue
        analysis[tf] = {
            'patterns': detect_candle_patterns(df),
            'trendline': detect_trendlines(df)
        }
    return analysis
