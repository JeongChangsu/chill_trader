import talib
import numpy as np
import pandas as pd
import logging


def calculate_ichimoku(df):
    try:
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)
        logging.info("Ichimoku Cloud 계산 완료")
        return df
    except Exception as e:
        logging.error(f"Ichimoku 계산 오류: {e}")
        return df


def calculate_volume_profile(df, bins=50):
    try:
        price_range = np.linspace(df['low'].min(), df['high'].max(), bins)
        volume_profile = [{'price_level': price_range[i], 'volume': df['volume'][(df['close'] >= price_range[i]) &
                                                                                 (df['close'] < price_range[
                                                                                     i + 1])].sum()}
                          for i in range(len(price_range) - 1)]
        logging.info("Volume Profile 계산 완료")
        return pd.DataFrame(volume_profile)
    except Exception as e:
        logging.error(f"Volume Profile 계산 오류: {e}")
        return pd.DataFrame()


def calculate_indicators(candle_data):
    indicators = {}
    for tf, df in candle_data.items():
        if df.empty:
            logging.warning(f"{tf} 데이터가 비어 있음")
            continue
        df = calculate_ichimoku(df)
        df['MA5'] = talib.SMA(df['close'], timeperiod=5)
        df['MA20'] = talib.SMA(df['close'], timeperiod=20)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
        indicators[tf] = {
            'Ichimoku': df[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']],
            'MA5': df['MA5'], 'MA20': df['MA20'], 'RSI': df['RSI'], 'MACD': df['MACD'],
            'Volume_Profile': calculate_volume_profile(df)
        }
        logging.info(f"{tf} 지표 계산 완료: RSI={df['RSI'].iloc[-1]:.2f}, MACD={df['MACD'].iloc[-1]:.2f}")
    return indicators
