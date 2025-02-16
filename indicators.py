# indicators.py
import ta
import pandas_ta

import pandas as pd


def calculate_technical_indicators(df):
    """
    입력받은 DataFrame에 기술적 지표를 계산하여 추가합니다.
    """
    # 1. 추세 지표 (EMA, MACD, ADX, DMI, Aroon)
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # ADX, DMI (추세 강도)
    adx = df.ta.adx(length=14)  # pandas_ta 사용, DataFrame 반환
    df['adx'] = adx['ADX_14']
    df['plus_di'] = adx['DMP_14']  # +DI
    df['minus_di'] = adx['DMN_14']  # -DI

    # Aroon (추세 방향 및 강도)
    aroon = df.ta.aroon(length=25)  # pandas_ta 사용, DataFrame 반환
    df['aroon_up'] = aroon['AROONU_25']
    df['aroon_down'] = aroon['AROOND_25']

    # 2. 모멘텀 지표 (RSI)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # 3. 변동성 지표 (ATR, Bollinger Bands)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()

    # 4. 캔들 패턴
    patterns_df = df.ta.cdl_pattern(name="all")  # 모든 캔들 패턴 계산

    # 캔들 패턴 컬럼명에 맞춰서 boolean 값 추출
    df['engulfing_bullish'] = patterns_df['CDL_ENGULFING'] > 0  # Bullish Engulfing (양수: Bullish)
    df['engulfing_bearish'] = patterns_df['CDL_ENGULFING'] < 0  # Bearish Engulfing (음수: Bearish)
    df['morning_star'] = patterns_df['CDL_MORNINGSTAR'] != 0  # Morning Star (0이 아니면 패턴)
    df['evening_star'] = patterns_df['CDL_EVENINGSTAR'] != 0  # Evening Star (0이 아니면 패턴)
    df['hammer'] = patterns_df['CDL_HAMMER'] != 0  # Hammer (0이 아니면 패턴)
    df['hanging_man'] = patterns_df['CDL_HANGINGMAN'] != 0  # Hanging Man (0이 아니면 패턴)
    df['doji'] = patterns_df['CDL_DOJI_10_0.1'] != 0  # Doji (0이 아니면 패턴, CDL_DOJI_10_0.1 컬럼 사용)

    # 5. 거래량 분석 (거래량 변화율, 거래량-가격 다이버전스, Volume Oscillator 추가) - ta library
    df['volume_change'] = df['volume'].pct_change() * 100  # Volume Change (%)

    vo_df = df.ta.kvo()  # KVO returns DataFrame
    df['volume_oscillator'] = vo_df['KVO_34_55_13']  # Select 'KVO_34_55_13' column

    # 가격 - EMA 차이 (percent diff) - ta library
    df['ema50_diff'] = (df['close'] - df['ema50']) / df['ema50'] * 100
    df['ema200_diff'] = (df['close'] - df['ema200']) / df['ema200'] * 100

    return df


def calculate_volume_divergence(df, period=20):
    """
    거래량-가격 다이버전스 (Volume-Price Divergence) 를 분석하는 함수
    """
    # 1. 가격 상승 + 거래량 감소  => Bearish Divergence (weakening uptrend)
    price_up_volume_down = (df['close'].diff() > 0) & (df['volume'].diff() < 0)

    # 2. 가격 하락 + 거래량 감소  => Bullish Divergence (weakening downtrend)
    price_down_volume_down = (df['close'].diff() < 0) & (df['volume'].diff() < 0)

    df['bearish_divergence'] = price_up_volume_down.rolling(window=period).sum()
    df['bullish_divergence'] = price_down_volume_down.rolling(window=period).sum()
    return df


def calculate_donchian_channel(df, window=20):
    """
    Donchian Channels를 계산하여 박스권 상단/하단을 반환합니다.
    """
    df['donchian_upper'] = df['high'].rolling(window=window).max()
    df['donchian_lower'] = df['low'].rolling(window=window).min()
    df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
    return df
