# config.py
import os
import pytz

# 상수
SYMBOL = "BTC/USDT"
HYPE_SYMBOL = "BTC/USDC:USDC"
DECISIONS_LOG_FILE = "trading_decisions.csv"
CLOSED_POSITIONS_FILE = "closed_positions.csv"
TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

# API 키 (환경 변수에서 가져오기)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
HYPERLIQUID_WALLET_ADDRESS = os.environ.get('HYPE_ADDRESS')
HYPERLIQUID_PRIVATE_KEY = os.environ.get('HYPE_PRIVATE_KEY')

# 시간대
KST = pytz.timezone("Asia/Seoul")

# 세션 정보
SESSION_GUIDES = {
    "OVERNIGHT": "Low liquidity. Be cautious of fake breakouts. Use tighter stops.",
    "ASIAN": "08:00-09:00 KST: Potential volatility spike. After 09:00: Trend may develop.",
    "LONDON": "16:00 KST open: Expect high volatility. Trade the dominant trend.",
    "US": "Highest volume and volatility. Be prepared for reversals.",
    "TRANSITION": "Low liquidity, potential trend formation before Asian open.",
    "ASIAN_WEEKEND": "Lower liquidity, increased volatility. Be cautious.",
    "LONDON_WEEKEND": "Lower liquidity, increased volatility. Be cautious.",
    "US_WEEKEND": "Lower liquidity, increased volatility. Be cautious. Watch for sudden price swings.",
    "US_US_HOLIDAY": "US market closed. Expect lower liquidity and potentially erratic price movements.",
}
