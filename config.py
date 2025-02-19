# config.py
import os

# 상수
SYMBOL = "BTC/USDT"

# API 키 (환경 변수에서 가져오기)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')