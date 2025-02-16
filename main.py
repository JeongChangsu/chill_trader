import ccxt
import os

# 1. API 인증 정보 설정 (환경 변수 또는 직접 입력)
wallet_address = os.getenv('HYPE_ADDRESS')
private_key = os.getenv('HYPE_PRIVATE_KEY')

# 2. Hyperliquid 거래소 객체 생성
exchange = ccxt.hyperliquid({
    'privateKey': private_key,
    'walletAddress': wallet_address})

exchange.fetch_balance()