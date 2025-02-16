# main.py
import os
import ccxt
import asyncio
import logging

from google import genai
from utils import setup_logging, log_decision
from gemini import generate_trading_decision, parse_trading_decision
from telegram_utils import send_telegram_message, escape_markdown_v2
from config import SYMBOL, HYPE_SYMBOL, TIMEFRAMES, DECISIONS_LOG_FILE, GOOGLE_API_KEY
from data_fetcher import (fetch_multi_tf_data, gather_additional_data,
                          fetch_economic_data, parse_economic_data, fetch_liquidation_map)
from strategy import determine_market_regime, adjust_indicator_thresholds, select_strategy, get_current_session_kst
from hyperliquid_utils import (get_hyperliquid_position, get_hyperliquid_balance,
                               create_hyperliquid_order, close_expired_positions)


async def main():
    setup_logging()
    logging.info("Trading bot started.")

    # Hyperliquid 거래소 객체 생성
    exchange = ccxt.hyperliquid({
        'walletAddress': os.environ.get('HYPE_ADDRESS'),
        'privateKey': os.environ.get('HYPE_PRIVATE_KEY'),
        'options': {'defaultType': 'swap', },
    })

    # Gemini 클라이언트 초기화
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

    # 1. 포지션 종료 (거래 기간 만료)
    close_expired_positions(exchange, HYPE_SYMBOL)

    # 2. 초기 포지션 확인
    position = get_hyperliquid_position(exchange, HYPE_SYMBOL)
    if position:
        return

    # 3. 잔고 확인
    balance = get_hyperliquid_balance(exchange)
    logging.info(f"Current balance: {balance:.2f} USDC")

    try:
        # 4. 비동기 데이터 수집
        multi_tf_data, additional_data = await asyncio.gather(
            fetch_multi_tf_data(exchange, HYPE_SYMBOL, TIMEFRAMES, limit=300),
            gather_additional_data(exchange, HYPE_SYMBOL)
        )

        if not multi_tf_data:
            logging.error("Failed to fetch multi-timeframe data. Retrying in next cycle.")
            return

        # 경제 지표 데이터
        econ_data_raw = fetch_economic_data()
        econ_summary = parse_economic_data(econ_data_raw) if econ_data_raw else "No economic data available."

        # 5. 시장 상황(장세) 결정
        market_regime = determine_market_regime(multi_tf_data)
        thresholds = adjust_indicator_thresholds(market_regime)

        # 6. 전략 선택
        strategy = select_strategy(market_regime)
        if not strategy:
            logging.info(f"No suitable strategy found for market regime: {market_regime}")
            return

        # 7. 현재 세션 확인
        current_session = get_current_session_kst()

        # 8. Gemini Pro를 이용한 최종 거래 결정
        fetch_liquidation_map()  # 청산맵 다운로드 (동기 함수)
        gemini_response_text = generate_trading_decision(
            gemini_client, multi_tf_data, market_regime, strategy, thresholds, current_session, additional_data,
            econ_summary
        )
        logging.info(f"Gemini Raw Response: {gemini_response_text}")
        decision = parse_trading_decision(gemini_response_text)
        log_decision(decision, SYMBOL, DECISIONS_LOG_FILE)  # 로깅

        # 9. 거래 실행
        if decision['final_action'].upper() == 'NO TRADE':
            logging.info("No Trade")
            send_telegram_message(f"*거래 없음 (NO TRADE)*\n\n*이유:* {escape_markdown_v2(decision['rationale'])}")
        elif decision['final_action'] in ['GO LONG', 'GO SHORT']:
            order = create_hyperliquid_order(exchange, HYPE_SYMBOL, decision,
                                             float(decision['leverage'].replace('x', '')))
            if order:
                current_side = decision['final_action'].split()[-1]
                entry_price = float(order[0]['price'])
                side_emoji = "🟢 매수" if current_side == "LONG" else "🔴 매도"
                message = (
                    f"*{side_emoji} 포지션 진입* ({SYMBOL})\n\n"
                    f"*레버리지:* {decision['leverage']}\n"
                    f"*기간:* {decision['trade_term']}\n"
                    f"*진입 가격:* {entry_price:.2f}\n"
                    f"*목표 가격 (TP):* {decision['tp_price']}\n"
                    f"*손절 가격 (SL):* {decision['sl_price']}\n"
                    f"*거래량(USDC):* {decision['amount_usd']}\n\n"
                    f"*분석:* {escape_markdown_v2(decision['rationale'])}"
                )
                send_telegram_message(message)
            else:
                send_telegram_message(f"*거래 실패* ({SYMBOL})\n\n*이유:* {escape_markdown_v2(decision['rationale'])}")
        else:
            logging.warning(f"Invalid decision from Gemini: {decision['final_action']}")

    except Exception as e:
        logging.error(f"An error occurred in the main loop: {e}")


if __name__ == "__main__":
    asyncio.run(main())
