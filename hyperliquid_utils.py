# hyperliquid_utils.py
import re
import pytz
import logging

import pandas as pd

from config import KST
from datetime import datetime, timedelta


def create_hyperliquid_order(exchange, symbol, decision, leverage):
    """Hyperliquid 거래소에 지정가 주문 생성 (TP/SL 포함)"""
    try:
        order_type = 'limit'
        side = 'buy' if decision['final_action'] == 'GO LONG' else 'sell'
        amount = round(float(decision['amount']), 5)
        price = float(decision['limit_order_price'])  # 지정가

        # TP/SL 가격 (문자열 -> 숫자)
        tp_price = float(decision['tp_price'])
        sl_price = float(decision['sl_price'])

        exchange.set_margin_mode('isolated', symbol, params={'leverage': leverage})

        # Hyperliquid는 여러 개의 주문을 하나의 list로 받는다.
        orders = [
            {  # 1. 지정가 매수 주문
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price,
            },
            {  # 2. Take Profit (TP) 주문
                'symbol': symbol,
                'type': order_type,  # 또는 'limit' (Hyperliquid API 문서 확인 필요)
                'side': 'sell',
                'amount': amount,
                'price': tp_price,
                'params': {'reduceOnly': True, 'triggerPrice': tp_price, 'stopPrice': tp_price,
                           'takeProfitPrice': tp_price},  # triggerPrice, stopPrice, takeProfitPrice 모두 시도
            },
            {  # 3. Stop Loss (SL) 주문
                'symbol': symbol,
                'type': order_type,  # 또는 'limit' (Hyperliquid API 문서 확인 필요)
                'side': 'sell',
                'amount': amount,
                'price': sl_price,
                'params': {'reduceOnly': True, 'stopLossPrice': sl_price, 'triggerPrice': sl_price,
                           'stopPrice': sl_price},  # triggerPrice, stopPrice, stopLossPrice 모두 시도
            },
        ]

        order_response = exchange.create_orders(orders)  # create_orders 함수 사용

        logging.info(f"Hyperliquid order created: {order_response}")
        return order_response

    except Exception as e:
        logging.error(f"Error creating order on Hyperliquid: {e}")
        return None


def get_hyperliquid_position(exchange, symbol):
    """Hyperliquid 거래소에서 현재 포지션 확인"""
    try:
        positions = exchange.fetch_positions(symbols=[symbol])
        return positions[0] if positions else None  # 단일 포지션 정보 반환
    except Exception as e:
        logging.error(f"Error fetching position from Hyperliquid: {e}")
        return None


def get_hyperliquid_balance(exchange):
    """Hyperliquid 거래소에서 사용 가능한 잔고 확인"""
    try:
        balance = exchange.fetch_balance()
        return float(balance['USDC']['free'])
    except Exception as e:
        logging.error(f"Error fetching balance from Hyperliquid: {e}")
        return 0.0


def close_expired_positions(exchange, symbol):
    """Hyperliquid 거래소에서 거래 기간이 만료된 포지션을 종료"""
    try:
        position = get_hyperliquid_position(exchange, symbol)
        orders = exchange.fetch_open_orders(symbol)

        if not orders and not position:
            return

        utc_entry_time_str = orders[0]['datetime']
        utc_entry_time = datetime.strptime(utc_entry_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")

        # 한국 시간대로 변환
        korea_timezone = pytz.timezone('Asia/Seoul')

        # UTC 시간대 정보 추가
        entry_time = pytz.utc.localize(utc_entry_time).astimezone(korea_timezone)
        df = pd.read_csv('trading_decisions.csv')
        trade_term = list(df['trade_term'])[-1]

        now_kst = datetime.now(KST)

        match = re.match(r"(\d+)([mhdw])", trade_term)
        if not match:
            logging.warning(f"Invalid trade_term format: {trade_term}")
            return

        term_value = int(match.group(1))
        term_unit = match.group(2)

        if term_unit == 'm':
            expiration_time = entry_time + timedelta(minutes=term_value)
        elif term_unit == 'h':
            expiration_time = entry_time + timedelta(hours=term_value)
        elif term_unit == 'd':
            expiration_time = entry_time + timedelta(days=term_value)
        elif term_unit == 'w':
            expiration_time = entry_time + timedelta(weeks=term_value)
        else:
            logging.warning(f"Invalid trade_term unit: {term_unit}")
            return

        if not position:
            if len(orders) >= 3:
                if term_unit == 'm':
                    if now_kst >= expiration_time:
                        order_ids = [order['id'] for order in orders]
                        exchange.cancel_orders(order_ids, symbol)
                        return
                else:
                    if now_kst >= entry_time + timedelta(minutes=30):
                        order_ids = [order['id'] for order in orders]
                        exchange.cancel_orders(order_ids, symbol)
                        return

        if now_kst >= expiration_time:
            logging.info(f"Closing expired position: {position}")
            try:
                close_side = 'sell' if position['side'] == 'long' else 'buy'
                current_price = exchange.fetch_ticker('BTC/USDC:USDC')['last']
                closing_order = exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=abs(float(position['contracts'])),
                    price=current_price,
                    params={'reduceOnly': True}
                )
                logging.info(f"Position closed: {closing_order}")
                # 텔레그램 메시지 전송 (telegram_utils.py 활용)
                from telegram_utils import send_telegram_message  # 상대 경로 import
                send_telegram_message(
                    f"*포지션 자동 종료* ({symbol})\n\n"
                    f"*만료 시간:* {expiration_time.strftime('%Y-%m-%d %H:%M:%S (KST)')}\n"
                    f"*사유:* 거래 기간({trade_term}) 만료"
                )

            except Exception as e:
                logging.error(f"Error creating market order on Hyperliquid in close expired positions: {e}")
                return None

    except Exception as e:
        logging.error(f"Error closing expired positions: {e}")
