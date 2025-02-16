# utils.py
import os
import csv
import logging
import holidays

from config import KST
from datetime import datetime


def setup_logging(level=logging.INFO):
    """로깅 기본 설정"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_current_session_kst():
    """KST 기준 현재 세션, 주말, 미국 공휴일 정보"""
    now = datetime.now(KST)
    hour = now.hour
    is_weekend = now.weekday() >= 5
    us_holidays = holidays.US()
    is_us_holiday = now.date() in us_holidays

    if 0 <= hour < 8:
        session = "OVERNIGHT"
    elif 8 <= hour < 16:
        session = "ASIAN"
    elif 16 <= hour < 22:
        session = "LONDON"
    elif 22 <= hour < 24 or 0 <= hour < 6:
        session = "US"
    elif 6 <= hour < 8:
        session = "TRANSITION"
    else:
        session = "UNDEFINED"

    if is_weekend:
        session += "_WEEKEND"
    if is_us_holiday:
        session += "_US_HOLIDAY"
    return session


def log_decision(decision, symbol, filename):
    """거래 결정 내용을 CSV 파일에 기록"""
    # CSV 파일 로깅은 동기적으로 처리 (비동기 X)
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["timestamp", "symbol", "final_action", "leverage", "trade_term", "tp_price", "sl_price",
                 "limit_order_price", "rationale"])
        writer.writerow([datetime.utcnow(), symbol, decision["final_action"], decision["leverage"],
                         decision["trade_term"], decision["tp_price"], decision["sl_price"],
                         decision["limit_order_price"], decision["rationale"]])
    logging.info("Trading decision logged to file.")
