# telegram_utils.py
import re
import logging
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_telegram_message(message):
    """Telegram 메시지 전송"""
    if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        logging.error("Telegram bot token or chat ID is not set.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        logging.info("Telegram message sent successfully")
    except Exception as e:
        logging.error(f"Error during Telegram message sending: {e}")


def escape_markdown_v2(text):
    """Telegram Markdown V2에서 특수 문자 이스케이프"""
    escape_chars = r"[_*\[\]()~`>#\+\-=|{}\.!]"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)
