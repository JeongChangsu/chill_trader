# telegram_utils.py

import os
import re
import csv
import time
import pytz
import logging
from datetime import datetime

import telegram
from telegram.ext import CommandHandler, Updater

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Telegram variables (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# KST 타임존 설정
KST = pytz.timezone("Asia/Seoul")

# CSV 파일 기본 경로 (필요에 따라 수정)
BASE_CSV_PATH = "/Users/changpt/PycharmProjects/chill_trader/"
CLOSED_POSITIONS_FILE = os.path.join(BASE_CSV_PATH, "closed_positions.csv")

# 전역 변수
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
updater = Updater(bot=bot, use_context=True)
dispatcher = updater.dispatcher
trading_active = False  # 트레이딩 활성화 상태
current_csv_version = 1
current_csv_filepath = ""


def escape_markdown_v2(text):
    """Telegram Markdown V2에서 문제가 될 수 있는 모든 특수 문자를 이스케이프 처리."""
    escape_chars = r"[_*\[\]()~`>#\+\-=|{}\.!]"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def send_telegram_message(message):
    """텔레그램 메시지 전송 함수 (Markdown V2 형식)."""
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='MarkdownV2')
        logging.info("Telegram message sent successfully.")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")



def start_trading(update, context):
    """자동매매 시작, 새 CSV 파일 생성"""
    global trading_active, current_csv_version, current_csv_filepath
    if trading_active:
        update.message.reply_text("이미 자동매매가 실행 중입니다.")
        return

    trading_active = True
    # 새 CSV 파일 경로 생성
    current_csv_filepath = os.path.join(BASE_CSV_PATH, f"trading_log_v{current_csv_version}.csv")
    current_csv_version += 1 # 버전 증가

    # 새 CSV 파일 생성 및 헤더 작성
    with open(current_csv_filepath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['timestamp', 'symbol', 'side', 'entry_price', 'amount', 'leverage',
                             'tp_price', 'sl_price', 'exit_price', 'profit', 'is_win'])

    update.message.reply_text(f"자동매매를 시작합니다. 매매일지는 {current_csv_filepath}에 기록됩니다.")
    logging.info(f"Trading started. Logging to {current_csv_filepath}")


def stop_trading(update, context):
    """자동매매 종료, CSV 파일 닫기, 최종 결과 보고"""
    global trading_active, current_csv_filepath
    if not trading_active:
        update.message.reply_text("자동매매가 실행 중이지 않습니다.")
        return

    trading_active = False

    # 최종 결과 보고 (calculate_daily_performance 함수 활용)
    total_profit, total_trades, win_rate = calculate_daily_performance(current_csv_filepath)
    message = (
        f"*자동매매 종료 보고서*\n\n"
        f"*총 거래 횟수:* {total_trades}\n"
        f"*총 수익:* {total_profit:.2f} USDT\n"
        f"*승률:* {win_rate:.2f}%\n"
        f"*매매일지 파일:* {current_csv_filepath}"
    )

    # 결과 전송 및 파일 닫기 안내.
    update.message.reply_text(escape_markdown_v2(message))
    current_csv_filepath = "" # 파일 경로 초기화
    logging.info("Trading stopped.")



def get_status(update, context):
    """현재 자동매매 상태, CSV 파일 정보, 요약된 매매일지 정보 제공"""
    global trading_active, current_csv_filepath

    if trading_active:
        status_message = f"자동매매가 실행 중입니다.\n"
        if current_csv_filepath:
            status_message += f"매매일지는 {current_csv_filepath}에 기록되고 있습니다.\n"
            # 요약된 매매일지 정보 (최근 5 거래)
            try:
                df = pd.read_csv(current_csv_filepath)
                if not df.empty:
                    last_5_trades = df.tail(5)
                    trades_summary = "\n".join([
                        f"  - {row['timestamp']} {row['symbol']} {row['side']} {row['entry_price']}"
                        for _, row in last_5_trades.iterrows()
                    ])
                    status_message += f"\n*최근 5 거래 요약:*\n{escape_markdown_v2(trades_summary)}"
                else:
                    status_message += "\n최근 거래 기록이 없습니다."
            except Exception as e:
                status_message += f"\n매매일지 요약 정보 조회 중 오류 발생: {e}"
        else:
             status_message += "매매일지 파일 경로가 설정되지 않았습니다."
    else:
        status_message = "자동매매가 실행 중이지 않습니다."

    update.message.reply_text(escape_markdown_v2(status_message))



def calculate_daily_performance(csv_filepath):
    """
    주어진 CSV 파일을 읽어 일일 수익률, 승률 등을 계산.
    """
    if not os.path.isfile(csv_filepath):
        return 0, 0, 0  # 파일 없으면 0 반환

    total_profit = 0
    total_trades = 0
    winning_trades = 0

    with open(csv_filepath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # 모든 거래 기록 사용 (날짜 필터링 X)
                total_trades += 1
                profit = float(row['profit'])
                total_profit += profit
                if row['is_win'] == 'True':
                    winning_trades += 1
            except (ValueError, KeyError) as e:
                logging.error(f"Error parsing row in closed positions file: {e}, row: {row}")
                continue

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    return total_profit, total_trades, win_rate


def initialize_telegram_bot():
    """텔레그램 봇 초기화 및 핸들러 등록"""
    try:
        logging.info("Initializing Telegram bot...")
        # 핸들러 등록
        dispatcher.add_handler(CommandHandler("start", start_trading))
        dispatcher.add_handler(CommandHandler("stop", stop_trading))
        dispatcher.add_handler(CommandHandler("status", get_status))

        # 폴링 시작
        updater.start_polling()
        logging.info("Telegram bot initialized successfully.")

    except Exception as e:
        logging.error(f"Error initializing Telegram bot: {e}")
