import logging
import sys
import configparser
from datetime import datetime

# 설정 파일 읽기
config = configparser.ConfigParser()
config.read('config.ini')
log_level = config.get('logging', 'level', fallback='INFO')


def setup_logging():
    """로깅 시스템을 설정합니다."""
    logger = logging.getLogger('TradingBot')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)

    # 파일 핸들러
    file_handler = logging.FileHandler(f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
    file_handler.setFormatter(file_format)

    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()

if __name__ == '__main__':
    logger.debug("Debug message test")
    logger.info("Info message test")
    logger.warning("Warning message test")
    logger.error("Error message test")
