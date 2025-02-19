# main.py

import json
import logging

from market import analyze_market
from strategy import analyze_strategy


def main():
    market_analysis_result = analyze_market()
    strategy_analysis_result = analyze_strategy(market_analysis_result)

    if strategy_analysis_result:  # 빈 dict 가 아닌 경우에만 출력
        json_output = json.dumps(strategy_analysis_result, indent=4, ensure_ascii=False)
        print(json_output)
    else:
        logging.error("Trading strategy analysis failed to produce a valid result.")
