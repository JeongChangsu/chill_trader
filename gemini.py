# gemini.py
import re
import os
import logging

from PIL import Image
from datetime import datetime
from google.genai import types
from config import KST, SESSION_GUIDES

def generate_gemini_prompt(multi_tf_data, market_regime, strategy, thresholds,
                           current_session, additional_data, econ_summary):
    """Gemini Pro 모델에 전달할 Prompt를 생성"""

    # 전략 정보
    strategy_name = strategy["name"]
    strategy_description = strategy["description"]
    primary_tf = strategy["primary_timeframe"]

    # 지표 요약
    indicators_summary = ""
    for tf, data in multi_tf_data.items():
        indicators_summary += f"**{tf}:**\n"
        indicators_summary += f"  - Price: {data['current_price']:.2f}\n"
        for ind_name, ind_params in strategy["indicators"].items():
            if ind_name in data and data[ind_name] is not None:
                indicators_summary += f"  - {ind_name.upper()}: {data[ind_name]:.2f}\n"

    # 진입/청산 규칙
    entry_rules_long = "\n".join([f"  - {rule}" for rule in strategy["entry_rules"].get("long", [])])
    entry_rules_short = "\n".join([f"  - {rule}" for rule in strategy["entry_rules"].get("short", [])])

    # TP/SL 규칙
    tp_rule = strategy["exit_rules"]["tp"]
    sl_rule = strategy["exit_rules"]["sl"]

    # 현재 시간 (KST)
    now_kst = datetime.now(KST)
    current_time_kst = now_kst.strftime("%Y-%m-%d %H:%M:%S (KST)")

    # 세션별 가이드
    session_guide = SESSION_GUIDES.get(current_session, "No specific guidance for this session.")

    prompt_text_1 = f"""
**Objective:** Make optimal trading decisions for BTC/USDT.

**Market Context:**
- Regime: **{market_regime.upper()}**
- Strategy: **{strategy_name}** ({strategy_description})
- Primary Timeframe: **{primary_tf}**
- Current Session: **{current_session}** ({current_time_kst})
- Session Guide: {session_guide}
- Economic Events: {econ_summary}

**Technical Analysis Summary:**
{indicators_summary}

**Key Indicator Thresholds:**
- RSI Oversold: {thresholds.get('rsi_oversold', 'N/A')}
- RSI Overbought: {thresholds.get('rsi_overbought', 'N/A')}
- Donchian Window: {thresholds.get('donchian_window', 'N/A')}

**Additional Market Data:**
- Funding Rate: {additional_data.get('funding_rate', 'N/A')}
- Open Interest: {additional_data.get('open_interest', 'N/A')}
- Order Book: Bid={additional_data.get('order_book', {}).get('bid', 'N/A')}, Ask={additional_data.get('order_book', {}).get('ask', 'N/A')}, Spread={additional_data.get('order_book', {}).get('spread', 'N/A')}
- Fear & Greed: {additional_data.get('fear_and_greed_index', ('N/A', 'N/A'))[0]} ({additional_data.get('fear_and_greed_index', ('N/A', 'N/A'))[1]})

**Strategy Guidelines:**

- **Entry Rules (Long):**
{entry_rules_long}

- **Entry Rules (Short):**
{entry_rules_short}

- **Take Profit (TP):** {tp_rule}
- **Stop Loss (SL):** {sl_rule}
    """
    prompt_text_2 = f"""
**Liquidation Map Analysis Guide(Image Provided):**
- **Support and Resistance:** Identify potential support and resistance levels.
- **Cascading Liquidations:** Assess the risk of cascading liquidations.
- **Volatility Prediction:**  Estimate potential volatility.
- **Risk Assessment:** Compare long vs. short liquidation levels.

**Task:**

Based on all provided information, decide: **GO LONG, GO SHORT, or NO TRADE.**

If GO LONG or GO SHORT, also determine:
- **Recommended Leverage:** (Based on strategy's recommendation)
- **Trade Term:** (Based on strategy's recommendation)
- **Take Profit Price:** 
- **Stop Loss Price:**
- **Limit Order Price:**
- **Amount (USD):**
- **Rationale:** (Explain your decision, including liquidation map, indicators, and market context.)

**Output Format (Comma-Separated):**

Final Action, Recommended Leverage, Trade Term, Take Profit Price, Stop Loss Price, Limit Order Price, Amount (USD), Rationale
    """

    return prompt_text_1, prompt_text_2


def generate_trading_decision(gemini_client, multi_tf_data, market_regime, strategy, thresholds,
                              current_session, additional_data, econ_summary):
    """Gemini Pro 모델에 프롬프트 전달, 거래 결정 반환"""
    prompt_part_1, prompt_part_2 = generate_gemini_prompt(
        multi_tf_data, market_regime, strategy, thresholds,
        current_session, additional_data, econ_summary
    )

    home_dir = os.path.expanduser("~")
    downloads_path = os.path.join(home_dir, "Downloads", "Liquidation Map.png")

    image = Image.open(downloads_path)

    logging.info("------- Gemini Prompt -------")
    logging.info(f"{prompt_part_1}\n{prompt_part_2}")
    logging.info("------- End Prompt -------")

    sys_instruct = "You are a world-class cryptocurrency trader specializing in BTC/USDT."
    response = gemini_client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05",
        config=types.GenerateContentConfig(system_instruction=sys_instruct),
        contents=[prompt_part_1, image, prompt_part_2]
    )

    try:
        os.remove(downloads_path)
        logging.info("Deleted the liquidation heatmap image file after processing.")
    except Exception as e:
        logging.error(f"Error deleting the image file: {e}")
    return response.text


def parse_trading_decision(response_text):
    """Gemini 응답 텍스트 파싱"""
    decision = {
        "final_action": "NO TRADE",
        "leverage": "N/A",
        "trade_term": "N/A",
        "tp_price": "N/A",
        "sl_price": "N/A",
        "limit_order_price": "N/A",
        "amount_usd": "N/A",
        "rationale": "N/A"
    }

    if not response_text:
        logging.warning("parse_trading_decision received empty response_text.")
        return decision

    try:
        match = re.search(r"GO (LONG|SHORT).*?,(.*?)x, *(.*?), *(.*?), *(.*?), *(.*?), *(.*?), *(.*)", response_text,
                          re.DOTALL | re.IGNORECASE)

        if match:
            decision["final_action"] = f"GO {match.group(1).upper()}"
            decision["leverage"] = match.group(2).strip()
            decision["trade_term"] = match.group(3).strip()
            decision["tp_price"] = match.group(4).strip()
            decision["sl_price"] = match.group(5).strip()
            decision["limit_order_price"] = match.group(6).strip()
            decision["amount_usd"] = match.group(7).strip()
            decision["rationale"] = match.group(8).strip()

    except Exception as e:
        logging.error(f"Error parsing Gemini response: {e}")
        decision["rationale"] = response_text

    logging.info("Parsed Trading Decision:")
    logging.info(decision)
    return decision
