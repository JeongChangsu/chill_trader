import re
import json
import logging

from google import genai  # google.genai import 추가
from config import GOOGLE_API_KEY  # GOOGLE_API_KEY import 추가

# Gemini Client initialization (updated)
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# 로깅 설정 (PEP8 스타일)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def select_trading_strategy(market_analysis_result: dict) -> dict:
    """
    시장 분석 결과에 따라 트레이딩 전략을 선택하는 함수 (Gemini API Client 업데이트).

    Args:
        market_analysis_result (dict): 1단계 시장 분석 결과 (analyze_market() 함수의 출력).

    Returns:
        dict: 트레이딩 전략 선택 결과 (JSON 포맷).
              예시: {"selected_trading_strategy": "Momentum Trading", "strategy_choice_rationale": "..."}
    """
    try:
        market_situation_category = market_analysis_result.get("market_situation_category")
        confidence_level = market_analysis_result.get("confidence_level")
        supporting_rationale = market_analysis_result.get("supporting_rationale")

        if not market_situation_category:
            raise ValueError("Market situation category is missing from market analysis result.")

        prompt_text = f"""You are a world-class cryptocurrency trading strategist. Based on the current market situation, please select the most appropriate trading strategy from the following options:

        * Momentum Trading (Trend Following)
        * Mean Reversion Trading

        **Current Market Situation:** {market_situation_category} (Confidence Level: {confidence_level})
        **Supporting Rationale:** {supporting_rationale} (from Stage 1 Market Analysis)

        **Constraints:**

        * In a **Bull Market**, prioritize **Momentum Trading (Long)** to maximize profit from the upward trend.
        * In a **Bear Market**, you must choose between **Momentum Trading (Short)** and **Mean Reversion Trading**. Consider the risk level and potential profitability of each strategy in the current Bear Market context. If Momentum Trading (Short) is deemed too risky for automated trading or if the Bear Market is showing signs of range-bound behavior, Mean Reversion Trading might be a safer alternative. Explain your choice.
        * In a **Range-Bound Market**, use **Mean Reversion Trading** to capitalize on price oscillations within the range.

        **Output Requirements:**

        * **Selected Trading Strategy:** (e.g., "Momentum Trading", "Mean Reversion Trading") - Clearly state the selected strategy.
        * **Strategy Choice Rationale:** - Briefly explain why you selected the chosen strategy based on the current market situation and constraints.  For Bear Market strategy selection, explicitly justify your choice between Momentum Trading (Short) and Mean Reversion Trading.

        **Example Output Format (JSON):**
        ```json
        {{
          "selected_trading_strategy": "Momentum Trading",
          "strategy_choice_rationale": "The market is classified as a Bull Market with high confidence. Momentum Trading (Long) is the most suitable strategy to capture gains in a strong uptrend."
        }}
        ```"""  # JSON 포맷 예시 수정 (중괄호 두 개)

        prompt_parts = [prompt_text]  # prompt_parts 배열 생성

        logging.info("Sending strategy selection prompt to Gemini...")
        response = gemini_client.models.generate_content(  # Gemini API 호출 방식 업데이트
            model="gemini-2.0-flash-thinking-exp-01-21",  # 모델명 확인 필요 (최신 모델로 업데이트)
            # model="gemini-2.0-pro-exp-02-05",  # 모델명 업데이트
            contents=prompt_parts  # contents 파라미터 사용
        )

        response_text = response.text  # response.text로 결과 텍스트 획득
        logging.info(f"Gemini API response for strategy selection: {response_text}")

        gemini_output = re.sub(r"```json\n?(.*?)\n?```", r"\1", response_text, flags=re.DOTALL)
        strategy_selection_result = json.loads(gemini_output)  # JSON 파싱 오류 가능성 (try-except 처리)
        selected_strategy = strategy_selection_result.get("selected_trading_strategy")
        strategy_choice_rationale = strategy_selection_result.get("strategy_choice_rationale")

        if not selected_strategy or not strategy_choice_rationale:
            raise ValueError("Gemini API response missing 'selected_trading_strategy' or 'strategy_choice_rationale'.")

        strategy_result = {
            "selected_trading_strategy": selected_strategy,
            "strategy_choice_rationale": strategy_choice_rationale
        }
        logging.info(f"Trading Strategy Selection Result: {strategy_result}")
        return strategy_result

    except Exception as e:
        logging.error(f"Error selecting trading strategy: {e}")
        return {}  # 에러 발생 시 빈 dict 반환 or None 반환 (main.py 에서 처리 방식에 따라 결정)


def set_strategy_parameters(market_analysis_result: dict, selected_strategy_result: dict) -> dict:
    """
    선택된 트레이딩 전략에 대한 파라미터를 Gemini 모델을 통해 설정하는 함수 (Gemini API Client 업데이트).

    Args:
        market_analysis_result (dict): 1단계 시장 분석 결과 (analyze_market() 함수의 출력).
        selected_strategy_result (dict): 트레이딩 전략 선택 결과 (select_trading_strategy() 함수의 출력).

    Returns:
        dict: 전략 파라미터 설정 결과 (JSON 포맷).
              예시: {"strategy_parameters": {"MA_short": 50, ...}, "parameter_setting_rationale": "..."}
    """
    try:
        market_situation_category = market_analysis_result.get("market_situation_category")
        confidence_level = market_analysis_result.get("confidence_level")
        supporting_rationale = market_analysis_result.get("supporting_rationale")
        selected_trading_strategy = selected_strategy_result.get("selected_trading_strategy")
        strategy_choice_rationale = selected_strategy_result.get("strategy_choice_rationale")

        if not selected_trading_strategy:
            raise ValueError("Selected trading strategy is missing from strategy selection result.")

        prompt_text = f"""You are a world-class cryptocurrency trading expert, skilled in optimizing trading strategy parameters. Based on the current market situation and the selected trading strategy, please determine the optimal parameters for the strategy.

        **Current Market Situation:** {market_situation_category} (Confidence Level: {confidence_level})
        **Selected Trading Strategy:** {selected_trading_strategy}
        **Strategy Choice Rationale:** {strategy_choice_rationale} (from Strategy Selection Stage)
        **Market Analysis Rationale:** {supporting_rationale} (from Stage 1 Market Analysis)

        **Parameter Setting Guidelines:**

        * **For Momentum Trading:**
            * **Moving Average Periods (MA_short, MA_long):** Suggest appropriate periods for short-term and long-term Moving Averages, considering the current market volatility and trend strength. Provide a range or specific values. (e.g., MA_short: 20-50, MA_long: 100-200)
            * **RSI Overbought/Oversold Levels (RSI_overbought, RSI_oversold):** Suggest RSI levels for overbought and oversold conditions, optimized for Momentum Trading in the current market. (e.g., RSI_overbought: 70-80, RSI_oversold: 30-40)
            * **Stop Loss and Take Profit (SL_percentage, TP_ratio):** Recommend Stop Loss percentage (e.g., 1-3% of entry price) and Take Profit ratio (Risk-Reward Ratio, e.g., 1:2, 1:3) suitable for Momentum Trading in this market. Consider using Trailing Stop Loss if appropriate for Bull/Bear Market.

        * **For Mean Reversion Trading:**
            * **RSI Overbought/Oversold Levels (RSI_overbought, RSI_oversold):** Suggest RSI levels for overbought and oversold conditions, optimized for Mean Reversion Trading in the current market. (e.g., RSI_overbought: 60-70, RSI_oversold: 30-40 - narrower range than Momentum Trading)
            * **Bollinger Bands Parameters (BB_length, BB_std_dev):** Suggest period and standard deviation for Bollinger Bands, suitable for capturing range-bound price movements. (e.g., BB_length: 20, BB_std_dev: 2)
            * **Take Profit Percentage (TP_percentage):** Recommend Take Profit percentage for Mean Reversion trades, considering typical range-bound volatility. (e.g., 1-3% of entry price - smaller than Momentum Trading TP)
            * **Stop Loss Percentage (SL_percentage):** Recommend Stop Loss percentage for Mean Reversion trades. (e.g., 1-2% of entry price - tighter Stop Loss than Momentum Trading)

        **Output Requirements:**

        * **Strategy Parameters:** (JSON format dictionary) - Provide a JSON dictionary containing the recommended parameter values for the selected trading strategy.  The keys should be parameter names (e.g., "MA_short", "RSI_overbought"), and the values should be the suggested numerical values or ranges.
        * **Parameter Setting Rationale:** - Briefly explain the reasoning behind your parameter recommendations, considering the current market situation and the characteristics of the selected strategy.

        **Example Output Format (JSON):**
        ```json
        {{
          "strategy_parameters": {{
            "MA_short": 50,
            "MA_long": 200,
            "RSI_overbought": 75,
            "RSI_oversold": 35,
            "SL_percentage": 2,
            "TP_ratio": 3
          }},
          "parameter_setting_rationale": "Based on the Bull Market conditions, wider RSI overbought/oversold levels and a higher Take Profit ratio are recommended to maximize profit potential.  MA periods are set to standard values for trend identification."
        }}
        ```"""  # JSON 포맷 예시 수정 (중괄호 두 개)

        prompt_parts = [prompt_text]  # prompt_parts 배열 생성

        logging.info("Sending strategy parameter setting prompt to Gemini...")
        response = gemini_client.models.generate_content(  # Gemini API 호출 방식 업데이트
            model="gemini-2.0-flash-thinking-exp-01-21",  # 모델명 확인 필요 (최신 모델로 업데이트)
            # model="gemini-2.0-pro-exp-02-05",  # 모델명 업데이트
            contents=prompt_parts  # contents 파라미터 사용
        )

        response_text = response.text  # response.text로 결과 텍스트 획득
        logging.info(f"Gemini API response for strategy selection: {response_text}")

        gemini_output = re.sub(r"```json\n?(.*?)\n?```", r"\1", response_text, flags=re.DOTALL)

        parameter_setting_result = json.loads(gemini_output)  # JSON 파싱 오류 가능성 (try-except 처리)
        strategy_parameters = parameter_setting_result.get("strategy_parameters")
        parameter_setting_rationale = parameter_setting_result.get("parameter_setting_rationale")

        if not strategy_parameters or not parameter_setting_rationale:
            raise ValueError("Gemini API response missing 'strategy_parameters' or 'parameter_setting_rationale'.")

        parameter_result = {
            "strategy_parameters": strategy_parameters,
            "parameter_setting_rationale": parameter_setting_rationale
        }
        logging.info(f"Strategy Parameter Setting Result: {parameter_result}")
        return parameter_result


    except Exception as e:
        logging.error(f"Error setting strategy parameters: {e}")
        return {}  # 에러 발생 시 빈 dict 반환 or None 반환 (main.py 에서 처리 방식에 따라 결정)


def analyze_strategy(market_analysis_result: dict) -> dict:  # gemini_client 인자 제거
    """
    2단계 트레이딩 전략 선택 및 파라미터 설정 기능을 통합한 메인 함수 (Gemini API Client 업데이트).

    Args:
        market_analysis_result (dict): 1단계 시장 분석 결과 (analyze_market() 함수의 출력).

    Returns:
        dict: 2단계 트레이딩 전략 분석 결과 (JSON 포맷).
              예시: {
                    "selected_trading_strategy": "Momentum Trading",
                    "strategy_choice_rationale": "...",
                    "strategy_parameters": { ... },
                    "parameter_setting_rationale": "..."
                  }
    """
    logging.info("Starting Trading Strategy Analysis...")

    if not market_analysis_result:  # market_analysis_result가 비어있는 경우 (None 또는 빈 dict)
        logging.warning("Market analysis result is empty. Strategy analysis aborted.")
        return {}  # 빈 dict 반환 or None 반환 (main.py 에서 에러 처리 방식에 따라 결정)

    selected_strategy_result = select_trading_strategy(market_analysis_result)  # gemini_client 인자 제거
    if not selected_strategy_result:  # 전략 선택 실패 시
        logging.warning("Strategy selection failed. Strategy analysis aborted.")
        return {}  # 빈 dict 반환 or None 반환 (main.py 에서 에러 처리 방식에 따라 결정)

    parameter_setting_result = set_strategy_parameters(market_analysis_result,
                                                       selected_strategy_result)  # gemini_client 인자 제거
    if not parameter_setting_result:  # 파라미터 설정 실패 시
        logging.warning("Strategy parameter setting failed. Strategy analysis aborted.")
        return {}  # 빈 dict 반환 or None 반환 (main.py 에서 에러 처리 방식에 따라 결정)

    strategy_analysis_result = {**selected_strategy_result,
                                **parameter_setting_result}  # dict merge (Python 3.5+), or use update()
    logging.info("Trading Strategy Analysis Completed.")
    logging.info(f"Trading Strategy Analysis Result: {strategy_analysis_result}")
    return strategy_analysis_result
