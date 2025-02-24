# ai_integration.py
from google import genai
import os
import json
import pandas as pd

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')


class GeminiIntegration:

    def __init__(self, api_key):
        self.google_api_key = GOOGLE_API_KEY
        self.gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = "gemini-2.0-pro-exp-02-05"  # Use the specified model

    def generate_system_prompt(self):
        return """
        You are a world-class quantitative trader and risk management expert at a leading cryptocurrency hedge fund.
        Your goal is to maximize profit while minimizing risk, with a strong emphasis on capital preservation.
        You have access to cutting-edge technical analysis, pattern recognition, and market context analysis tools.
        You are extremely cautious and avoid false signals (whipsaws).
        You use a multi-timeframe approach, considering the 1-hour and 4-hour charts.

        Analyze the provided market data and provide a clear trading decision. Explain your reasoning in detail,
        citing specific indicators, patterns, and market conditions. Be as specific as possible. If the information
        provided is insufficient to make a high-confidence decision, do *NOT* provide a trade, and explain why.

        Strive to provide a concrete answer. Minimize responses like "Not Applicable".

        **Input Data:**

        *   Market Context: Trend, volatility, volume, and whether the market is sideways (ranging).
        *   Technical Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, Supertrend, ADX, and more.
        *   Candlestick and Chart Patterns.
        *   (Future) On-chain data (whale activity, exchange balance changes, etc.).
        *   (Future) Market sentiment (Fear & Greed Index, sentiment scores, etc.).
        *   (Future) Current position (entry price, quantity, leverage).
        *   (Future) Recent trade history.

        **Output Format (JSON):**

        ```json
        {
          "action": "buy" | "sell" | "hold",  // "buy" for long, "sell" for short, "hold" for no action
          "entry_price": { "value": number, "unit": "USD" }, //  Limit order price (if applicable).
          "leverage": { "value": number, "unit": "x" },    // Leverage (e.g., 1, 2, 5).  Use 1 for no leverage.
          "take_profit": { "value": number, "unit": "USD" }, // Take profit price.
          "stop_loss": { "value": number, "unit": "USD" },  // Stop loss price.
          "reasoning": string,  // Detailed explanation of the decision-making process. Be specific.
          "duration": { "value": number, "unit": "hours" },   // Expected duration of the trade (in hours).
          "confidence": number, // Confidence level of this trade (0.0 to 1.0, where 1.0 is the highest confidence).
          "precautions": string // Any warnings, caveats, or additional advice.
        }
        ```
        """

    def generate_user_prompt(self, market_context, indicators, patterns, position=None, recent_trades=None):
        """Generates a user prompt for Gemini."""

        # Market Context
        market_summary = f"Market Context: Trend: {market_context['trend']}, Volatility: {market_context['volatility']}, Volume: {market_context['volume']}, Sideways: {market_context['sideways']}"

        # Technical Indicators (formatted for readability)
        indicator_str = "Technical Indicators:\n"
        for key, value in indicators.items():
            if isinstance(value, tuple):  # Handle indicators that return tuples (e.g., Bollinger Bands)
                if all(isinstance(item, pd.Series) for item in value):
                    indicator_str += f"* {key}: " + ", ".join(
                        [f"{sub_val.iloc[-1]:.2f}" if not pd.isna(sub_val.iloc[-1]) else "N/A" for sub_val in
                         value]) + "\n"
                else:
                    indicator_str += f"* {key}: " + ", ".join(
                        [f"{v:.2f}" if isinstance(v, (int, float)) and not pd.isna(v) else "N/A" for v in value]) + "\n"
            elif isinstance(value, pd.Series):
                indicator_str += f"* {key}: {value.iloc[-1]:.2f if not pd.isna(value.iloc[-1]) else 'N/A'}\n"
            else:  # Other
                indicator_str += f"* {key}: {value:.2f}\n"  #

        # Patterns
        patterns_str = "Patterns:\n" + str(patterns)

        # On-chain data and market sentiment (placeholders for now)
        onchain_str = "On-chain Data: (To be added later)\n"
        sentiment_str = "Market Sentiment: (To be added later)\n"

        # Current Position and Recent Trades (placeholders for now)
        position_str = "Current Position: (To be added later)\n"
        recent_trades_str = "Recent Trades: (To be added later)\n"

        user_prompt = f"""
        {market_summary}

        {indicator_str}

        {patterns_str}

        {onchain_str}

        {sentiment_str}

        {position_str}

        {recent_trades_str}

        Given this information, provide a trading decision.  Be as specific and quantitative as possible with your entry price, leverage, take profit, and stop loss. Explain your reasoning in detail, citing specific indicators, patterns, and market conditions that support your decision. If the information is insufficient, explain *why* you cannot make a high-confidence decision.  Express your confidence level as a number between 0 and 1. Provide also expected duration and precautions.
        """

        return user_prompt

    def get_ai_decision(self, market_context, indicators, patterns, position=None, recent_trades=None):

        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(market_context, indicators, patterns, position, recent_trades)

        prompt_parts = [
            system_prompt,
            user_prompt
        ]
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=prompt_parts
            )
            # Parse the JSON response
            return json.loads(response.text)

        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return None
