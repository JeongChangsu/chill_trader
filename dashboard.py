import panel as pn
import pandas as pd
import hvplot.pandas
from data_collector import DataCollector
from indicator_calculator import calculate_indicators
from chart_analyzer import analyze_charts
import asyncio
import logging

pn.extension()


class TradingDashboard:
    def __init__(self):
        self.collector = DataCollector()
        self.tf_selector = pn.widgets.Select(name='Timeframe', options=['5m', '15m', '1h', '4h', '1d'], value='1h')
        self.plot_pane = pn.pane.HoloViews()
        self.update_button = pn.widgets.Button(name="Update", button_type="primary")
        self.update_button.on_click(self._manual_update)
        self.layout = pn.Column(
            pn.Row(self.tf_selector, self.update_button),
            self.plot_pane
        )
        asyncio.create_task(self._start_data_collection())

    async def _start_data_collection(self):
        await self.collector.fetch_real_time_data()

    def _update_plot(self, tf):
        df = self.collector.candle_data[tf]
        if df.empty:
            logging.warning(f"{tf} 데이터 없음")
            return pn.pane.Markdown("데이터 수집 중...")
        indicators = calculate_indicators({tf: df})
        analysis = analyze_charts({tf: df})

        candlestick = df.hvplot.candlestick(x='timestamp', y=['open', 'high', 'low', 'close'], title=f"{tf} 차트")
        ma5 = df.hvplot.line(x='timestamp', y='MA5', color='orange', label='MA5')
        if analysis[tf]['trendline']:
            trend = analysis[tf]['trendline']
            x = [df['timestamp'].iloc[0], df['timestamp'].iloc[-1]]
            y = [trend['intercept'] + trend['slope'] * i for i in [0, len(df) - 1]]
            trendline = pd.DataFrame({'timestamp': x, 'trend': y}).hvplot.line(x='timestamp', y='trend', color='red')
            return candlestick * ma5 * trendline
        return candlestick * ma5

    def _manual_update(self, event):
        self.plot_pane.object = self._update_plot(self.tf_selector.value)


dashboard = TradingDashboard()
dashboard.layout.servable()
