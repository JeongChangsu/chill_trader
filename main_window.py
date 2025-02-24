import sys
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import asyncio
from data_collector import DataCollector
from indicator_calculator import IndicatorCalculator
from chart_analyzer import ChartAnalyzer
import logging

# GUI 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("gui.log"), logging.StreamHandler()]
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Trading Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # 데이터 수집기, 지표 계산기, 차트 분석기 초기화
        self.collector = DataCollector()
        self.calculator = IndicatorCalculator()
        self.analyzer = ChartAnalyzer()

        # UI 설정
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.tab_widgets = {}
        for tf in self.timeframes:
            tab = QtWidgets.QWidget()
            self.tabs.addTab(tab, tf)
            layout = QtWidgets.QVBoxLayout()
            tab.setLayout(layout)

            # 캔들 데이터 테이블
            table = QtWidgets.QTableWidget()
            table.setColumnCount(6)
            table.setHorizontalHeaderLabels(['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            layout.addWidget(table)

            # 차트
            fig, ax = plt.subplots(figsize=(10, 4))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)

            # 지표 표시
            indicator_label = QtWidgets.QLabel("Indicators: Waiting for data...")
            layout.addWidget(indicator_label)

            # 분석 표시
            analysis_label = QtWidgets.QLabel("Analysis: Waiting for data...")
            layout.addWidget(analysis_label)

            self.tab_widgets[tf] = {
                'table': table,
                'canvas': canvas,
                'ax': ax,
                'indicator_label': indicator_label,
                'analysis_label': analysis_label
            }

        # 실시간 데이터 수집 시작
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.start_data_collection())

        # UI 업데이트 타이머
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(1000)  # 1초마다 UI 업데이트

    async def start_data_collection(self):
        """실시간 데이터 수집 시작"""
        logging.info("Starting real-time data collection...")
        await self.collector.fetch_real_time_data()

    def update_ui(self):
        """UI를 실시간 데이터로 업데이트"""
        candle_data = self.collector.candle_data
        indicators = self.calculator.calculate_indicators(candle_data)
        analysis = self.analyzer.analyze_charts(candle_data, indicators)

        for tf in self.timeframes:
            widgets = self.tab_widgets[tf]

            # 캔들 데이터 테이블 업데이트
            table = widgets['table']
            if tf in candle_data and candle_data[tf]:
                table.setRowCount(min(len(candle_data[tf]), 100))  # 최대 100개만 표시
                for row, candle in enumerate(candle_data[tf][-100:]):  # 최신 100개
                    table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{candle[0]}"))
                    table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{candle[1]:.2f}"))
                    table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{candle[2]:.2f}"))
                    table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{candle[3]:.2f}"))
                    table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{candle[4]:.2f}"))
                    table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{candle[5]:.2f}"))

            # 차트 업데이트
            ax = widgets['ax']
            canvas = widgets['canvas']
            ax.clear()
            if tf in candle_data and candle_data[tf]:
                closes = [candle[4] for candle in candle_data[tf]]
                ax.plot(closes, label='Close', color='blue')
                if tf in indicators and indicators[tf]:
                    ma5 = indicators[tf]['MA5']
                    ma20 = indicators[tf]['MA20']
                    ax.plot([ma5] * len(closes), label='MA5', linestyle='--', color='orange')
                    ax.plot([ma20] * len(closes), label='MA20', linestyle='--', color='green')
                ax.legend()
                canvas.draw()

            # 지표 업데이트
            indicator_label = widgets['indicator_label']
            if tf in indicators and indicators[tf]:
                indicator_label.setText(
                    f"MA5: {indicators[tf]['MA5']:.2f}, MA20: {indicators[tf]['MA20']:.2f}, "
                    f"RSI: {indicators[tf]['RSI']:.2f}"
                )
            else:
                indicator_label.setText("Indicators: Insufficient data")

            # 분석 업데이트
            analysis_label = widgets['analysis_label']
            if tf in analysis and analysis[tf]:
                patterns = analysis[tf]['patterns']
                trend = analysis[tf]['trendlines']
                sr = analysis[tf]['support_resistance']
                analysis_text = f"Patterns: {patterns}\nTrendline: {trend}\nS/R: {sr}"
                analysis_label.setText(analysis_text)
            else:
                analysis_label.setText("Analysis: Insufficient data")

    def closeEvent(self, event):
        """창을 닫을 때 데이터 수집 중지"""
        self.collector.stop()
        self.loop.stop()
        event.accept()


def run_app():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # PyQt와 asyncio 통합
    QtCore.QTimer.singleShot(0, lambda: asyncio.ensure_future(window.loop.run_forever()))
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
