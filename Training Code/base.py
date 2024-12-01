import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum


class Factor(Enum):
    """因子类型枚举"""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    VOLUME = "volume"
    REVERSAL = "reversal"
    RSI = "rsi"  # 新增RSI因子
    MACD = "macd"  # 新增MACD因子


class FactorCalculator(ABC):
    """抽象因子计算基类"""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> float:
        pass


class MomentumFactor(FactorCalculator):
    def __init__(self, period: int = 12):
        """
        改进的动量因子，使用指数加权动量

        Parameters:
        period: 周期（默认12个5分钟 = 1小时）
        """
        self.period = period

    def calculate(self, df: pd.DataFrame) -> float:
        if len(df) < self.period:
            return 0.0

        try:
            # 计算指数加权动量
            prices = df['close'].tail(self.period)
            returns = np.log(prices / prices.shift(1))
            # 使用指数加权平均，给近期收益更高权重
            weighted_return = returns.ewm(span=self.period).mean().iloc[-1]

            # 将结果标准化到[-1, 1]范围
            return float(np.tanh(weighted_return * 10))

        except Exception as e:
            print(f"动量因子计算出错: {str(e)}")
            return 0.0


class RSIFactor(FactorCalculator):
    def __init__(self, period: int = 14):
        """
        相对强弱指数(RSI)因子

        Parameters:
        period: RSI计算周期（默认14个5分钟周期）
        """
        self.period = period

    def calculate(self, df: pd.DataFrame) -> float:
        if len(df) < self.period + 1:
            return 0.0

        try:
            # 计算价格变化
            price_diff = df['close'].diff()

            # 分别计算上涨和下跌幅度
            gains = price_diff.copy()
            gains[gains < 0] = 0
            losses = -price_diff.copy()
            losses[losses < 0] = 0

            # 计算平均上涨和下跌幅度
            avg_gains = gains.rolling(window=self.period).mean()
            avg_losses = losses.rolling(window=self.period).mean()

            # 计算相对强度
            rs = avg_gains.iloc[-1] / avg_losses.iloc[-1] if avg_losses.iloc[-1] != 0 else 0
            rsi = 100 - (100 / (1 + rs))

            # 将RSI值标准化到[-1, 1]范围
            # RSI > 70 超买区间，RSI < 30 超卖区间
            normalized_rsi = -1.0 if rsi > 70 else (1.0 if rsi < 30 else (rsi - 50) / 20)

            return float(np.clip(normalized_rsi, -1, 1))

        except Exception as e:
            print(f"RSI因子计算出错: {str(e)}")
            return 0.0


class MACDFactor(FactorCalculator):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        MACD因子

        Parameters:
        fast_period: 快线周期（默认12个5分钟周期）
        slow_period: 慢线周期（默认26个5分钟周期）
        signal_period: 信号线周期（默认9个5分钟周期）
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, df: pd.DataFrame) -> float:
        if len(df) < max(self.fast_period, self.slow_period) + self.signal_period:
            return 0.0

        try:
            close_prices = df['close']

            # 计算快线和慢线的EMA
            fast_ema = close_prices.ewm(span=self.fast_period, adjust=False).mean()
            slow_ema = close_prices.ewm(span=self.slow_period, adjust=False).mean()

            # 计算MACD线
            macd_line = fast_ema - slow_ema

            # 计算信号线
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

            # 计算MACD柱状图
            macd_histogram = macd_line - signal_line

            # 获取最新值
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_hist = macd_histogram.iloc[-1]

            # 计算综合得分
            # 1. MACD与信号线的差距
            diff_score = (current_macd - current_signal) / close_prices.iloc[-1]

            # 2. MACD柱状图的变化趋势
            hist_change = current_hist - macd_histogram.iloc[-2]
            hist_score = np.sign(hist_change) * min(abs(hist_change) / close_prices.iloc[-1], 1)

            # 3. 组合得分
            combined_score = 0.7 * diff_score + 0.3 * hist_score

            # 标准化到[-1, 1]范围
            return float(np.tanh(combined_score * 10))

        except Exception as e:
            print(f"MACD因子计算出错: {str(e)}")
            return 0.0


class VolatilityFactor(FactorCalculator):
    def __init__(self, period: int = 24):
        """
        简化的波动率因子

        Parameters:
        period: 周期（默认24个5分钟 = 2小时）
        """
        self.period = period

    def calculate(self, df: pd.DataFrame) -> float:
        if len(df) < self.period:
            return 0.0

        try:
            # 使用对数收益率
            returns = np.log(df['close'] / df['close'].shift(1)).tail(self.period)

            # 计算当前波动率（短期）
            current_vol = returns.tail(self.period // 4).std()

            # 计算历史波动率（长期）
            hist_vol = returns.std()

            if hist_vol == 0:
                return 0.0

            # 计算波动率比率并标准化
            vol_ratio = (current_vol / hist_vol) - 1

            # 返回负的波动率信号（波动率上升为负面信号）
            return float(-np.tanh(vol_ratio * 3))

        except Exception as e:
            print(f"波动率因子计算出错: {str(e)}")
            return 0.0


class TrendFactor(FactorCalculator):
    def __init__(self, period: int = 36):
        """
        简化的趋势因子，使用指数移动平均

        Parameters:
        period: 周期（默认36个5分钟 = 3小时）
        """
        self.period = period

    def calculate(self, df: pd.DataFrame) -> float:
        if len(df) < self.period:
            return 0.0

        try:
            prices = df['close'].tail(self.period)

            # 计算指数移动平均
            ema = prices.ewm(span=self.period).mean()

            # 计算当前价格相对于EMA的位置
            price_ratio = (prices.iloc[-1] / ema.iloc[-1]) - 1

            # 计算EMA的斜率
            ema_slope = (ema.iloc[-1] / ema.iloc[-2]) - 1

            # 组合两个信号
            trend_score = 0.7 * price_ratio + 0.3 * ema_slope

            return float(np.tanh(trend_score * 5))

        except Exception as e:
            print(f"趋势因子计算出错: {str(e)}")
            return 0.0


class ReversalFactor(FactorCalculator):
    def __init__(self, period: int = 24):
        """
        简化的反转因子，使用价格偏离度

        Parameters:
        period: 周期（默认24个5分钟 = 2小时）
        """
        self.period = period

    def calculate(self, df: pd.DataFrame) -> float:
        if len(df) < self.period:
            return 0.0

        try:
            prices = df['close'].tail(self.period)

            # 计算移动平均
            ma = prices.mean()

            # 计算价格偏离度
            deviation = (prices.iloc[-1] / ma) - 1

            # 计算超买超卖指标
            returns = prices.pct_change()
            pos_returns = returns[returns > 0].sum()
            neg_returns = abs(returns[returns < 0].sum())

            if pos_returns + neg_returns == 0:
                rsi = 0.5
            else:
                rsi = pos_returns / (pos_returns + neg_returns)

            # 综合价格偏离度和RSI
            reversal_score = -0.6 * deviation - 0.4 * (rsi - 0.5) * 2

            return float(np.clip(reversal_score, -1, 1))

        except Exception as e:
            print(f"反转因子计算出错: {str(e)}")
            return 0.0


class VolumeFactor(FactorCalculator):
    def __init__(self, period: int = 18):
        """
        优化标准化方法的成交量因子

        Parameters:
        period: 计算周期，默认18个5分钟周期（1.5小时）
        """
        self.period = period

    def _process_volume(self, volumes: pd.Series) -> pd.Series:
        """
        成交量预处理：
        1. 处理0值和异常值
        2. 对数变换
        3. 使用min-max标准化到[-1, 1]范围
        """
        # 处理0值
        volumes = volumes.replace(0, np.nan)
        volumes = volumes.fillna(method='ffill').fillna(method='bfill')

        # 处理异常值 (2.0 * IQR法则)
        Q1 = volumes.quantile(0.25)
        Q3 = volumes.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR
        volumes = np.clip(volumes, lower_bound, upper_bound)

        # 对数变换
        volumes = np.log1p(volumes)

        # Min-Max标准化到[-1, 1]范围
        vol_min = volumes.min()
        vol_max = volumes.max()
        if vol_max > vol_min:
            volumes = 2 * (volumes - vol_min) / (vol_max - vol_min) - 1

        return volumes

    def calculate(self, df: pd.DataFrame) -> float:
        """
        计算成交量因子:
        1. 相对成交量比例
        2. 量价关系
        """
        if len(df) < self.period:
            return 0.0

        try:
            # 获取计算窗口的数据
            window_data = df.tail(self.period).copy()

            # 处理成交量数据
            volumes = self._process_volume(window_data['volume'])

            # 1. 计算相对成交量比例
            current_vol = volumes.iloc[-1]  # 当前成交量
            avg_vol = volumes.iloc[:-1].mean()  # 历史平均成交量
            vol_ratio = (current_vol - avg_vol) if avg_vol != 0 else 0

            # 2. 计算量价关系
            # 使用收益率的绝对值来表示价格变动幅度
            returns = window_data['close'].pct_change()
            abs_returns = returns.abs()

            # 最近的量价关系与历史的对比
            current_ratio = (volumes.iloc[-1] / abs_returns.iloc[-1]) if abs_returns.iloc[-1] != 0 else 0
            hist_ratio = (volumes.iloc[:-1] / abs_returns.iloc[:-1]).mean()
            price_vol_ratio = (current_ratio - hist_ratio) if hist_ratio != 0 else 0

            # 组合两个信号
            score = 0.6 * np.tanh(vol_ratio * 2) + 0.4 * np.tanh(price_vol_ratio * 2)
            return float(np.clip(score, -1, 1))

        except Exception as e:
            print(f"成交量因子计算出错: {str(e)}")
            return 0.0