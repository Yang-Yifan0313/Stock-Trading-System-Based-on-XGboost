from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from enum import Enum
from trading_data_manager import TradingDataManager, TradingPoint

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class TradingPoint:
    """交易点数据结构"""
    position: int                # 当前位置索引
    historical_data: pd.DataFrame  # 包含因子计算所需的历史数据
    current_price: float        # 当前价格
    future_price: float         # 未来价格（用于回测


class PredictionResult(Enum):
    """预测结果枚举"""
    EXTREME_DOWN = 0
    DOWN = 1
    NEUTRAL = 2
    UP = 3
    EXTREME_UP = 4


@dataclass
class StockPrediction:
    """股票预测结果数据结构"""
    stock_code: str
    prediction: PredictionResult
    confidence: float
    position: int  # 替换timestamp为position

    class TradingDataManager:
        """交易数据管理器"""

        def __init__(self, warmup_period: int = 300, prediction_window: int = 49):
            """
            Parameters
            ----------
            warmup_period : int
                因子计算需要的历史数据点数
            prediction_window : int
                预测窗口的数据点数
            """
            self.warmup_period = warmup_period
            self.prediction_window = prediction_window
            self.logger = logging.getLogger(__name__)

    def prepare_trading_sequence(self,
                                 stock_data: Dict[str, pd.DataFrame]
                                 ) -> Dict[str, List[TradingPoint]]:
        """准备交易序列"""
        trading_sequences = {}

        for stock_code, df in stock_data.items():
            try:
                sequence = []
                # 从预热期结束后开始，每隔prediction_window个点取一个交易点
                for pos in range(self.warmup_period,
                                 len(df) - self.prediction_window,
                                 self.prediction_window):
                    # 获取历史数据窗口
                    historical_data = df.iloc[pos - self.warmup_period:pos + 1].copy()
                    historical_data = historical_data.reset_index(drop=True)

                    # 获取当前价格和未来价格
                    current_price = float(df.iloc[pos]['close'])
                    future_price = float(df.iloc[pos + self.prediction_window]['close'])

                    # 创建交易点
                    trading_point = TradingPoint(
                        position=pos,
                        historical_data=historical_data,
                        current_price=current_price,
                        future_price=future_price
                    )

                    sequence.append(trading_point)

                if sequence:
                    trading_sequences[stock_code] = sequence
                    self.logger.info(f"股票 {stock_code} 生成了 {len(sequence)} 个交易点")

            except Exception as e:
                self.logger.error(f"处理股票 {stock_code} 时出错: {str(e)}")
                continue

        return trading_sequences

    def get_next_trading_point(self,
                               trading_sequences: Dict[str, List[TradingPoint]],
                               current_position: Optional[int] = None
                               ) -> Dict[str, Optional[TradingPoint]]:
        """获取下一个交易点"""
        next_points = {}

        # 如果当前位置为None，则返回每个序列的第一个点
        if current_position is None:
            for stock_code, sequence in trading_sequences.items():
                next_points[stock_code] = sequence[0] if sequence else None
            return next_points

        # 获取下一个交易点
        for stock_code, sequence in trading_sequences.items():
            if not sequence:
                next_points[stock_code] = None
                continue

            try:
                # 获取当前点的索引
                current_index = next(
                    (i for i, point in enumerate(sequence)
                     if point.position == current_position),
                    None
                )

                if current_index is None:
                    next_points[stock_code] = None
                    continue

                # 获取下一个点
                next_index = current_index + 1
                if next_index < len(sequence):
                    next_points[stock_code] = sequence[next_index]
                else:
                    next_points[stock_code] = None

            except Exception as e:
                self.logger.error(f"获取股票 {stock_code} 下一个点时出错: {str(e)}")
                next_points[stock_code] = None

        return next_points


import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from base import (Factor, FactorCalculator, MomentumFactor, VolatilityFactor,
                               TrendFactor, RSIFactor, MACDFactor, ReversalFactor)
from factor_trainer import ModelManager


class PredictionManager:
    """优化后的预测管理器"""

    def __init__(self, model_path: str = "models"):
        """
        初始化预测管理器

        Parameters:
        -----------
        model_path : str
            模型存储路径
        """
        self.model_manager = ModelManager(model_path)
        self.models: Dict[str, Tuple[xgb.XGBClassifier, StandardScaler]] = {}
        # 初始化因子计算器
        self.factors: Dict[Factor, FactorCalculator] = {
            Factor.MOMENTUM: MomentumFactor(period=120),  # 10小时
            Factor.VOLATILITY: VolatilityFactor(period=240),  # 20小时
            Factor.TREND: TrendFactor(period=300),  # 25小时
            Factor.RSI: RSIFactor(period=144),  # 12小时
            Factor.MACD: MACDFactor(
                fast_period=24,  # 2小时
                slow_period=52,  # 4.3小时
                signal_period=18  # 1.5小时
            ),
            Factor.REVERSAL: ReversalFactor(period=240)  # 20小时
        }
        self.logger = logging.getLogger(__name__)

    def load_models(self, stock_codes: List[str]):
        """加载股票模型"""
        # 对股票代码进行排序
        sorted_codes = sorted(stock_codes, key=lambda x: int(x))

        for stock_code in sorted_codes:
            try:
                result = self.model_manager.load_model(stock_code)
                if result is not None:
                    model, scaler, _ = result
                    self.models[stock_code] = (model, scaler)
            except Exception as e:
                self.logger.error(f"加载股票 {stock_code} 的模型时发生错误: {str(e)}")
                continue

    def prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        准备模型预测所需的特征

        Parameters:
        -----------
        df : pd.DataFrame
            历史数据（已经过优化，只包含计算因子所需的数据点）

        Returns:
        --------
        Optional[np.ndarray]:
            特征数组，如果计算失败则返回None
        """
        try:
            # 计算因子值
            factor_values = {}
            for factor_name, calculator in self.factors.items():
                try:
                    signal = calculator.calculate(df)
                    factor_values[factor_name.value] = [signal]
                except Exception as e:
                    self.logger.error(f"计算因子 {factor_name.value} 失败: {str(e)}")
                    return None

            # 转换为DataFrame
            X = pd.DataFrame(factor_values)
            return X.values

        except Exception as e:
            self.logger.error(f"特征准备失败: {str(e)}")
            return None

    def predict(self, stock_code: str, df: pd.DataFrame, position: int) -> Optional[StockPrediction]:
        """
        对单只股票进行预测

        Parameters
        -----------
        stock_code : str
            股票代码
        df : pd.DataFrame
            历史数据
        position : int
            当前位置索引

        Returns
        --------
        Optional[StockPrediction]:
            预测结果
        """
        if stock_code not in self.models:
            self.logger.warning(f"股票 {stock_code} 的模型未加载")
            return None

        try:
            # 准备特征
            X = self.prepare_features(df)
            if X is None:
                return None

            # 获取模型和标准化器
            model, scaler = self.models[stock_code]

            # 标准化特征
            X_scaled = scaler.transform(X)

            # 获取预测结果和概率
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            confidence = probabilities[prediction]

            return StockPrediction(
                stock_code=stock_code,
                prediction=PredictionResult(prediction),
                confidence=confidence,
                position=position  # 使用position而不是timestamp
            )

        except Exception as e:
            self.logger.error(f"预测股票 {stock_code} 失败: {str(e)}")
            return None

    def predict_all(self, historical_data: Dict[str, pd.DataFrame], current_position: int) -> List[StockPrediction]:
        """
        对所有股票进行预测

        Parameters
        -----------
        historical_data : Dict[str, pd.DataFrame]
            股票历史数据字典
        current_position : int
            当前位置索引

        Returns
        --------
        List[StockPrediction]:
            所有股票的预测结果
        """
        predictions = []
        for stock_code, df in historical_data.items():
            if stock_code in self.models:
                pred = self.predict(stock_code, df, current_position)
                if pred is not None:
                    predictions.append(pred)
                    self.logger.info(
                        f"股票 {stock_code} 预测结果: {pred.prediction.name}, "
                        f"置信度: {pred.confidence:.4f}, 位置: {pred.position}"
                    )
            else:
                self.logger.warning(f"股票 {stock_code} 没有可用的模型")

        self.logger.info(f"共完成 {len(predictions)} 只股票的预测")
        if len(predictions) > 0:
            up_predictions = [p for p in predictions if
                              p.prediction in [PredictionResult.UP, PredictionResult.EXTREME_UP]]
            self.logger.info(f"其中看涨股票数量: {len(up_predictions)}")

        return predictions

class TradeStatus(Enum):
    """交易状态枚举"""
    PENDING = "pending"  # 等待执行
    EXECUTED = "executed"  # 已执行
    FAILED = "failed"  # 执行失败
    CANCELLED = "cancelled"  # 已取消


@dataclass
class TradeOrder:
    """交易订单数据结构"""
    stock_code: str
    action: str  # "buy" or "sell"
    shares: int
    price: float
    status: TradeStatus
    position: int  # 使用position替代timestamp
    prediction: Optional[PredictionResult] = None
    confidence: Optional[float] = None

@dataclass
class Position:
    """持仓信息数据结构"""
    stock_code: str
    shares: int
    entry_price: float
    entry_position: int  # 替换entry_time为entry_position
    prediction: PredictionResult
    confidence: float

class PortfolioManager:
    """投资组合管理器"""

    def __init__(self,
                 initial_capital: float,
                 extreme_up_weight: float = 1.5,
                 up_weight: float = 1.0,
                 max_single_position: float = 0.2,
                 max_total_position: float = 0.8):
        """
        初始化投资组合管理器

        Parameters
        ----------
        initial_capital : float
            初始资金
        extreme_up_weight : float, optional
            极端上涨的权重, by default 1.5
        up_weight : float, optional
            上涨的权重, by default 1.0
        max_single_position : float, optional
            单个股票最大仓位比例, by default 0.2
        max_total_position : float, optional
            最大总仓位比例, by default 0.8
        """
        # 初始化资金和权重参数
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.extreme_up_weight = extreme_up_weight
        self.up_weight = up_weight
        self.max_single_position = max_single_position
        self.max_total_position = max_total_position

        # 初始化其他属性
        self.positions: Dict[str, Position] = {}
        self.current_prices = {}
        self.daily_returns = []
        self.logger = logging.getLogger(__name__)

    def update_current_prices(self, prices: Dict[str, float]):
        """
        更新当前价格

        Parameters
        ----------
        prices : Dict[str, float]
            股票当前价格字典，key为股票代码，value为价格
        """
        self.current_prices = prices

    def calculate_position_sizes(self,
                                 predictions: List[StockPrediction],
                                 current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        计算目标持仓规模

        Parameters
        ----------
        predictions : List[StockPrediction]
            预测结果列表
        current_prices : Dict[str, float]
            当前价格字典

        Returns
        -------
        Dict[str, float]
            每只股票的目标持仓金额
        """
        # 更新当前价格
        self.update_current_prices(current_prices)

        # 筛选看涨的股票
        bullish_predictions = [p for p in predictions
                               if p.prediction in [PredictionResult.UP, PredictionResult.EXTREME_UP]]

        if not bullish_predictions:
            return {}

        # 计算权重
        weights = []
        for pred in bullish_predictions:
            base_weight = (self.extreme_up_weight
                           if pred.prediction == PredictionResult.EXTREME_UP
                           else self.up_weight)
            weight = base_weight * pred.confidence
            weights.append(weight)

        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # 计算可用资金
        available_capital = self.current_capital * self.max_total_position

        # 分配资金
        position_sizes = {}
        for pred, weight in zip(bullish_predictions, weights):
            # 计算目标金额
            target_amount = min(
                available_capital * weight,
                self.current_capital * self.max_single_position
            )
            position_sizes[pred.stock_code] = target_amount

        return position_sizes

    def update_positions(self,
                         buys: Dict[str, Tuple[int, float]],
                         sells: Dict[str, Tuple[int, float]],
                         predictions: List[StockPrediction]):
        """
        更新持仓信息

        Parameters
        ----------
        buys : Dict[str, Tuple[int, float]]
            买入信息，key为股票代码，value为(股数, 价格)的元组
        sells : Dict[str, Tuple[int, float]]
            卖出信息，key为股票代码，value为(股数, 价格)的元组
        predictions : List[StockPrediction]
            预测结果列表
        """
        # 记录当日收益
        daily_pnl = 0

        # 处理卖出
        for stock_code, (shares, price) in sells.items():
            if stock_code in self.positions:
                pos = self.positions[stock_code]
                realized_pnl = shares * (price - pos.entry_price)
                daily_pnl += realized_pnl
                self.current_capital += shares * price
                if shares >= pos.shares:
                    del self.positions[stock_code]
                else:
                    pos.shares -= shares

        # 处理买入
        for stock_code, (shares, price) in buys.items():
            pred = next((p for p in predictions if p.stock_code == stock_code), None)
            if pred is None:
                continue

            position = Position(
                stock_code=stock_code,
                shares=shares,
                entry_price=price,
                entry_position=pred.position,
                prediction=pred.prediction,
                confidence=pred.confidence
            )
            self.positions[stock_code] = position
            self.current_capital -= shares * price

        # 记录收益率
        if self.initial_capital > 0:
            daily_return = daily_pnl / self.initial_capital
            self.daily_returns.append(daily_return)

    def get_position_ratio(self) -> float:
        """
        计算当前持仓比例

        Returns
        -------
        float
            当前持仓比例，范围[0, 1]
        """
        if not self.positions:
            return 0.0

        total_position_value = 0.0
        for stock_code, pos in self.positions.items():
            price = self.current_prices.get(stock_code, pos.entry_price)
            total_position_value += pos.shares * price

        return min(max(total_position_value / self.initial_capital, 0), 1)

    def get_position_summary(self) -> Dict:
        """
        获取持仓摘要信息

        Returns
        -------
        Dict
            持仓摘要信息字典
        """
        return {
            'total_capital': self.current_capital,
            'position_ratio': self.get_position_ratio() * 100,  # 转换为百分比
            'cumulative_return': (self.current_capital / self.initial_capital - 1) * 100,
            'daily_returns_mean': np.mean(self.daily_returns) * 100 if self.daily_returns else 0,
            'daily_returns_std': np.std(self.daily_returns) * 100 if self.daily_returns else 0,
            'positions': [
                {
                    'stock_code': pos.stock_code,
                    'shares': pos.shares,
                    'entry_price': pos.entry_price,
                    'current_price': self.current_prices.get(pos.stock_code, pos.entry_price),
                    'prediction': pos.prediction.name,
                    'confidence': pos.confidence,
                    'entry_position': pos.entry_position,
                    'unrealized_pnl': (self.current_prices.get(pos.stock_code,
                                                               pos.entry_price) - pos.entry_price) * pos.shares
                }
                for pos in self.positions.values()
            ]
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        计算最大回撤

        Parameters:
        -----------
        returns : np.ndarray
            收益率数组

        Returns:
        --------
        float:
            最大回撤比例
        """
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown) if len(drawdown) > 0 else 0


from data_processor import process_raw_data, filter_stocks


class TradingSystem:
    def __init__(self, data_file: str, initial_capital: float,
                 model_path: str = "models",
                 extreme_up_weight: float = 1.5,
                 up_weight: float = 1.0,
                 max_single_position: float = 0.2,
                 max_total_position: float = 0.8):
        """
        初始化交易系统
        """
        self.logger = logging.getLogger(__name__)
        self.data_file = data_file
        self.prediction_manager = PredictionManager(model_path)
        self.portfolio_manager = PortfolioManager(
            initial_capital=initial_capital,
            extreme_up_weight=extreme_up_weight,
            up_weight=up_weight,
            max_single_position=max_single_position,
            max_total_position=max_total_position
        )
        self.data_manager = TradingDataManager()
        self.trading_sequences = {}
        self.current_position = None
        self.trade_history = []

    def initialize(self) -> bool:
        """初始化交易系统"""
        try:
            # 读取数据
            df = pd.read_excel(self.data_file, header=[0, 1]) if self.data_file.endswith('.xlsx') \
                else pd.read_csv(self.data_file, header=[0, 1])
            #num_rows = len(df) 
            #thirty_percent_rows = int(num_rows * 0.9)
            #df = df.iloc[int(num_rows * 0.5):thirty_percent_rows]

            # 打印原始数据信息
            self.logger.info("原始数据信息：")
            self.logger.info(f"索引类型: {type(df.index)}")
            self.logger.info(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
            self.logger.info(f"数据形状: {df.shape}")
            self.logger.info(f"列名: {df.columns.tolist()[:10]}...")

            # 处理数据并准备交易序列
            raw_data = process_raw_data(df)

            # 打印处理后的数据信息
            for stock_code, stock_df in raw_data.items():
                self.logger.info(f"\n股票 {stock_code} 处理后数据信息：")
                self.logger.info(f"时间范围: {stock_df.index[0]} 到 {stock_df.index[-1]}")
                self.logger.info(f"数据点数: {len(stock_df)}")
                self.logger.info(f"列名: {stock_df.columns.tolist()}")
                break  # 只打印第一只股票的信息

            processed_data = filter_stocks(raw_data)
            self.trading_sequences = self.data_manager.prepare_trading_sequence(processed_data)

            # 打印交易序列信息
            for stock_code, sequence in list(self.trading_sequences.items())[:1]:  # 只打印第一只股票的信息
                self.logger.info(f"\n股票 {stock_code} 交易序列信息：")
                self.logger.info(f"序列长度: {len(sequence)}")

                # 修改这部分，使用position而不是timestamp
                if sequence:
                    first_point = sequence[0]
                    self.logger.info(
                        f"第一个点数据范围: 从位置 {first_point.position - self.data_manager.warmup_period} "
                        f"到 {first_point.position}"
                    )
                    self.logger.info(f"历史数据形状: {first_point.historical_data.shape}")
                    self.logger.info(f"历史数据列名: {first_point.historical_data.columns.tolist()}")

            # 加载模型
            stock_codes = list(self.trading_sequences.keys())
            self.prediction_manager.load_models(stock_codes)

            return True

        except Exception as e:
            self.logger.error(f"交易系统初始化失败: {str(e)}")
            return False

    def _log_time_point_start(self, time_point_count: int, total_points: int):
        """记录时间点开始的日志"""
        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"开始处理时间点 [{time_point_count}/{total_points}] "
                         f"{self.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'=' * 50}")

    def _log_predictions(self, predictions: List[StockPrediction]):
        """记录预测结果"""
        self.logger.info("\n预测结果汇总:")
        self.logger.info(f"总计预测股票数量: {len(predictions)}")

        bullish_predictions = [p for p in predictions
                               if p.prediction in [PredictionResult.UP, PredictionResult.EXTREME_UP]]
        self.logger.info(f"看涨股票数量: {len(bullish_predictions)}")

        if predictions:
            self.logger.info("\n各股票预测详情:")
            for pred in sorted(predictions, key=lambda x: int(x.stock_code)):
                self.logger.info(
                    f"股票 {pred.stock_code:>3}: {pred.prediction.name:<11} "
                    f"置信度: {pred.confidence:.4f}")

    def _log_trading_execution(self, orders: List[TradeOrder]):
        """记录交易执行情况"""
        if not orders:
            self.logger.info("\n本交易点无交易执行")
            return

        self.logger.info("\n交易执行情况:")
        by_action = {"buy": [], "sell": []}
        for order in orders:
            by_action[order.action].append(order)

        if by_action["sell"]:
            self.logger.info("\n平仓交易:")
            for order in sorted(by_action["sell"], key=lambda x: int(x.stock_code)):
                self.logger.info(
                    f"股票 {order.stock_code:>3} 平仓: {order.shares:>6}股, "
                    f"价格: {order.price:.2f}, 金额: {order.shares * order.price:,.2f}")

        if by_action["buy"]:
            self.logger.info("\n开仓交易:")
            for order in sorted(by_action["buy"], key=lambda x: int(x.stock_code)):
                self.logger.info(
                    f"股票 {order.stock_code:>3} 开仓: {order.shares:>6}股, "
                    f"价格: {order.price:.2f}, 金额: {order.shares * order.price:,.2f}")

    def _log_time_point_summary(self, summary: Dict):
        """记录交易点总结"""
        self.logger.info("\n当前交易点总结:")
        self.logger.info(f"位置: {self.current_position}")  # 使用position替代timestamp
        self.logger.info(f"总资产: {summary['total_capital']:,.2f}")
        self.logger.info(f"持仓比例: {summary['position_ratio'] * 100:.2f}%")

        if summary['positions']:
            self.logger.info("\n当前持仓:")
            for pos in sorted(summary['positions'], key=lambda x: int(x['stock_code'])):
                self.logger.info(
                    f"股票 {pos['stock_code']:>3}: {pos['shares']:>6}股, "
                    f"成本价: {pos['entry_price']:.2f}, "
                    f"当前盈亏: {pos.get('unrealized_pnl', 0):,.2f}")

    def _execute_trades(self,
                        target_positions: Dict[str, float],
                        current_prices: Dict[str, float],
                        exit_prices: Dict[str, float],
                        predictions: List[StockPrediction],
                        force_clear: bool = False) -> List[TradeOrder]:
        """
        执行交易

        Parameters:
        -----------
        target_positions : Dict[str, float]
            目标仓位
        current_prices : Dict[str, float]
            当前价格
        exit_prices : Dict[str, float]
            平仓价格
        predictions : List[StockPrediction]
            预测结果
        force_clear : bool
            是否强制清仓（用于最后一个交易时间点）

        Returns:
        --------
        List[TradeOrder]:
            交易订单列表
        """
        orders = []
        current_positions = self.portfolio_manager.positions

        # 处理平仓
        for stock_code, position in list(current_positions.items()):  # 使用list复制防止遍历时修改
            if position.shares > 0:
                # 如果是强制清仓或到达平仓时间点
                if force_clear or stock_code in exit_prices:
                    exit_price = current_prices.get(stock_code) or exit_prices.get(stock_code)
                    if exit_price:
                        order = TradeOrder(
                            stock_code=stock_code,
                            action="sell",
                            shares=position.shares,
                            price=exit_price,
                            status=TradeStatus.EXECUTED,
                            timestamp=self.current_time
                        )
                        orders.append(order)
                        self.logger.info(
                            f"股票 {stock_code} {'强制清仓' if force_clear else '达到平仓时间'}，"
                            f"以 {exit_price:.2f} 价格平仓 {position.shares} 股"
                        )

        # 如果不是强制清仓，则处理开仓
        if not force_clear:
            for stock_code, target_amount in target_positions.items():
                if stock_code in current_prices:
                    price = current_prices[stock_code]
                    shares = int(target_amount / price)
                    if shares > 0:
                        pred = next((p for p in predictions if p.stock_code == stock_code), None)
                        order = TradeOrder(
                            stock_code=stock_code,
                            action="buy",
                            shares=shares,
                            price=price,
                            status=TradeStatus.EXECUTED,
                            timestamp=self.current_time,
                            prediction=pred.prediction if pred else None,
                            confidence=pred.confidence if pred else None
                        )
                        orders.append(order)
                        self.logger.info(f"股票 {stock_code} 开仓，以 {price:.2f} 价格买入 {shares} 股")

        return orders

    def execute_trading(self) -> Dict:
        """执行交易的主函数"""
        try:
            all_summaries = []

            # 获取交易点数量
            first_stock_code = next(iter(self.trading_sequences))
            total_points = len(self.trading_sequences[first_stock_code])
            point_count = 0

            # 获取第一个交易点
            current_points = self.data_manager.get_next_trading_point(self.trading_sequences)
            if not current_points:
                return {"error": "没有可交易的时间点"}

            first_point = next(iter(current_points.values()))
            self.current_position = first_point.position

            while True:
                point_count += 1
                self.logger.info(f"\n=== 处理交易点 [{point_count}/{total_points}] ===")

                # 获取当前点的数据
                historical_data = {
                    stock_code: point.historical_data
                    for stock_code, point in current_points.items()
                    if point is not None
                }
                current_prices = {
                    stock_code: point.current_price
                    for stock_code, point in current_points.items()
                    if point is not None
                }
                exit_prices = {
                    stock_code: point.future_price
                    for stock_code, point in current_points.items()
                    if point is not None
                }

                # 更新当前价格 - 在这里添加
                self.portfolio_manager.update_current_prices(current_prices)

                # 判断是否是最后一个交易点
                is_last_point = (point_count >= total_points)

                # 执行平仓操作
                self.logger.info("\n=== 执行平仓操作 ===")
                close_orders = self._execute_close_positions(
                    current_prices, exit_prices, force_clear=is_last_point)

                if close_orders:
                    self._log_trading_execution(close_orders)
                    self._update_portfolio(close_orders, [])

                # 如果不是最后一个点，执行新的预测和开仓
                if not is_last_point:
                    # 执行预测
                    predictions = self.prediction_manager.predict_all(
                        historical_data,
                        self.current_position
                    )
                    self._log_predictions(predictions)

                    # 计算目标仓位
                    target_positions = self.portfolio_manager.calculate_position_sizes(
                        predictions, current_prices)

                    # 执行开仓操作
                    open_orders = self._execute_open_positions(
                        target_positions, current_prices, predictions)

                    if open_orders:
                        self._log_trading_execution(open_orders)
                        self._update_portfolio(open_orders, predictions)
                else:
                    self.logger.info("\n=== 最终清仓完成 ===")

                # 生成并记录当前时间点的交易摘要
                current_summary = self.get_trading_summary()
                self._log_time_point_summary(current_summary)
                all_summaries.append(current_summary)

                if is_last_point:
                    break

                # 获取下一个交易点
                next_points = self.data_manager.get_next_trading_point(
                    self.trading_sequences, self.current_position)

                next_point = next(point for point in next_points.values()
                                  if point is not None)
                self.current_position = next_point.position
                current_points = next_points

            # 返回交易结果
            final_metrics = self.get_performance_metrics()
            self.logger.info("最终交易指标:")
            for key, value in final_metrics.items():
                self.logger.info(f"{key}: {value}")

            return {
                "summaries": all_summaries,
                "final_metrics": final_metrics  # 确保这里有值
            }

        except Exception as e:
            self.logger.error(f"交易执行失败: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def _execute_open_positions(self,
                                target_positions: Dict[str, float],
                                current_prices: Dict[str, float],
                                predictions: List[StockPrediction]) -> List[TradeOrder]:
        """
        执行开仓操作

        Parameters
        -----------
        target_positions : Dict[str, float]
            目标仓位
        current_prices : Dict[str, float]
            当前价格
        predictions : List[StockPrediction]
            预测结果

        Returns
        --------
        List[TradeOrder]:
            开仓订单列表
        """
        open_orders = []

        for stock_code, target_amount in target_positions.items():
            if stock_code in current_prices:
                price = current_prices[stock_code]
                shares = int(target_amount / price)
                if shares > 0:
                    pred = next((p for p in predictions if p.stock_code == stock_code), None)
                    order = TradeOrder(
                        stock_code=stock_code,
                        action="buy",
                        shares=shares,
                        price=price,
                        status=TradeStatus.EXECUTED,
                        position=self.current_position,  # 使用position替代timestamp
                        prediction=pred.prediction if pred else None,
                        confidence=pred.confidence if pred else None
                    )
                    open_orders.append(order)
                    self.logger.info(f"股票 {stock_code} 开仓，以 {price:.2f} 价格买入 {shares} 股")

        return open_orders

    def _execute_close_positions(self,
                                 current_prices: Dict[str, float],
                                 exit_prices: Dict[str, float],
                                 force_clear: bool = False) -> List[TradeOrder]:
        """
        执行平仓操作

        Parameters
        -----------
        current_prices : Dict[str, float]
            当前价格
        exit_prices : Dict[str, float]
            平仓价格
        force_clear : bool
            是否强制清仓

        Returns
        --------
        List[TradeOrder]:
            平仓订单列表
        """
        close_orders = []
        current_positions = self.portfolio_manager.positions

        for stock_code, position in list(current_positions.items()):
            if position.shares > 0:
                if force_clear or stock_code in exit_prices:
                    exit_price = current_prices.get(stock_code) or exit_prices.get(stock_code)
                    if exit_price:
                        order = TradeOrder(
                            stock_code=stock_code,
                            action="sell",
                            shares=position.shares,
                            price=exit_price,
                            status=TradeStatus.EXECUTED,
                            position=self.current_position,  # 使用position替代timestamp
                            prediction=None,
                            confidence=None
                        )
                        close_orders.append(order)
                        self.logger.info(
                            f"股票 {stock_code} {'强制清仓' if force_clear else '达到平仓时间'}，"
                            f"以 {exit_price:.2f} 价格平仓 {position.shares} 股"
                        )

        return close_orders

    def _update_portfolio(self, orders: List[TradeOrder],
                          predictions: List[StockPrediction]):
        """
        更新投资组合

        Parameters:
        -----------
        orders : List[TradeOrder]
            交易订单列表
        predictions : List[StockPrediction]
            预测结果列表
        """
        buys = {}
        sells = {}

        for order in orders:
            if order.status != TradeStatus.EXECUTED:
                continue

            if order.action == "buy":
                buys[order.stock_code] = (order.shares, order.price)
            elif order.action == "sell":
                sells[order.stock_code] = (order.shares, order.price)

        self.portfolio_manager.update_positions(buys, sells, predictions)
        self.trade_history.extend(orders)

    def get_trading_summary(self) -> Dict:
        """获取当前交易点的交易摘要"""
        summary = self.portfolio_manager.get_position_summary()

        # 只获取当前位置的交易
        current_trades = [order for order in self.trade_history
                          if order.position == self.current_position]

        summary['trades'] = {
            'position': self.current_position,  # 使用position替代timestamp
            'total_trades': len(current_trades),
            'buy_trades': len([t for t in current_trades if t.action == "buy"]),
            'sell_trades': len([t for t in current_trades if t.action == "sell"]),
            'trades_detail': [
                {
                    'stock_code': t.stock_code,
                    'action': t.action,
                    'shares': t.shares,
                    'price': t.price,
                    'amount': t.shares * t.price,
                    'prediction': t.prediction.name if t.prediction else None,
                    'confidence': t.confidence if t.confidence else None
                }
                for t in current_trades
            ]
        }

        return summary

    def get_performance_metrics(self) -> Dict:
        """获取绩效指标"""
        metrics = {
        "总交易次数": len(self.trade_history),
        "买入交易次数": sum(1 for t in self.trade_history if t.action == "buy"),
        "卖出交易次数": sum(1 for t in self.trade_history if t.action == "sell"),
        "累计收益率": (self.portfolio_manager.current_capital / self.portfolio_manager.initial_capital - 1) * 100,
        }

        # 计算每个交易点的收益率
        point_returns = self.portfolio_manager.daily_returns
        if point_returns:
            returns_array = np.array(point_returns)
            metrics.update({
                "平均点收益率": f"{np.mean(returns_array) * 100:.4f}%",
                "收益率标准差": f"{np.std(returns_array) * 100:.4f}%",
                "夏普比率": f"{np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0:.4f}",
                "最大回撤": f"{self.portfolio_manager._calculate_max_drawdown(returns_array) * 100:.2f}%",
            })

        # 重新计算胜率，需要记录每笔交易的买入价和卖出价
        if self.trade_history:
            # 创建一个字典来跟踪每只股票的买入价格
            entry_prices = {}
            profitable_trades = 0
            total_closed_trades = 0

            # 按时间顺序处理交易
            for trade in self.trade_history:
                stock_code = trade.stock_code

                if trade.action == "buy":
                    # 记录买入价格
                    entry_prices[stock_code] = trade.price
                elif trade.action == "sell" and stock_code in entry_prices:
                    # 计算这笔交易是否盈利
                    entry_price = entry_prices[stock_code]
                    if trade.price > entry_price:
                        profitable_trades += 1
                    total_closed_trades += 1
                    # 清除这只股票的买入价格记录
                    del entry_prices[stock_code]

            if total_closed_trades > 0:
                win_rate = profitable_trades / total_closed_trades
                metrics["胜率"] = f"{win_rate * 100:.2f}%"
                metrics["盈利交易数"] = profitable_trades
                metrics["总平仓交易数"] = total_closed_trades

        return metrics