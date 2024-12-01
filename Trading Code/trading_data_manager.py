from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass


@dataclass
class TradingPoint:
    """交易点数据结构"""
    position: int  # 当前位置索引
    historical_data: pd.DataFrame  # 包含因子计算所需的历史数据
    current_price: float  # 当前价格
    future_price: float  # 未来价格（用于回测）


class TradingDataManager:
    """交易数据管理器"""

    def __init__(self, warmup_period: int = 300, prediction_window: int = 49):
        """
        初始化交易数据管理器

        Parameters
        ----------
        warmup_period : int
            因子计算所需的历史数据点数
        prediction_window : int
            预测窗口的数据点数
        """
        self.warmup_period = warmup_period
        self.prediction_window = prediction_window
        self.logger = logging.getLogger(__name__)

    def prepare_trading_sequence(self,
                                 stock_data: Dict[str, pd.DataFrame]
                                 ) -> Dict[str, List[TradingPoint]]:
        """
        为每只股票准备交易序列

        Parameters
        ----------
        stock_data : Dict[str, pd.DataFrame]
            股票数据字典，key为股票代码，value为数据DataFrame

        Returns
        -------
        Dict[str, List[TradingPoint]]
            每只股票的交易点列表
        """
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
                    self.logger.info(
                        f"股票 {stock_code} 生成了 {len(sequence)} 个交易点"
                    )
                    self.logger.info(f"第一个点位置: {sequence[0].position}")
                    self.logger.info(f"最后一个点位置: {sequence[-1].position}")

            except Exception as e:
                self.logger.error(f"处理股票 {stock_code} 时出错: {str(e)}")
                continue

        return trading_sequences

    def get_next_trading_point(self,
                               trading_sequences: Dict[str, List[TradingPoint]],
                               current_position: Optional[int] = None
                               ) -> Dict[str, Optional[TradingPoint]]:
        """
        获取每只股票的下一个交易点

        Parameters
        ----------
        trading_sequences : Dict[str, List[TradingPoint]]
            所有股票的交易点序列
        current_position : Optional[int]
            当前位置索引，如果为None则返回第一个交易点

        Returns
        -------
        Dict[str, Optional[TradingPoint]]
            每只股票的下一个交易点
        """
        next_points = {}

        # 如果当前位置为None，则返回每个序列的第一个点
        if current_position is None:
            for stock_code, sequence in trading_sequences.items():
                next_points[stock_code] = sequence[0] if sequence else None
                if sequence:
                    self.logger.debug(
                        f"股票 {stock_code} 初始点位置: {sequence[0].position}"
                    )
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
                    self.logger.warning(
                        f"股票 {stock_code} 找不到当前位置 {current_position}"
                    )
                    continue

                # 获取下一个点
                next_index = current_index + 1
                if next_index < len(sequence):
                    next_points[stock_code] = sequence[next_index]
                    self.logger.debug(
                        f"股票 {stock_code} 下一个点位置: {sequence[next_index].position}"
                    )
                else:
                    next_points[stock_code] = None
                    self.logger.debug(f"股票 {stock_code} 没有下一个点")

            except Exception as e:
                self.logger.error(f"获取股票 {stock_code} 下一个点时出错: {str(e)}")
                next_points[stock_code] = None

        return next_points