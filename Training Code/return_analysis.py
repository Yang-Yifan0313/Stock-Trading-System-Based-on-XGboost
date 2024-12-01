# return_analysis.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from data_processor import process_raw_data, filter_stocks
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ReturnAnalyzer:
    def __init__(self, prediction_window: int = 49):
        """
        Parameters:
        -----------
        prediction_window : int
            预测窗口大小，默认49（一天）
        """
        self.prediction_window = prediction_window
        self.threshold_cache = {}  # 缓存每只股票的阈值

    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """计算未来收益率"""
        future_prices = df['close'].shift(-self.prediction_window)
        current_prices = df['close']
        returns = (future_prices / current_prices - 1) * 100  # 转换为百分比
        return returns

    def get_dynamic_thresholds(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算单只股票的动态阈值

        Returns:
        --------
        Dict[str, float]: 包含不同级别的阈值
        """
        thresholds = {
            'extreme_down': np.percentile(returns, 10),  # 极端下跌阈值
            'down': np.percentile(returns, 30),  # 下跌阈值
            'up': np.percentile(returns, 70),  # 上涨阈值
            'extreme_up': np.percentile(returns, 90)  # 极端上涨阈值
        }

        return thresholds

    def get_return_class(self, return_value: float, stock_code: str = None,
                         thresholds: Dict[str, float] = None) -> int:
        """
        基于动态阈值将收益率转换为分类标签

        Parameters:
        -----------
        return_value : float
            收益率值
        stock_code : str, optional
            股票代码，用于获取缓存的阈值
        thresholds : Dict[str, float], optional
            直接提供的阈值字典

        Returns:
        --------
        int: 分类标签 (0-4)
        """
        if thresholds is None:
            if stock_code is None or stock_code not in self.threshold_cache:
                raise ValueError("必须提供thresholds或有效的stock_code")
            thresholds = self.threshold_cache[stock_code]

        if return_value <= thresholds['extreme_down']:
            return 0  # 极端下跌
        elif return_value <= thresholds['down']:
            return 1  # 下跌
        elif return_value <= thresholds['up']:
            return 2  # 震荡
        elif return_value <= thresholds['extreme_up']:
            return 3  # 上涨
        else:
            return 4  # 极端上涨

    def analyze_single_stock(self, df: pd.DataFrame, stock_code: str) -> Dict:
        """
        分析单只股票的收益率分布

        Parameters:
        -----------
        df : pd.DataFrame
            股票数据
        stock_code : str
            股票代码

        Returns:
        --------
        Dict: 包含统计信息的字典
        """
        returns = self.calculate_returns(df)
        returns = returns.dropna()

        if len(returns) == 0:
            return None

        # 计算动态阈值
        thresholds = self.get_dynamic_thresholds(returns)
        self.threshold_cache[stock_code] = thresholds

        # 基本统计量
        stats = {
            'count': len(returns),
            'mean': returns.mean(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'skew': returns.skew(),
            'kurt': returns.kurtosis(),
            'thresholds': thresholds,
            'percentiles': {
                '1%': np.percentile(returns, 1),
                '5%': np.percentile(returns, 5),
                '10%': np.percentile(returns, 10),
                '25%': np.percentile(returns, 25),
                '50%': np.percentile(returns, 50),
                '75%': np.percentile(returns, 75),
                '90%': np.percentile(returns, 90),
                '95%': np.percentile(returns, 95),
                '99%': np.percentile(returns, 99),
            }
        }

        # 使用动态阈值计算分布
        labels = ['极端下跌', '下跌', '震荡', '上涨', '极端上涨']
        class_counts = [0] * 5
        for ret in returns:
            class_idx = self.get_return_class(ret, thresholds=thresholds)
            class_counts[class_idx] += 1

        distribution = {label: count for label, count in zip(labels, class_counts)}
        distribution_pct = {label: count / len(returns) * 100
                            for label, count in distribution.items()}

        return {
            'stats': stats,
            'distribution': distribution,
            'distribution_pct': distribution_pct,
            'returns': returns
        }

    def analyze_stock_returns(self, stock_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """分析所有股票的收益率分布"""
        analysis_results = {}

        for stock_code, df in stock_dict.items():
            result = self.analyze_single_stock(df, stock_code)
            if result is not None:
                analysis_results[stock_code] = result

        return analysis_results

    def print_analysis_results(self, results: Dict[str, Dict]):
        """打印分析结果"""
        print("\n=== 股票收益率分析结果 ===")

        # 汇总所有股票的分布情况
        total_distribution = defaultdict(int)
        total_count = 0

        for stock_code, result in sorted(results.items(), key=lambda x: int(x[0])):
            print(f"\n股票代码: {stock_code}")
            stats = result['stats']

            print("\n基本统计量:")
            print(f"样本数: {stats['count']}")
            print(f"平均收益率: {stats['mean']:.2f}%")
            print(f"标准差: {stats['std']:.2f}%")
            print(f"最小值: {stats['min']:.2f}%")
            print(f"最大值: {stats['max']:.2f}%")
            print(f"偏度: {stats['skew']:.2f}")
            print(f"峰度: {stats['kurt']:.2f}")

            print("\n分位数:")
            percentiles = stats['percentiles']
            for pct, value in percentiles.items():
                print(f"{pct}: {value:.2f}%")

            print("\n动态阈值:")
            thresholds = stats['thresholds']
            for threshold_name, value in thresholds.items():
                print(f"{threshold_name}: {value:.2f}%")

            print("\n收益率分布:")
            distribution_pct = result['distribution_pct']
            for label, percentage in distribution_pct.items():
                count = result['distribution'][label]
                print(f"{label}: {percentage:.2f}% ({count}个样本)")
                total_distribution[label] += count
                total_count += count

        # 打印所有股票的综合分布情况
        print("\n=== 所有股票的综合分布情况 ===")
        for label, count in total_distribution.items():
            percentage = (count / total_count) * 100
            print(f"{label}: {percentage:.2f}% ({count}个样本)")

    def plot_return_distribution(self, returns: pd.Series, thresholds: Dict[str, float],
                                 title: str = "收益率分布"):
        """
        绘制收益率分布图，包括阈值线
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(returns, bins=50, kde=True)

        # 添加阈值线
        colors = ['red', 'orange', 'green', 'orange', 'red']
        for (name, value), color in zip(thresholds.items(), colors):
            plt.axvline(x=value, color=color, linestyle='--',
                        label=f'{name}: {value:.2f}%')

        plt.title(title)
        plt.xlabel("收益率 (%)")
        plt.ylabel("频数")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_thresholds(self, filename: str):
        """保存阈值到文件"""
        threshold_df = pd.DataFrame.from_dict(self.threshold_cache, orient='index')
        threshold_df.to_csv(filename)

    def load_thresholds(self, filename: str):
        """从文件加载阈值"""
        threshold_df = pd.read_csv(filename, index_col=0)
        self.threshold_cache = threshold_df.to_dict('index')


def main():
    # 读取数据
    print("读取数据...")
    df = pd.read_excel("train.xlsx", header=[0, 1])

    # 处理数据
    print("处理数据...")
    stock_dict = process_raw_data(df)
    filtered_stocks = filter_stocks(stock_dict)

    # 分析收益率
    print("分析收益率...")
    analyzer = ReturnAnalyzer()
    results = analyzer.analyze_stock_returns(filtered_stocks)

    # 打印结果
    analyzer.print_analysis_results(results)

    # 保存阈值
    analyzer.save_thresholds("stock_thresholds.csv")

    # 示例：绘制某只股票的分布图
    stock_code = '1'  # 选择一只股票
    if stock_code in results:
        analyzer.plot_return_distribution(
            results[stock_code]['returns'],
            results[stock_code]['stats']['thresholds'],
            f"股票 {stock_code} 收益率分布"
        )


if __name__ == "__main__":
    main()