from datetime import datetime
import logging

import numpy as np

from trading_system import TradingSystem
import json
import os
import matplotlib.pyplot as plt
import pandas as pd


def setup_logging():
    """设置日志配置"""
    # 创建logs目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 生成日志文件名（使用当前时间）
    log_filename = f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M")}.log'

    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    return logging.getLogger(__name__)


def save_results(results: dict, filename: str):
    """保存交易结果到文件"""
    # 获取logger
    logger = logging.getLogger(__name__)

    try:
        # 添加检查
        if "final_metrics" not in results:
            logger.warning("结果中缺少final_metrics，添加空字典")
            results["final_metrics"] = {}

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_converted = json.loads(
            json.dumps(results, default=convert_numpy)
        )

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, ensure_ascii=False, indent=4)
        logger.info(f"结果已保存至: {filename}")
    except Exception as e:
        logger.error(f"保存结果时出错: {str(e)}")


def print_trading_summary(summary: dict):
    """
    打印交易摘要

    Parameters:
    -----------
    summary : dict
        交易摘要数据
    """
    print("\n=== 当日交易摘要 ===")
    print(f"总资产: {summary['total_capital']:,.2f}")
    print(f"当前持仓比例: {summary['position_ratio'] * 100:.2f}%")
    print(f"今日交易数: {summary['trades']['total_trades']}")

    if summary['trades']['trades_detail']:
        print("\n交易详情:")
        for trade in summary['trades']['trades_detail']:
            action = "买入" if trade['action'] == "buy" else "卖出"
            print(f"股票 {trade['stock_code']}: {action} {trade['shares']}股, "
                  f"价格: {trade['price']:.2f}, 金额: {trade['amount']:,.2f}")


def print_performance_metrics(metrics: dict):
    """
    打印绩效指标

    Parameters:
    -----------
    metrics : dict
        绩效指标数据
    """
    print("\n=== 绩效指标 ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")


def calculate_performance_metrics(trade_history):
    """计算累计收益率、回撤率和日收益率"""
    cumulative_returns = []
    daily_returns = []

    total_capital = 10000000  # 初始资本
    previous_capital = total_capital

    for trade in trade_history:
        # 假设 TradeOrder 类有属性 action, price, shares
        if trade.action == 'buy':
            total_capital -= trade.price * trade.shares
        else:  # trade.action == 'sell'
            total_capital += trade.price * trade.shares

        # 计算每日收益率
        daily_return = (total_capital - previous_capital) / previous_capital if previous_capital else 0
        daily_returns.append(daily_return)
        previous_capital = total_capital

        # 更新累计收益率
        cumulative_return = (total_capital - 10000000) / 10000000  # 相对初始资本
        cumulative_returns.append(cumulative_return)

    return cumulative_returns, daily_returns



def plot_performance(cumulative_returns, daily_returns):
    """绘制绩效图表"""
    plt.figure(figsize=(12, 8))

    # 累计收益率
    plt.subplot(3, 1, 1)
    plt.plot(cumulative_returns, color='blue')
    plt.title('Cumulative Returns')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Return')
    plt.grid()

    # 回撤率
    plt.subplot(3, 1, 2)
    if cumulative_returns:  # 检查是否有数据
        max_so_far = 0
        drawdowns = []
        for i in range(len(cumulative_returns)):
            max_so_far = max(max_so_far, cumulative_returns[i])
            drawdown = max_so_far - cumulative_returns[i]
            drawdowns.append(drawdown)
        plt.plot(drawdowns, color='red')
        plt.title('Drawdowns')
    else:
        plt.title('Drawdowns')
        plt.ylabel('Drawdown')
        plt.text(0.5, 0.5, 'No data available.', horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.xlabel('Trade Number')
    plt.grid()

    # 日收益率
    plt.subplot(3, 1, 3)
    plt.plot(daily_returns, color='orange')
    plt.title('Daily Returns')
    plt.xlabel('Trade Number')
    plt.ylabel('Daily Return')
    plt.grid()

    plt.tight_layout()
    plt.show()

def main():
    # 设置日志
    logger = setup_logging()

    try:
        # 交易系统参数
        params = {
            'data_file': 'test.xlsx',
            'initial_capital': 10000000,
            'model_path': 'models',
            'extreme_up_weight': 1.5,
            'up_weight': 1.0,
            'max_single_position': 0.0065,
            'max_total_position': 0.6
        }

        # 初始化交易系统
        logger.info("初始化交易系统...")
        trading_system = TradingSystem(**params)

        # 系统初始化
        if not trading_system.initialize():
            logger.error("交易系统初始化失败")
            return

        # 执行交易
        logger.info("开始执行交易策略...")
        results = trading_system.execute_trading()

        # 计算绩效指标
        cumulative_returns, daily_returns = calculate_performance_metrics(trading_system.trade_history)

        # 绘制绩效图表
        plot_performance(cumulative_returns, daily_returns)

        summary = {
            "总交易次数": len(trading_system.trade_history),
            "买入交易次数": sum(1 for order in trading_system.trade_history if order.action == "buy"),
            "卖出交易次数": sum(1 for order in trading_system.trade_history if order.action == "sell"),
        }

        # 更新 results 的 final_metrics
        if "final_metrics" not in results:
            results["final_metrics"] = {}

        # 检查结果是否包含错误
        if "error" in results:
            logger.error(f"交易执行失败: {results['error']}")
            return

        # 检查final_metrics是否存在
        if "final_metrics" not in results:
            logger.error("结果中缺少final_metrics")
            return

        metrics = results["final_metrics"]

        # 打印最终的交易统计和绩效指标
        print("\n=== 交易统计总结 ===")
        print(f"可用指标: {list(metrics.keys())}")  # 打印所有可用的指标名称

        # 使用get方法安全获取指标值
        if "总交易次数" not in metrics:
            logger.warning("无法获取交易次数指标，交易历史可能为空")

        print(f"Total number of transactions: {metrics.get('总交易次数', '未知')}")
        print(f"Number of buy trades: {metrics.get('买入交易次数', '未知')}")
        print(f"Number of sell trades: {metrics.get('卖出交易次数', '未知')}")
        print(f"Cumulative return: {metrics.get('累计收益率', '未知')}")

        if '胜率' in metrics:
            print(f"Winning percentage: {metrics['胜率']}")

        if '夏普比率' in metrics:
            print(f"Sharpe rate: {metrics['夏普比率']}")
            print(f"Maximum drawdown: {metrics.get('最大回撤', '未知')}")

        # 保存结果前先检查
        if os.path.exists('results'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            results_filename = f'results/trading_results_{timestamp}.json'
            save_results(results, results_filename)
        else:
            logger.error("results目录不存在")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()