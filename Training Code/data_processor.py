# data_processor_copy.py
import pandas as pd
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


def process_raw_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """处理原始数据，将宽格式数据转换为每个股票的独立DataFrame"""
    stock_dict = {}

    try:
        # 获取股票列名
        stock_columns = [col for col in df.columns.levels[0]
                         if isinstance(col, str) and col.startswith('STOCK_')]
        print(f"发现股票数量: {len(stock_columns)}")

        for stock in stock_columns:
            try:
                # 构建单个股票的DataFrame
                stock_df = pd.DataFrame({
                    'close': pd.to_numeric(df[stock]['close_px'], errors='coerce'),
                    'high': pd.to_numeric(df[stock]['high_px'], errors='coerce'),
                    'low': pd.to_numeric(df[stock]['low_px'], errors='coerce'),
                    'open': pd.to_numeric(df[stock]['open_px'], errors='coerce'),
                    'volume': pd.to_numeric(df[stock]['volume'], errors='coerce')
                }, index=df.index)

                # 数据清理
                stock_df = stock_df.dropna()

                if not stock_df.empty:
                    stock_num = stock.split('_')[1]
                    stock_dict[stock_num] = stock_df

            except Exception as e:
                print(f"处理股票 {stock} 时出错: {str(e)}")
                continue

    except Exception as e:
        print(f"数据处理过程出错: {str(e)}")
        raise

    print(f"处理完成，共成功处理 {len(stock_dict)} 只股票的数据")
    return stock_dict


def filter_stocks(data_dict: Dict[str, pd.DataFrame],
                  min_volume: float = 100,  # 降低成交量要求
                  min_price: float = 1.0,
                  max_volatility: float = 0.05) -> Dict[str, pd.DataFrame]:  # 降低波动率阈值
    """筛选股票，去除不符合条件的股票数据"""
    filtered_stocks = {}

    for code, df in data_dict.items():
        try:
            # 计算筛选指标 - 使用更短的窗口
            volume_ma = df['volume'].rolling(window=3).mean()  # 降低至15分钟
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=6).std()  # 降低至30分钟

            # 应用筛选条件
            if (volume_ma.mean() >= min_volume and
                    df['close'].mean() >= min_price and
                    volatility.mean() <= max_volatility):
                filtered_stocks[code] = df

        except Exception as e:
            print(f"处理股票 {code} 时出错: {str(e)}")
            continue

    print(f"筛选后剩余股票数量: {len(filtered_stocks)}")
    return filtered_stocks