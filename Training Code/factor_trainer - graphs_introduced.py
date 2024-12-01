# factor_trainer_test.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from base import Factor, FactorCalculator
from data_processor import process_raw_data, filter_stocks
from return_analysis import ReturnAnalyzer
import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ModelManager:
    def __init__(self, base_path="models"):
        """
        Parameters:
        -----------
        base_path : str
            模型保存的基础路径
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _get_model_names(self, timestamp: str) -> Dict[str, str]:
        """生成模型文件名"""
        return {
            "model": f"model_{timestamp}.joblib",
            "scaler": f"scaler_{timestamp}.joblib",
            "metrics": f"metrics_{timestamp}.json"
        }

    def save_model(self, stock_code: str, model: xgb.XGBClassifier,
                   scaler: StandardScaler, metrics: dict) -> bool:
        """
        保存模型、标准化器和指标

        Returns:
        --------
        bool: 保存是否成功
        """
        try:
            # 创建股票特定的文件夹
            stock_path = os.path.join(self.base_path, f"stock_{stock_code}")
            os.makedirs(stock_path, exist_ok=True)

            # 生成时间戳和文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_names = self._get_model_names(timestamp)

            # 保存模型
            model_path = os.path.join(stock_path, file_names["model"])
            joblib.dump(model, model_path, protocol=4)  # 使用协议4以确保兼容性

            # 保存标准化器
            scaler_path = os.path.join(stock_path, file_names["scaler"])
            joblib.dump(scaler, scaler_path, protocol=4)

            # 保存指标
            metrics_path = os.path.join(stock_path, file_names["metrics"])
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)

            # 保存最新模型的引用（使用相对路径）
            latest_path = os.path.join(stock_path, "latest.json")
            latest_info = {
                "timestamp": timestamp,
                "model_name": file_names["model"],
                "scaler_name": file_names["scaler"],
                "metrics_name": file_names["metrics"]
            }
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(latest_info, f, ensure_ascii=False, indent=4)

            return True

        except Exception as e:
            print(f"保存股票 {stock_code} 的模型失败: {str(e)}")
            return False

    def load_model(self, stock_code: str,
                   timestamp: str = None) -> Optional[Tuple[xgb.XGBClassifier, StandardScaler, dict]]:
        """
        加载模型、标准化器和指标

        Returns:
        --------
        Optional[Tuple[xgb.XGBClassifier, StandardScaler, dict]]:
            如果加载成功返回(model, scaler, metrics)，否则返回None
        """
        try:
            # 构建股票文件夹路径
            stock_path = os.path.join(self.base_path, f"stock_{stock_code}")
            if not os.path.exists(stock_path):
                print(f"股票 {stock_code} 的模型文件夹不存在")
                return None

            # 获取时间戳和文件名
            if timestamp is None:
                latest_path = os.path.join(stock_path, "latest.json")
                if not os.path.exists(latest_path):
                    print(f"股票 {stock_code} 的latest.json不存在")
                    return None

                with open(latest_path, 'r', encoding='utf-8') as f:
                    latest_info = json.load(f)
                    timestamp = latest_info["timestamp"]
                    model_name = latest_info["model_name"]
                    scaler_name = latest_info["scaler_name"]
                    metrics_name = latest_info["metrics_name"]
            else:
                file_names = self._get_model_names(timestamp)
                model_name = file_names["model"]
                scaler_name = file_names["scaler"]
                metrics_name = file_names["metrics"]

            # 构建完整文件路径
            model_path = os.path.join(stock_path, model_name)
            scaler_path = os.path.join(stock_path, scaler_name)
            metrics_path = os.path.join(stock_path, metrics_name)

            # 检查文件是否存在
            if not all(os.path.exists(p) for p in [model_path, scaler_path, metrics_path]):
                print(f"股票 {stock_code} 的某些模型文件不存在")
                return None

            # 加载文件
            print(f"正在加载股票 {stock_code} 的模型文件...")
            model = joblib.load(model_path)
            print(f"模型加载成功")

            scaler = joblib.load(scaler_path)
            print(f"标准化器加载成功")

            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            print(f"指标加载成功")

            return model, scaler, metrics

        except Exception as e:
            print(f"加载股票 {stock_code} 的模型时发生错误: {str(e)}")
            return None

    def list_models(self, stock_code: str) -> list:
        """列出某只股票的所有可用模型"""
        stock_path = os.path.join(self.base_path, f"stock_{stock_code}")
        if not os.path.exists(stock_path):
            return []

        models = []
        for file in os.listdir(stock_path):
            if file.startswith("metrics_"):
                timestamp = file.replace("metrics_", "").replace(".json", "")
                try:
                    with open(os.path.join(stock_path, file), 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    models.append({
                        "timestamp": timestamp,
                        "accuracy": metrics.get("overall_accuracy", 0),
                        "up_prediction": metrics.get("up_prediction", {})
                    })
                except Exception as e:
                    print(f"读取股票 {stock_code} 的模型信息时出错: {str(e)}")
                    continue

        return sorted(models, key=lambda x: x["timestamp"], reverse=True)

class XGBoostFactorModel:
    def __init__(self,
                 train_window: int = 120,
                 pred_window: int = 49,
                 n_splits: int = 5):
        """
        Parameters:
        -----------
        train_window : int
            训练窗口大小
        pred_window : int
            预测窗口大小
        n_splits : int
            交叉验证折数
        """
        self.train_window = train_window
        self.pred_window = pred_window
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.return_analyzer = ReturnAnalyzer(prediction_window=pred_window)

    def _prepare_data(self, df: pd.DataFrame,
                      factors: Dict[Factor, FactorCalculator],
                      stock_code: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """改进的数据准备函数，使用ReturnAnalyzer的动态阈值"""

        # 计算最大预热期
        warmup_periods = {
            'momentum': 120,  # MomentumFactor period
            'volatility': 240,  # VolatilityFactor period
            'trend': 300,  # TrendFactor period
            'rsi': 144,  # RSIFactor period
            'macd': max(24 + 52 + 18, 52),  # MACD需要考虑fast、slow和signal periods
            'reversal': 240  # ReversalFactor period
        }
        max_warmup = max(warmup_periods.values())

        # 确保数据长度足够
        if len(df) < max_warmup + self.pred_window:
            raise ValueError(f"数据长度不足，需要至少 {max_warmup + self.pred_window} 个时间点")

        # 分析该股票的收益率特征并获取动态阈值
        analysis_result = self.return_analyzer.analyze_single_stock(df, stock_code)
        if analysis_result is None:
            raise ValueError(f"股票 {stock_code} 数据分析失败")

        # 计算因子值
        factor_values = {}
        for factor_name, calculator in factors.items():
            try:
                signals = []
                # 从预热期之后开始计算有效的因子值
                for i in range(max_warmup, len(df) - self.pred_window):
                    historical_df = df.iloc[:i + 1]
                    signal = calculator.calculate(historical_df)
                    signals.append(signal)

                # 补充预热期的数据为NaN
                warmup_signals = [np.nan] * max_warmup
                factor_values[factor_name.value] = warmup_signals + signals

            except Exception as e:
                print(f"计算因子 {factor_name.value} 出错: {str(e)}")
                continue

        X = pd.DataFrame(factor_values)

        # 使用ReturnAnalyzer计算收益率和分类标签
        returns = self.return_analyzer.calculate_returns(df)
        thresholds = analysis_result['stats']['thresholds']
        y = pd.Series([self.return_analyzer.get_return_class(r, thresholds=thresholds)
                       for r in returns.iloc[:-self.pred_window]])

        # 移除预热期和包含NaN的数据
        valid_data_start = max_warmup
        X = X.iloc[valid_data_start:]
        y = y.iloc[valid_data_start:]

        # 移除任何剩余的包含NaN的数据
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        # 计算样本权重 - 基于类别分布的倒数
        class_counts = y.value_counts()
        total_samples = len(y)
        class_weights = {cls: total_samples / (count * len(class_counts))
                         for cls, count in class_counts.items()}
        sample_weights = np.array([class_weights[label] for label in y])

        # 打印分析信息
        return X.values, y.values, sample_weights

    def _train_model(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> xgb.XGBClassifier:
        """训练XGBoost模型"""
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 创建时间序列交叉验证对象
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # 创建并训练模型
        model = xgb.XGBClassifier(
            learning_rate=0.05,
            n_estimators=500,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=1.0,
            min_child_weight=2,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            objective='multi:softmax',
            num_class=5,
            eval_metric=['mlogloss', 'merror'],
            use_label_encoder=False,
            random_state=42
        )

        # 保存训练过程
        eval_results = {}
        class_names = {
            0: "极端下跌",
            1: "下跌",
            2: "震荡",
            3: "上涨",
            4: "极端上涨"
        }

        # 使用时间序列交叉验证训练模型
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            weights_train = sample_weights[train_idx]

            print(f"\nFold {fold + 1}:")
            class_dist_train = pd.Series(y_train).value_counts()
            class_dist_val = pd.Series(y_val).value_counts()

            print("\n训练集类别分布:")
            for cls in sorted(class_dist_train.index):
                print(f"{class_names[cls]}: {class_dist_train[cls]}")

            print("\n验证集类别分布:")
            for cls in sorted(class_dist_val.index):
                print(f"{class_names[cls]}: {class_dist_val[cls]}")

            # 训练模型
            model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )

        return model

    def _get_feature_importance(self, model: xgb.XGBClassifier,
                                factors: Dict[Factor, FactorCalculator]) -> Dict[str, float]:
        """获取特征重要性"""
        importance_dict = {}
        for factor_name, importance in zip(factors.keys(), model.feature_importances_):
            importance_dict[factor_name.value] = float(importance)
        return importance_dict


import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from base import Factor, FactorCalculator
from data_processor import process_raw_data, filter_stocks
from return_analysis import ReturnAnalyzer
import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 其他类和方法...

class FactorTrainer:
    def __init__(self, prediction_window: int = 49):
        self.prediction_window = prediction_window
        self.xgb_model = XGBoostFactorModel(pred_window=prediction_window)
        self.model_manager = ModelManager()

    def analyze_factor_metrics(self, model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray,
                               factors: Dict[Factor, FactorCalculator]) -> Dict[str, Dict[str, float]]:
        """计算因子指标"""
        X_scaled = self.xgb_model.scaler.transform(X)
        predictions = model.predict(X_scaled)

        # 计算混淆矩阵
        class_names = {
            0: "Extreme Down",
            1: "Down",
            2: "Neutral",
            3: "Up",
            4: "Extreme Up"
        }

        # 计算混淆矩阵
        cm = confusion_matrix(y, predictions)

        # 绘制混淆矩阵热图
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names.values(), yticklabels=class_names.values())
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix Heatmap')
        # plt.show()
        # 计算各类别指标
        class_metrics = {}
        for class_idx in range(5):
            class_mask = y == class_idx
            if np.any(class_mask):
                class_pred = predictions[class_mask]
                accuracy = np.mean(class_pred == class_idx)
                precision = np.sum((predictions == class_idx) & (y == class_idx)) / \
                            (np.sum(predictions == class_idx) + 1e-10)
                recall = np.sum((predictions == class_idx) & (y == class_idx)) / \
                         (np.sum(y == class_idx) + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

                class_metrics[class_names[class_idx]] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

        def plot_actual_vs_predicted(self, model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray):
            """绘制实际值与预测值的散点图"""
            X_scaled = self.xgb_model.scaler.transform(X)
            predictions = model.predict(X_scaled)

            # 绘制散点图
            plt.figure(figsize=(10, 6))
            plt.scatter(y, predictions, alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # 45度参考线
            plt.title('Actual vs Predicted')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.xlim([y.min() - 0.5, y.max() + 0.5])
            plt.ylim([y.min() - 0.5, y.max() + 0.5])
            plt.grid()
            plt.show()

        # 计算总体指标
        overall_accuracy = np.mean(predictions == y)

        # 计算上涨预测准确率
        up_mask = (y == 3) | (y == 4)
        pred_up_mask = (predictions == 3) | (predictions == 4)
        up_precision = np.sum((pred_up_mask) & (up_mask)) / (np.sum(pred_up_mask) + 1e-10)
        up_recall = np.sum((pred_up_mask) & (up_mask)) / (np.sum(up_mask) + 1e-10)
        up_f1 = 2 * (up_precision * up_recall) / (up_precision + up_recall + 1e-10)

        # 获取特征重要性
        feature_importance = self.xgb_model._get_feature_importance(model, factors)

        return {
            'feature_importance': feature_importance,
            'class_metrics': class_metrics,
            'overall_accuracy': overall_accuracy,
            'up_prediction': {
                'precision': up_precision,
                'recall': up_recall,
                'f1': up_f1
            }
        }

    def train(self, file_path: str, factors: Dict[Factor, FactorCalculator]) -> Dict[str, Dict]:
        """训练XGBoost模型并返回模型和指标"""
        print("\n开始训练五分类XGBoost模型...")

        # 读取数据
        print("读取数据...")
        df = pd.read_excel(file_path, header=[0, 1]) if file_path.endswith('.xlsx') else \
            pd.read_csv(file_path, header=[0, 1])

        # 处理数据
        stock_dict = process_raw_data(df)
        filtered_stocks = filter_stocks(stock_dict)

        all_metrics = {}
        sorted_stock_codes = sorted(filtered_stocks.keys(), key=lambda x: int(x))

        for stock_code in sorted_stock_codes:
            stock_data = filtered_stocks[stock_code]
            print(f"\n处理股票 {stock_code}...")

            try:
                # 准备训练数据
                X, y, sample_weights = self.xgb_model._prepare_data(stock_data, factors, stock_code)

                if len(X) == 0 or len(y) == 0:
                    print(f"股票 {stock_code} 没有有效数据")
                    continue

                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    print(f"股票 {stock_code} 的数据只包含单一类别 {unique_classes}，跳过训练")
                    continue

                try:
                    print("训练XGBoost模型...")
                    model = self.xgb_model._train_model(X, y, sample_weights)

                    # 计算指标并绘制混淆矩阵
                    metrics = self.analyze_factor_metrics(model, X, y, factors)
                    all_metrics[stock_code] = metrics

                    # 保存模型
                    self.model_manager.save_model(
                        stock_code,
                        model,
                        self.xgb_model.scaler,
                        metrics
                    )

                    print(f"\n股票 {stock_code} 的模型表现:")
                    print(f"总体准确率: {metrics['overall_accuracy']:.2%}")



                    print("\n上涨预测性能:")
                    up_pred = metrics['up_prediction']
                    print(f"精确率: {up_pred['precision']:.4f}")
                    print(f"召回率: {up_pred['recall']:.4f}")
                    print(f"F1分数: {up_pred['f1']:.4f}")

                    print("\n各类别指标:")
                    for class_name, class_metric in metrics['class_metrics'].items():
                        print(f"\n{class_name}:")
                        for metric_name, value in class_metric.items():
                            print(f"{metric_name}: {value:.4f}")

                    print("\n特征重要性:")
                    for factor, importance in metrics['feature_importance'].items():
                        print(f"{factor}: {importance:.4f}")

                except Exception as e:
                    print(f"模型训练出错: {str(e)}")
                    continue

            except Exception as e:
                print(f"处理股票 {stock_code} 时出错: {str(e)}")
                continue

        return all_metrics


if __name__ == "__main__":
    from base import (MomentumFactor, VolatilityFactor, TrendFactor,
                      RSIFactor, MACDFactor, ReversalFactor)

    # 初始化因子
    factors = {
        Factor.MOMENTUM: MomentumFactor(period=120),
        Factor.VOLATILITY: VolatilityFactor(period=240),
        Factor.TREND: TrendFactor(period=300),
        Factor.RSI: RSIFactor(period=144),
        Factor.MACD: MACDFactor(
            fast_period=24,
            slow_period=52,
            signal_period=18
        ),
        Factor.REVERSAL: ReversalFactor(period=240)
    }

    # 创建训练器
    trainer = FactorTrainer(prediction_window=49)

    # 训练模型
    metrics = trainer.train("train.xlsx", factors)