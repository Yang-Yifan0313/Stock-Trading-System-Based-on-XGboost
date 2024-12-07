# Stock Trading System based on XGBoost

An intelligent stock trading system that uses XGBoost and multi-factor models to capture market opportunities in the Hong Kong stock market. The system employs systematic data processing, factor-based analysis, and a multi-layered risk control mechanism. More details can be found in Report Group 3.pdf.

## 🌟 Project Structure

The project consists of two main modules:

### 1. Model Training Module
Three core components for training the XGBoost model:

1. `factor_trainer.py`: Factor calculation and model training
   - Implements factor extraction and processing
   - Handles XGBoost model training
   - Cross-validation and performance evaluation

2. `data_processor.py`: Data preprocessing and cleaning
   - Data quality control
   - Outlier detection and handling
   - Missing value processing
   - Feature engineering

3. `return_analyzer.py`: Return analysis and performance metrics
   - Calculates various performance metrics
   - Generates performance visualizations
   - Provides statistical analysis of returns

### 2. Trading Module
Core trading system components:

1. `trading_system.py`: Main trading system implementation
   - Trading strategy execution
   - Portfolio management
   - Risk control
   - Performance monitoring

2. `trading_data_manager.py`: Data management system
   - Trading sequence preparation
   - Historical data management
   - Real-time data processing
   - Trading point generation

3. `run_trading.py`: System execution and monitoring
   - Trading system initialization
   - Trade execution management
   - Performance tracking
   - Logging and reporting

## 🌟 Features

### Multi-Factor Analysis
Each stock is analyzed using a comprehensive Factor Model, which requires training using historical data before trading. Factor modeling includes:
- Momentum Factor (10-hour period)
  - Captures price momentum over 120 data points
  - Implements trend following strategy
- RSI Factor (12-hour period)
  - Measures overbought/oversold conditions
  - Uses 144 data points for calculation
- MACD Factor
  - Fast period: 2 hours (24 points)
  - Slow period: 4.3 hours (52 points)
  - Signal period: 1.5 hours (18 points)
- Volatility Factor (20-hour period)
  - Measures price volatility over 240 points
  - Used for risk assessment
- Trend Factor (25-hour period)
  - Long-term trend identification
  - Based on 300 data points
- Reversal Factor (20-hour period)
  - Identifies potential price reversals
  - Uses 240 data points for calculation
- Volume Factor (1.5-hour period)
  - Analyzes trading volume patterns
  - Short-term liquidity assessment

### Trading Strategy
- XGBoost-based prediction model requiring pre-training:
  - Individual model for each stock
  - Time series cross-validation (n_splits=5)
  - Sample weighting based on class distribution
  - Model persistence with versioning
- Model parameters:
  ```python
  params = {
      'learning_rate': 0.05,
      'n_estimators': 500,
      'max_depth': 6,
      'subsample': 0.9,
      'colsample_bytree': 0.9,
      'objective': 'multi:softmax',
      'num_class': 5
  }
  ```
- 5-category prediction scheme:
  - EXTREME_DOWN
  - DOWN
  - NEUTRAL
  - UP
  - EXTREME_UP
- Dynamic position sizing based on prediction confidence
- Time-series cross-validation for model evaluation

### Risk Management
- Position Limits:
  - Maximum single stock position: 0.65%
  - Maximum total position: 60%
- Pre-trade Risk Controls:
  - Volume threshold validation
  - Price movement limits
  - Volatility checks
- Automated Position Management:
  - Dynamic position adjustment
  - Stop-loss implementation
  - Take-profit execution
- Real-time Monitoring:
  - Portfolio exposure tracking
  - Drawdown monitoring
  - Performance metrics calculation

### High-Frequency Data Processing
- Data Processing Windows:
  - Primary window: 1-day (49 points)
  - Warmup period: 300 points
  - Factor calculation periods: 10-25 hours
- Real-time Features:
  ```python
  features = {
      'price_features': ['VWAP', 'price_momentum', 'price_volatility'],
      'volume_features': ['volume_ratio', 'OBV', 'volume_momentum'],
      'technical_features': ['RSI', 'MACD', 'BB_width'],
      'market_features': ['market_cap', 'turnover_ratio']
  }
  ```

## 💻 System Requirements

- Python 3.9+
- Required Libraries:
  ```
  pandas>=1.3.0
  numpy>=1.21.0
  xgboost>=1.5.0
  scikit-learn>=0.24.2
  matplotlib>=3.4.2
  seaborn>=0.11.1
  holidays>=0.11.1
  ta>=0.7.0
  ```

## 📦 Installation

1. Clone the repository
```bash
git clone https://github.com/Yang-Yifan0313/Stock-Trading-System-Based-on-XGboost.git
cd Stock-Trading-System-Based-on-XGboost
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables
```bash
cp .env.example .env
# Edit .env file with your settings
```

## 🚀 Usage

1. Model Training
- Before running the trading system, you need to train models for each stock:
  ```python
  python factor_trainer.py --data_file=train.xlsx
  ```
- Models will be saved in the `models` directory with the following structure:
  ```
  models/
  ├── stock_{code}/
  │   ├── model_{timestamp}.joblib     # XGBoost model
  │   ├── scaler_{timestamp}.joblib    # StandardScaler
  │   ├── metrics_{timestamp}.json     # Performance metrics
  │   └── latest.json                  # Latest model reference
  ```
- Each stock gets its own XGBoost model with parameters:
  ```python
  model_params = {
      'learning_rate': 0.05,
      'n_estimators': 500,
      'max_depth': 6,
      'subsample': 0.9,
      'colsample_bytree': 1.0,
      'min_child_weight': 2,
      'gamma': 0.1,
      'reg_alpha': 0.1,
      'reg_lambda': 1,
      'objective': 'multi:softmax',
      'num_class': 5  # 5-category classification
  }
  ```
- Training features comprehensive metrics:
  - Overall accuracy
  - Per-class precision, recall, and F1 scores
  - Upward trend prediction performance
  - Factor importance analysis

2. Data Preparation
- Required data format:
  ```python
  Required columns = [
      'open_px', 'close_px', 'high_px', 'low_px', 'volume',
  ]
  ```
- Data frequency: 5-minute intervals
- Place data files in the `data/` directory

2. System Configuration
- Edit trading parameters in `config/trading_config.yaml`:
  ```yaml
  trading:
    prediction_window: 49  # 1 trading day
    warmup_period: 300    # Historical data points
    max_single_position: 0.0065
    max_total_position: 0.6
    extreme_up_weight: 1.5
    up_weight: 1.0
  ```

3. Run the System
```python
# Initialize and run trading system
python run_trading.py --data_file=your_data.xlsx --initial_capital=10000000
```

4. Monitor Performance
- Real-time logging output
- Performance metrics calculation
- Trading summary generation
- Visualization of results

## 🔍 Monitoring and Logging

### Logging Configuration
```python
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M")}.log',
            'level': 'DEBUG'
        }
    }
})
```

### Performance Metrics
- Trading metrics:
  - Total number of trades
  - Win rate
  - Average return per trade
  - Maximum drawdown
- Portfolio metrics:
  - Cumulative return
  - Sharp ratio
  - Position ratios
  - Risk exposure
