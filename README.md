# Quantitative Finance Dashboard

A comprehensive Streamlit-based dashboard for quantitative finance analysis, featuring stock analysis, portfolio management, financial modeling, and risk assessment.

## Features

### 📊 Stock Analysis

- Real-time stock data fetching using Yahoo Finance
- Technical indicators (Moving Averages, RSI, MACD)
- Interactive candlestick charts with volume
- Key financial metrics and company information

### 💼 Portfolio Management

- Multi-asset portfolio analysis
- Portfolio optimization (Maximum Sharpe Ratio)
- Performance metrics (Return, Volatility, Sharpe Ratio, Max Drawdown)
- Interactive allocation charts and performance visualization

### 🔬 Financial Modeling

- **DCF Valuation**: Discounted Cash Flow analysis for stock valuation
- **Black-Scholes Options**: European call/put option pricing
- **Monte Carlo Simulation**: Stock price simulation with risk analysis

### ⚠️ Risk Assessment

- Value at Risk (VaR) calculations using multiple methods:
  - Historical VaR
  - Parametric VaR
  - Monte Carlo VaR
- Expected Shortfall (Conditional VaR)
- Correlation analysis and heatmaps
- Risk metrics visualization

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd quantitative-finance-dashboard
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Dependencies

- streamlit: Web app framework
- yfinance: Yahoo Finance data fetching
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- plotly: Interactive visualizations
- scipy: Scientific computing (optimization, statistics)

## Project Structure

```
quantitative-finance-dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── assets/
│   └── style.css         # Custom CSS styling
└── src/
    ├── stock_analysis.py      # Stock analysis functions
    ├── portfolio_management.py # Portfolio management functions
    ├── financial_modeling.py  # Financial modeling functions
    ├── risk_assessment.py     # Risk assessment functions
    └── utils.py               # Utility functions
```

## Features Overview

### Stock Analysis Module

- Fetch historical stock data
- Calculate technical indicators
- Visualize price action with interactive charts
- Display fundamental company information

### Portfolio Management Module

- Analyze multi-asset portfolios
- Calculate portfolio-level risk and return metrics
- Optimize portfolio weights for maximum Sharpe ratio
- Visualize portfolio allocation and performance

### Financial Modeling Module

- DCF valuation for intrinsic value calculation
- Black-Scholes model for option pricing
- Monte Carlo simulation for price forecasting

### Risk Assessment Module

- Multiple VaR calculation methods
- Expected shortfall analysis
- Asset correlation analysis
- Risk distribution visualization

## Customization

The dashboard uses custom CSS for styling. Modify `assets/style.css` to change the appearance.

## Error Handling

The application includes comprehensive error handling for:

- Invalid ticker symbols
- Network connectivity issues
- Invalid input parameters
- Calculation errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This dashboard is for educational and informational purposes only. It should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.
