import pandas as pd
import numpy as np
import streamlit as st

def format_currency(value):
    """Format number as currency."""
    try:
        return f"${value:,.2f}"
    except:
        return str(value)

def format_percentage(value):
    """Format number as percentage."""
    try:
        return f"{value:.2f}%"
    except:
        return str(value)

def calculate_returns(data, method='simple'):
    """Calculate returns from price data."""
    if method == 'simple':
        returns = data.pct_change()
    elif method == 'log':
        returns = np.log(data / data.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")
    return returns.dropna()

def annualize_returns(returns, periods_per_year=252):
    """Annualize returns."""
    compounded_growth = (1 + returns).prod()
    n_periods = len(returns)
    annualized_return = compounded_growth ** (periods_per_year / n_periods) - 1
    return annualized_return

def annualize_volatility(returns, periods_per_year=252):
    """Annualize volatility."""
    return returns.std() * np.sqrt(periods_per_year)

def validate_date_range(start_date, end_date):
    """Validate date range."""
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return False
    if end_date > pd.Timestamp.today().date():
        st.error("End date cannot be in the future")
        return False
    return True

def validate_tickers(tickers):
    """Basic validation for ticker symbols."""
    if not tickers:
        st.error("Please enter at least one ticker symbol")
        return False
    for ticker in tickers:
        if not ticker.strip():
            st.error("Ticker symbols cannot be empty")
            return False
    return True

def safe_divide(a, b, default=0):
    """Safe division to avoid division by zero."""
    try:
        return a / b if b != 0 else default
    except:
        return default

def calculate_cagr(start_value, end_value, periods):
    """Calculate Compound Annual Growth Rate."""
    try:
        cagr = (end_value / start_value) ** (1 / periods) - 1
        return cagr
    except:
        return 0