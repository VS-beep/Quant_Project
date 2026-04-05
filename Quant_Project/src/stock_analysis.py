import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_moving_averages(data, short_window=50, long_window=200):
    """Calculate short and long term moving averages."""
    data = data.copy()
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    return data

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    data = data.copy()
    data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
    data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    data['MACD_signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    return data

def plot_stock_chart(data, ticker):
    """Create interactive stock price chart with indicators."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI'),
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.05)

    # Enhanced candlestick chart with green/red colors
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='#22c55e',  # Green for bullish
        decreasing_line_color='#dc2626',  # Red for bearish
        increasing_fillcolor='#22c55e',
        decreasing_fillcolor='#dc2626'
    ), row=1, col=1)

    # Enhanced moving averages
    if 'SMA_short' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_short'],
            name='SMA 50',
            line=dict(color='#60a5fa', width=2)
        ), row=1, col=1)

    if 'SMA_long' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_long'],
            name='SMA 200',
            line=dict(color='#ea580c', width=2)
        ), row=1, col=1)

    # Enhanced volume bars with color coding
    colors = ['#22c55e' if close > open else '#dc2626'
              for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.7,
        hovertemplate='<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)

    # Enhanced RSI with better styling
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='#a78bfa', width=2)
        ), row=3, col=1)

        # Overbought/Oversold levels with better styling
        fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", line_width=1,
                      annotation_text="Overbought", annotation_position="top right", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", line_width=1,
                      annotation_text="Oversold", annotation_position="bottom right", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#6b7280", line_width=1, row=3, col=1)

    # Enhanced layout with dark theme
    fig.update_layout(
        title=dict(
            text=f'{ticker} Stock Analysis',
            font=dict(size=20, color='#22c55e'),
            x=0.5
        ),
        paper_bgcolor='#0d1a0d',
        plot_bgcolor='#0a0a0a',
        font=dict(family="Arial, sans-serif", size=12, color='#c7e9c0'),
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#c7e9c0')
        )
    )

    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#1a1a1a',
        linecolor='#22c55e',
        linewidth=1
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor='#1a1a1a',
        linecolor='#22c55e',
        linewidth=1
    )

    return fig

def get_stock_info(ticker):
    """Get basic stock information."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A')
        }
    except Exception as e:
        st.error(f"Error getting info for {ticker}: {str(e)}")
        return {}