import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

def fetch_portfolio_data(tickers, start_date, end_date):
    """Fetch historical data for multiple stocks."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if data.empty:
            raise ValueError("No data found for the selected tickers")
        return data
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")
        return pd.DataFrame()

def calculate_portfolio_returns(data, weights):
    """Calculate portfolio returns."""
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

def calculate_portfolio_metrics(returns, weights, risk_free_rate=0.02):
    """Calculate key portfolio metrics."""
    try:
        if len(returns) < 2:
            return {
                'Annual Return': 0,
                'Annual Volatility': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown': 0
            }

        # Annualize returns properly using geometric mean
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_volatility = returns.std() * np.sqrt(252)

        # Risk-free rate should be daily for Sharpe ratio calculation
        daily_risk_free = risk_free_rate / 252
        excess_returns = returns - daily_risk_free

        # Handle case where volatility is zero
        if returns.std() == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
    except Exception as e:
        st.error(f"Error calculating portfolio metrics: {str(e)}")
        return {
            'Annual Return': 0,
            'Annual Volatility': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown': 0
        }

def optimize_portfolio(data, target_return=None):
    """Optimize portfolio for minimum volatility or maximum Sharpe ratio."""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(data.columns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    def negative_sharpe_ratio(weights):
        portfolio_return = (1 + np.sum(mean_returns * weights)) ** 252 - 1
        portfolio_vol = portfolio_volatility(weights)
        return -(portfolio_return - 0.02) / portfolio_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    if target_return:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: ((1 + np.sum(mean_returns * x)) ** 252 - 1) - target_return}
        )

    initial_weights = np.array([1/num_assets] * num_assets)

    if target_return:
        result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x if result.success else initial_weights

def plot_portfolio_allocation(weights, tickers):
    """Plot portfolio allocation pie chart."""
    # Dark theme color palette with green shades
    colors = ['#22c55e', '#16a34a', '#15803d', '#166534', '#4ade80', '#86efac', '#dcfce7', '#f0fdf4']

    fig = px.pie(
        values=weights,
        names=tickers,
        title='💰 Portfolio Allocation',
        color_discrete_sequence=colors[:len(tickers)]
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br>Value: %{value:.1%}<extra></extra>'
    )

    fig.update_layout(
        paper_bgcolor='#0d1a0d',
        plot_bgcolor='#0a0a0a',
        font=dict(family="Inter, sans-serif", size=12, color='#c7e9c0'),
        title_font=dict(size=18, color='#22c55e'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color='#c7e9c0')
        )
    )

    return fig

def plot_portfolio_performance(portfolio_returns, benchmark_returns=None):
    """Plot portfolio cumulative returns."""
    cumulative_returns = (1 + portfolio_returns).cumprod()

    fig = go.Figure()

    # Portfolio line with enhanced styling
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#22c55e', width=3),
        fill='tozeroy',
        fillcolor='rgba(34, 197, 94, 0.1)'
    ))

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ea580c', width=2, dash='dash')
        ))

    fig.update_layout(
        title='📈 Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        paper_bgcolor='#0d1a0d',
        plot_bgcolor='#0a0a0a',
        font=dict(family="Inter, sans-serif", size=12, color='#c7e9c0'),
        title_font=dict(size=18, color='#22c55e'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1,
            tickformat='.1%'
        ),
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

    return fig

def calculate_efficient_frontier(data, num_portfolios=1000):
    """Calculate efficient frontier."""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(data.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = portfolio_return / portfolio_std  # Sharpe ratio

    return results, weights_record