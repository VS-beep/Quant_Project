import numpy as np
import pandas as pd
from scipy.stats import norm, t
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

def calculate_historical_var(returns, confidence_level=0.95):
    """Calculate Historical Value at Risk."""
    try:
        var = -np.percentile(returns, (1 - confidence_level) * 100)
        return var
    except Exception as e:
        st.error(f"Error calculating Historical VaR: {str(e)}")
        return None

def calculate_parametric_var(returns, confidence_level=0.95):
    """Calculate Parametric Value at Risk using normal distribution."""
    try:
        mean = returns.mean()
        std = returns.std()
        z_score = norm.ppf(1 - confidence_level)
        # VaR is the negative of the loss, so we take the negative
        var = -(mean + z_score * std)
        return var
    except Exception as e:
        st.error(f"Error calculating Parametric VaR: {str(e)}")
        return None

def calculate_monte_carlo_var(returns, confidence_level=0.95, num_simulations=10000):
    """Calculate Monte Carlo Value at Risk."""
    try:
        mean = returns.mean()
        std = returns.std()
        simulated_returns = np.random.normal(mean, std, num_simulations)
        var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
        return var
    except Exception as e:
        st.error(f"Error calculating Monte Carlo VaR: {str(e)}")
        return None

def calculate_expected_shortfall(returns, confidence_level=0.95):
    """Calculate Expected Shortfall (Conditional VaR - average loss beyond VaR)."""
    try:
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        es = returns[returns <= var_threshold].mean()
        return -es  # Return positive value for consistency with VaR
    except Exception as e:
        st.error(f"Error calculating Expected Shortfall: {str(e)}")
        return None

def calculate_beta(stock_returns, market_returns):
    """Calculate beta of a stock relative to market."""
    try:
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance
        return beta
    except Exception as e:
        st.error(f"Error calculating beta: {str(e)}")
        return None

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio."""
    try:
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe
    except Exception as e:
        st.error(f"Error calculating Sharpe ratio: {str(e)}")
        return None

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino ratio."""
    try:
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        return sortino
    except Exception as e:
        st.error(f"Error calculating Sortino ratio: {str(e)}")
        return None

def stress_test_portfolio(portfolio_returns, stress_scenarios):
    """Perform stress testing on portfolio."""
    try:
        results = {}
        for scenario, shock in stress_scenarios.items():
            stressed_returns = portfolio_returns * (1 + shock)
            stressed_cumulative = (1 + stressed_returns).cumprod()
            results[scenario] = {
                'Final Value': stressed_cumulative.iloc[-1],
                'Max Drawdown': (stressed_cumulative / stressed_cumulative.expanding().max() - 1).min()
            }
        return results
    except Exception as e:
        st.error(f"Error in stress testing: {str(e)}")
        return {}

def plot_risk_metrics(returns):
    """Plot risk metrics visualization."""
    fig = go.Figure()

    # Returns distribution with enhanced styling
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns Distribution',
        marker_color='#22c55e',
        opacity=0.8,
        hovertemplate='<b>Returns</b>: %{x:.2%}<br><b>Frequency</b>: %{y}<extra></extra>'
    ))

    # Add VaR lines with better styling
    var_95 = calculate_historical_var(returns, 0.95)
    var_99 = calculate_historical_var(returns, 0.99)

    if var_95:
        fig.add_vline(
            x=-var_95,
            line_dash="dash",
            line_color="#dc2626",
            line_width=2,
            annotation_text="95% VaR",
            annotation_position="top right",
            annotation_font_color="#dc2626"
        )
    if var_99:
        fig.add_vline(
            x=-var_99,
            line_dash="dash",
            line_color="#ea580c",
            line_width=2,
            annotation_text="99% VaR",
            annotation_position="top right",
            annotation_font_color="#ea580c"
        )

    fig.update_layout(
        title='⚠️ Returns Distribution with VaR Levels',
        xaxis_title='Returns',
        yaxis_title='Frequency',
        paper_bgcolor='#0d1a0d',
        plot_bgcolor='#0a0a0a',
        font=dict(family="Inter, sans-serif", size=12, color='#c7e9c0'),
        title_font=dict(size=18, color='#22c55e'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1,
            tickformat='.1%'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1
        ),
        showlegend=False
    )

    return fig

def calculate_correlation_matrix(data):
    """Calculate correlation matrix for assets."""
    try:
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        return corr_matrix
    except Exception as e:
        st.error(f"Error calculating correlation matrix: {str(e)}")
        return pd.DataFrame()

def plot_correlation_heatmap(corr_matrix):
    """Plot correlation heatmap with enhanced styling."""
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[
            [0.0, '#dc2626'],  # Red for negative correlation
            [0.5, '#374151'],  # Grey for no correlation
            [1.0, '#22c55e']   # Green for positive correlation
        ],
        range_color=[-1, 1],
        title="📊 Asset Correlation Matrix"
    )

    fig.update_layout(
        paper_bgcolor='#0d1a0d',
        plot_bgcolor='#0a0a0a',
        font=dict(family="Inter, sans-serif", size=12, color='#c7e9c0'),
        title_font=dict(size=18, color='#22c55e'),
        coloraxis_colorbar=dict(
            title="Correlation",
            titleside="right",
            tickformat=".2f",
            title_font_color='#22c55e',
            tickfont_color='#c7e9c0'
        )
    )

    return fig