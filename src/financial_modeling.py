import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

def dcf_valuation(free_cash_flows, discount_rate, terminal_growth_rate, shares_outstanding):
    """Calculate DCF valuation."""
    try:
        # Input validation
        if not free_cash_flows or shares_outstanding <= 0:
            raise ValueError("Invalid inputs for DCF calculation")

        if discount_rate <= 0 or terminal_growth_rate >= discount_rate or terminal_growth_rate < 0:
            raise ValueError("Invalid discount or growth rates")
        
        # Additional check to prevent division by zero or near-zero denominator
        if abs(discount_rate - terminal_growth_rate) < 0.0001:
            raise ValueError("Discount rate and terminal growth rate are too close (separation must be > 0.01%)")

        # Calculate present value of free cash flows
        pv_fcfs = [fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(free_cash_flows)]

        # Calculate terminal value using Gordon Growth Model
        terminal_value = free_cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        pv_terminal = terminal_value / (1 + discount_rate) ** len(free_cash_flows)

        # Total enterprise value
        enterprise_value = sum(pv_fcfs) + pv_terminal

        # Per share value (assuming no debt for simplicity)
        per_share_value = enterprise_value / shares_outstanding

        return {
            'Enterprise Value': enterprise_value,
            'Per Share Value': per_share_value,
            'PV of FCFs': sum(pv_fcfs),
            'PV of Terminal Value': pv_terminal
        }
    except Exception as e:
        st.error(f"Error in DCF calculation: {str(e)}")
        return {}

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price."""
    try:
        # Input validation
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("All inputs must be positive")

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)  # Ensure non-negative price
    except Exception as e:
        st.error(f"Error in Black-Scholes call calculation: {str(e)}")
        return None

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price."""
    try:
        # Input validation
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("All inputs must be positive")

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put_price, 0)  # Ensure non-negative price
    except Exception as e:
        st.error(f"Error in Black-Scholes put calculation: {str(e)}")
        return None

def monte_carlo_simulation(S, T, r, sigma, num_simulations=10000, num_steps=252):
    """Run Monte Carlo simulation for stock price."""
    try:
        dt = T / num_steps
        simulations = np.zeros((num_simulations, num_steps + 1))
        simulations[:, 0] = S

        for i in range(1, num_steps + 1):
            z = np.random.standard_normal(num_simulations)
            simulations[:, i] = simulations[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

        return simulations
    except Exception as e:
        st.error(f"Error in Monte Carlo simulation: {str(e)}")
        return None

def plot_monte_carlo(simulations, S):
    """Plot Monte Carlo simulation results with enhanced styling."""
    fig = go.Figure()

    # Plot a few sample paths with subtle styling
    for i in range(min(50, len(simulations))):  # Reduced to 50 for cleaner look
        fig.add_trace(go.Scatter(
            y=simulations[i],
            mode='lines',
            line=dict(width=0.5, color='#81C784'),
            showlegend=False,
            opacity=0.3
        ))

    # Plot final distribution with money theme
    final_prices = simulations[:, -1]
    fig.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=50,
        name='Final Price Distribution',
        marker_color='#4CAF50',
        opacity=0.8,
        hovertemplate='<b>Price Range</b>: %{x:.2f}<br><b>Frequency</b>: %{y}<extra></extra>'
    ))

    # Add starting price line
    fig.add_hline(
        y=S,
        line_dash="dash",
        line_color="#2E7D32",
        line_width=2,
        annotation_text=f"Starting Price: ${S}",
        annotation_position="bottom right",
        annotation_font_color="#2E7D32"
    )

    fig.update_layout(
        title=f'Monte Carlo Simulation (Starting Price: ${S})',
        xaxis_title='Time Steps',
        yaxis_title='Price ($)',
        paper_bgcolor='#F1F8E9',
        plot_bgcolor='#F9FBE7',
        font=dict(family="Inter, sans-serif", size=12, color='#1B5E20'),
        title_font=dict(size=18, color='#2E7D32'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#C8E6C9',
            linecolor='#4CAF50',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#C8E6C9',
            linecolor='#4CAF50',
            linewidth=1,
            tickformat='$,.0f'
        ),
        showlegend=True,
        legend=dict(
            title="Legend",
            font_color='#1B5E20'
        )
    )

    return fig

def calculate_var(returns, confidence_level=0.95):
    """Calculate Value at Risk (VaR) - returns positive loss magnitude."""
    try:
        var = -np.percentile(returns, (1 - confidence_level) * 100)
        return var
    except Exception as e:
        st.error(f"Error calculating VaR: {str(e)}")
        return None

def calculate_cvar(returns, confidence_level=0.95):
    """Calculate Conditional Value at Risk (CVaR) - returns positive loss magnitude."""
    try:
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = -returns[returns <= var_threshold].mean()
        return cvar
    except Exception as e:
        st.error(f"Error calculating CVaR: {str(e)}")
        return None