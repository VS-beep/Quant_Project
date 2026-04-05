import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_analysis import *
from portfolio_management import *
from financial_modeling import *
from risk_assessment import *
from lstm_predictor import *
from utils import *

# Page configuration
st.set_page_config(
    page_title="Quantitative Finance Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None
if 'last_start_date' not in st.session_state:
    st.session_state.last_start_date = None
if 'last_end_date' not in st.session_state:
    st.session_state.last_end_date = None

@st.cache_data
def fetch_stock_data_cached(ticker, start_date, end_date):
    """Cached version of stock data fetching."""
    return fetch_stock_data(ticker, start_date, end_date)

@st.cache_data
def fetch_portfolio_data_cached(tickers, start_date, end_date):
    """Cached version of portfolio data fetching."""
    return fetch_portfolio_data(tickers, start_date, end_date)

@st.cache_data
def calculate_technical_indicators(data, short_window, long_window, rsi_window):
    """Cached calculation of technical indicators."""
    data_with_indicators = data.copy()
    data_with_indicators = calculate_moving_averages(
        data_with_indicators, short_window, long_window
    )
    data_with_indicators = calculate_rsi(data_with_indicators, rsi_window)
    data_with_indicators = calculate_macd(data_with_indicators)
    return data_with_indicators

# Custom CSS with anime.js integration
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Anime.js animation classes */
    .animate-fade-in {
        opacity: 0;
        animation: fadeIn 0.8s ease-in-out forwards;
    }
    
    .animate-slide-in {
        transform: translateX(-20px);
        opacity: 0;
        animation: slideIn 0.6s ease-out forwards;
    }
    
    .animate-scale-up {
        transform: scale(0.95);
        opacity: 0;
        animation: scaleUp 0.5s ease-out forwards;
    }
    
    .animate-bounce {
        animation: bounce 0.8s ease-in-out infinite;
    }
    
    .animate-pulse-glow {
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    @keyframes fadeIn {
        to {
            opacity: 1;
        }
    }
    
    @keyframes slideIn {
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes scaleUp {
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes pulseGlow {
        0%, 100% { 
            box-shadow: 0 0 5px rgba(34, 197, 94, 0.5);
        }
        50% { 
            box-shadow: 0 0 20px rgba(34, 197, 94, 0.8);
        }
    }
    
    /* Animated gradient background */
    .gradient-animate {
        background: linear-gradient(
            -45deg,
            #0a0a0a,
            #0d1a0d,
            #0a0a0a,
            #0d1a0d
        );
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
<script>
    // Apply anime.js animations to elements when page loads
    document.addEventListener('DOMContentLoaded', function() {
        // Fade in headers
        anime({
            targets: '.main-header',
            opacity: [0, 1],
            duration: 1500,
            easing: 'easeOutQuad'
        });
        
        // Slide in sidebar
        anime({
            targets: '.sidebar-header',
            translateX: [-30, 0],
            opacity: [0, 1],
            duration: 800,
            easing: 'easeOutElastic(1, .6)',
            delay: 200
        });
    });
</script>
""", unsafe_allow_html=True)

# Load custom CSS if exists
css_file = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
if os.path.exists(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">Quantitative Finance Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)

    page = st.sidebar.selectbox(
        "Choose a module:",
        ["Stock Analysis", "Portfolio Management", "Financial Modeling", "Risk Assessment", "LSTM Prediction"]
    )

    # Date range selector
    st.sidebar.markdown("### Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    if not validate_date_range(start_date, end_date):
        st.stop()

    # Main content
    if page == "Stock Analysis":
        stock_analysis_page(start_date, end_date)
    elif page == "Portfolio Management":
        portfolio_management_page(start_date, end_date)
    elif page == "Financial Modeling":
        financial_modeling_page()
    elif page == "Risk Assessment":
        risk_assessment_page(start_date, end_date)
    elif page == "LSTM Prediction":
        lstm_prediction_page()

def stock_analysis_page(start_date, end_date):
    st.header("Stock Analysis")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", "AAPL").upper()

    # Check if we need to fetch new data
    data_changed = (
        st.session_state.last_ticker != ticker or
        st.session_state.last_start_date != start_date or
        st.session_state.last_end_date != end_date
    )

    if st.button("Analyze Stock") or data_changed:
        with st.spinner("Fetching data..."):
            data = fetch_stock_data_cached(ticker, start_date, end_date)
            if not data.empty:
                st.session_state.stock_data = data
                st.session_state.last_ticker = ticker
                st.session_state.last_start_date = start_date
                st.session_state.last_end_date = end_date

    # Use cached data if available
    data = st.session_state.stock_data
    if data is None or data.empty:
        st.warning("Please click 'Analyze Stock' to load data.")
        return

    # Stock info
    info = get_stock_info(ticker)
    if info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Company", info.get('name', ticker))
        with col2:
            st.metric("Sector", info.get('sector', 'N/A'))
        with col3:
            st.metric("Industry", info.get('industry', 'N/A'))

    # Technical indicators with session state for sliders
    st.subheader("Technical Indicators")

    # Initialize slider values in session state
    if 'short_window' not in st.session_state:
        st.session_state.short_window = 50
    if 'long_window' not in st.session_state:
        st.session_state.long_window = 200
    if 'rsi_window' not in st.session_state:
        st.session_state.rsi_window = 14

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.short_window = st.slider(
            "Short MA Window", 10, 100, st.session_state.short_window, key="short_ma",
            help="Number of days to calculate short-term moving average. Lower values = more responsive to recent price changes."
        )
        st.session_state.long_window = st.slider(
            "Long MA Window", 100, 300, st.session_state.long_window, key="long_ma",
            help="Number of days to calculate long-term moving average. Higher values = smoother trend line."
        )
    with col2:
        st.session_state.rsi_window = st.slider(
            "RSI Window", 5, 30, st.session_state.rsi_window, key="rsi",
            help="Relative Strength Index lookback period. Measures momentum on a scale of 0-100. Above 70 = overbought, below 30 = oversold."
        )

    # Calculate indicators with current slider values (cached)
    data_with_indicators = calculate_technical_indicators(
        data, st.session_state.short_window, st.session_state.long_window, st.session_state.rsi_window
    )

    # Chart
    st.plotly_chart(plot_stock_chart(data_with_indicators, ticker), width='stretch')

    # Key metrics
    st.subheader("Key Metrics")
    returns = calculate_returns(data['Close'])
    annual_return = annualize_returns(returns)
    annual_vol = annualize_volatility(returns)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", format_currency(data['Close'].iloc[-1]))
    with col2:
        st.metric("Annual Return", format_percentage(annual_return * 100))
    with col3:
        st.metric("Annual Volatility", format_percentage(annual_vol * 100))
    with col4:
        st.metric("52W High", format_currency(data['High'].max()))

def portfolio_management_page(start_date, end_date):
    st.header("Portfolio Management")

    st.subheader("Portfolio Composition")
    tickers_input = st.text_area("Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL)", "AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if not validate_tickers(tickers):
        st.stop()

    # Initialize portfolio weights in session state
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = {}
    if 'current_tickers' not in st.session_state:
        st.session_state.current_tickers = []

    # Check if tickers changed
    tickers_changed = st.session_state.current_tickers != tickers

    if st.button("Analyze Portfolio") or tickers_changed:
        with st.spinner("Fetching portfolio data..."):
            data = fetch_portfolio_data_cached(tickers, start_date, end_date)
            if not data.empty:
                st.session_state.portfolio_data = data
                st.session_state.current_tickers = tickers
                # Initialize weights for new tickers
                for ticker in tickers:
                    if ticker not in st.session_state.portfolio_weights:
                        st.session_state.portfolio_weights[ticker] = 100 // len(tickers)

    # Use cached data if available
    data = st.session_state.portfolio_data if 'portfolio_data' in st.session_state else None
    if data is None or data.empty:
        st.warning("Please click 'Analyze Portfolio' to load data.")
        return

    # Portfolio weights sliders
    st.subheader("Portfolio Weights")
    weights = []
    cols = st.columns(len(tickers))

    for i, ticker in enumerate(tickers):
        with cols[i]:
            # Update session state when slider changes
            current_weight = st.session_state.portfolio_weights.get(ticker, 100//len(tickers))
            new_weight = st.slider(
                f"{ticker} Weight (%)",
                0, 100,
                current_weight,
                key=f"weight_{ticker}",
                help="Percentage of portfolio allocated to this asset. Will be normalized to 100% total."
            )
            st.session_state.portfolio_weights[ticker] = new_weight
            weights.append(new_weight / 100)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        st.error("Total weight cannot be zero")
        st.stop()
    weights = np.array(weights) / total_weight

    # Calculate portfolio returns and metrics
    portfolio_returns = calculate_portfolio_returns(data, weights)
    metrics = calculate_portfolio_metrics(portfolio_returns)

    # Display metrics
    st.subheader("Portfolio Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Return", format_percentage(metrics['Annual Return'] * 100))
    with col2:
        st.metric("Annual Volatility", format_percentage(metrics['Annual Volatility'] * 100))
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    with col4:
        st.metric("Max Drawdown", format_percentage(metrics['Max Drawdown'] * 100))

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_portfolio_allocation(weights, tickers), width='stretch')
    with col2:
        st.plotly_chart(plot_portfolio_performance(portfolio_returns), width='stretch')

    # Optimization section
    st.subheader("Portfolio Optimization")
    if st.button("Optimize for Maximum Sharpe Ratio"):
        with st.spinner("Optimizing..."):
            optimal_weights = optimize_portfolio(data)
            st.write("Optimal Weights:")
            for ticker, weight in zip(tickers, optimal_weights):
                st.write(f"{ticker}: {weight:.2%}")
                # Update session state with optimal weights
                st.session_state.portfolio_weights[ticker] = int(weight * 100)

def financial_modeling_page():
    st.header("Financial Modeling")
    
    # Model selection with descriptions
    st.markdown("""
    <div style="background: linear-gradient(135deg, #303030 0%, #2a2a2a 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 20px;">
        <h3 style="color: #66bb6a; margin-top: 0;">Choose a Valuation Model</h3>
        <p style="color: #a0a0a0;">Explore different financial modeling approaches to analyze investments and understand potential outcomes.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different models
    tab1, tab2, tab3 = st.tabs(["DCF Valuation", "Black-Scholes Options", "Monte Carlo Simulation"])

    #  DCF VALUATION 
    with tab1:
        st.subheader("Discounted Cash Flow (DCF) Valuation")
        
        # Model definition
        with st.expander("What is DCF Valuation?", expanded=True):
            st.markdown("""
            **DCF Valuation** estimates a company's intrinsic value by projecting future cash flows and discounting them back to present value.
            
            **Formula:** Enterprise Value = Σ(FCF / (1+r)^n) + (TV / (1+r)^n)
            
            **Components:**
            - **Free Cash Flows (FCF):** Cash available after expenses and capital expenses
            - **Discount Rate (r):** Your required rate of return (WACC)  
            - **Terminal Value:** Perpetual value after forecast period
            
            **When to use:** Fundamental analysis, long-term valuation, mature companies with predictable cash flows
            
            **Best for:** Investors doing deep fundamental analysis
            """)

        # Initialize DCF parameters in session state
        if 'dcf_fcfs' not in st.session_state:
            st.session_state.dcf_fcfs = "100, 110, 121, 133.1"
        if 'dcf_discount' not in st.session_state:
            st.session_state.dcf_discount = 10
        if 'dcf_growth' not in st.session_state:
            st.session_state.dcf_growth = 2
        if 'dcf_shares' not in st.session_state:
            st.session_state.dcf_shares = 1000

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.dcf_fcfs = st.text_area(
                "Free Cash Flows (comma-separated)",
                st.session_state.dcf_fcfs,
                help="Projected yearly cash flows. Example: 100, 110, 121, 133.1",
                height=100
            )
        with col2:
            st.markdown("### Parameters")
            st.session_state.dcf_discount = st.slider(
                "Discount Rate (%)", 5, 20, st.session_state.dcf_discount,
                help="Your required rate of return (8-12% typical)"
            )
            st.session_state.dcf_growth = st.slider(
                "Terminal Growth Rate (%)", 0, 5, st.session_state.dcf_growth,
                help="Long-term growth (2-3% typical)"
            )
            st.session_state.dcf_shares = st.number_input(
                "Shares Outstanding (millions)",
                100, 10000, st.session_state.dcf_shares,
                help="Total shares in millions"
            )

        # Auto-calculate DCF
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Calculate DCF Value", key="dcf_calc", use_container_width=True):
                try:
                    fcfs = [float(x.strip()) for x in st.session_state.dcf_fcfs.split(',')]
                    result = dcf_valuation(fcfs, st.session_state.dcf_discount/100,
                                         st.session_state.dcf_growth/100,
                                         st.session_state.dcf_shares * 1e6)
                    if result:
                        st.session_state.dcf_result = result
                except ValueError:
                    st.error("Please enter valid numbers for free cash flows")

        if 'dcf_result' in st.session_state:
            result = st.session_state.dcf_result
            st.markdown("---")
            st.markdown("### Valuation Results")
            
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("Per Share Value", format_currency(result['Per Share Value']), delta="Intrinsic Value")
            with res_col2:
                st.metric("Enterprise Value", format_currency(result['Enterprise Value']))
            with res_col3:
                st.metric("PV of FCFs", format_currency(result['PV of FCFs']))
            with res_col4:
                st.metric("PV of Terminal", format_currency(result['PV of Terminal Value']))
            
            with st.expander("Detailed Breakdown"):
                for key, value in result.items():
                    st.write(f"**{key}:** {format_currency(value)}")

    #  BLACK-SCHOLES OPTIONS 
    with tab2:
        st.subheader("Black-Scholes Option Pricing Model")
        
        # Model definition
        with st.expander("What is Black-Scholes?", expanded=True):
            st.markdown("""
            **Black-Scholes** calculates the fair value of European-style options using probabilistic analysis.
            
            **Formula:** C = S₀N(d₁) - Ke^(-rT)N(d₂)
            
            **Where:**
            - **d₁ & d₂** are calculated from stock price, strike, volatility, time, and interest rate
            - **N(x)** is the cumulative standard normal distribution
            
            **Inputs:**
            - **Stock Price (S):** Current market price
            - **Strike Price (K):** Exercise price  
            - **Time to Expiration (T):** Years until expiry
            - **Volatility (σ):** Annualized price volatility
            - **Risk-free Rate (r):** Treasury yield
            
            **When to use:** Options traders, derivatives pricing, risk management
            
            **Best for:** Short-term trading, hedging, options pricing
            """)

        # Initialize BS parameters
        if 'bs_S' not in st.session_state:
            st.session_state.bs_S = 100.0
        if 'bs_K' not in st.session_state:
            st.session_state.bs_K = 100.0
        if 'bs_T' not in st.session_state:
            st.session_state.bs_T = 1.0
        if 'bs_r' not in st.session_state:
            st.session_state.bs_r = 4
        if 'bs_sigma' not in st.session_state:
            st.session_state.bs_sigma = 25
        if 'bs_type' not in st.session_state:
            st.session_state.bs_type = "Call"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Stock Parameters")
            st.session_state.bs_S = st.number_input(
                "Stock Price ($)", 0.1, 10000.0, st.session_state.bs_S,
                help="Current market price"
            )
            st.session_state.bs_K = st.number_input(
                "Strike Price ($)", 0.1, 10000.0, st.session_state.bs_K,
                help="Exercise price"
            )
        
        with col2:
            st.markdown("### Time & Rate")
            st.session_state.bs_T = st.number_input(
                "Time to Exp. (years)", 0.01, 10.0, st.session_state.bs_T,
                step=0.1,
                help="Time remaining in years"
            )
            st.session_state.bs_r = st.slider(
                "Risk-free Rate (%)", 0, 10, st.session_state.bs_r,
                help="Treasury yield"
            )
        
        with col3:
            st.markdown("### Volatility & Type")
            st.session_state.bs_sigma = st.slider(
                "Volatility (%)", 5, 100, st.session_state.bs_sigma,
                help="Annualized volatility"
            )
            st.session_state.bs_type = st.selectbox(
                "Option Type", ["Call", "Put"],
                help="Call or Put option"
            )

        # Auto-calculate
        if st.session_state.bs_type == "Call":
            price = black_scholes_call(
                st.session_state.bs_S, st.session_state.bs_K, st.session_state.bs_T,
                st.session_state.bs_r/100, st.session_state.bs_sigma/100
            )
        else:
            price = black_scholes_put(
                st.session_state.bs_S, st.session_state.bs_K, st.session_state.bs_T,
                st.session_state.bs_r/100, st.session_state.bs_sigma/100
            )

        if price is not None:
            st.markdown("---")
            opt_col1, opt_col2, opt_col3 = st.columns(3)
            with opt_col1:
                st.metric(f"{st.session_state.bs_type} Option Price", format_currency(price))
            with opt_col2:
                intrinsic = max(st.session_state.bs_S - st.session_state.bs_K, 0) if st.session_state.bs_type == "Call" else max(st.session_state.bs_K - st.session_state.bs_S, 0)
                st.metric("Intrinsic Value", format_currency(intrinsic))
            with opt_col3:
                time_value = price - intrinsic
                st.metric("Time Value", format_currency(time_value))

    #  MONTE CARLO SIMULATION 
    with tab3:
        st.subheader("Monte Carlo Stock Price Simulation")
        
        # Model definition
        with st.expander("What is Monte Carlo?", expanded=True):
            st.markdown("""
            **Monte Carlo** runs thousands of random simulations to forecast possible future stock prices.
            
            **Process:**
            1. Generate random price paths based on expected drift and volatility
            2. Run simulations (1K to 10K paths)
            3. Analyze distribution of final prices
            
            **Uses:**
            - Portfolio risk analysis
            - Scenario planning
            - Understanding probability distributions
            - VaR and stress testing
            
            **When to use:** Risk assessment, probability analysis, complex portfolios
            
            **Best for:** Risk managers, hedge funds, portfolio analysis
            """)

        # Initialize MC parameters
        if 'mc_S' not in st.session_state:
            st.session_state.mc_S = 100.0
        if 'mc_T' not in st.session_state:
            st.session_state.mc_T = 1.0
        if 'mc_r' not in st.session_state:
            st.session_state.mc_r = 4
        if 'mc_sigma' not in st.session_state:
            st.session_state.mc_sigma = 25
        if 'mc_sims' not in st.session_state:
            st.session_state.mc_sims = 10000

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Initial Conditions")
            st.session_state.mc_S = st.number_input(
                "Initial Price ($)", 0.1, 10000.0, st.session_state.mc_S, key="mc_S_input",
                help="Starting stock price"
            )
            st.session_state.mc_T = st.number_input(
                "Time Horizon (years)", 0.1, 10.0, st.session_state.mc_T, key="mc_T_input",
                step=0.1,
                help="Years to simulate"
            )
        
        with col2:
            st.markdown("### Market Assumptions")
            st.session_state.mc_r = st.slider(
                "Expected Return (%)", 0, 15, st.session_state.mc_r, key="mc_r_slider",
                help="Annual drift rate"
            )
            st.session_state.mc_sigma = st.slider(
                "Volatility (%)", 5, 100, st.session_state.mc_sigma, key="mc_sigma_slider",
                help="Annual volatility"
            )
        
        with col3:
            st.markdown("### Simulation Settings")
            st.session_state.mc_sims = st.selectbox(
                "Number of Paths", [1000, 5000, 10000, 25000],
                index=[1000, 5000, 10000, 25000].index(st.session_state.mc_sims) if st.session_state.mc_sims in [1000, 5000, 10000, 25000] else 2,
                help="More paths = more accuracy"
            )

        # Run simulation
        if st.button("Run Simulation", use_container_width=True, key="mc_run"):
            with st.spinner("Running simulation..."):
                simulations = monte_carlo_simulation(
                    st.session_state.mc_S, st.session_state.mc_T,
                    st.session_state.mc_r/100, st.session_state.mc_sigma/100,
                    st.session_state.mc_sims
                )
                if simulations is not None:
                    st.session_state.mc_simulations = simulations

        if 'mc_simulations' in st.session_state:
            simulations = st.session_state.mc_simulations
            st.plotly_chart(plot_monte_carlo(simulations, st.session_state.mc_S), use_container_width=True)

            final_prices = simulations[:, -1]
            st.markdown("---")
            st.markdown("### Simulation Analysis")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Expected Final Price", format_currency(np.mean(final_prices)), 
                         delta=f"+{format_percentage((np.mean(final_prices) / st.session_state.mc_S - 1) * 100)}")
            with res_col2:
                ci_low = np.percentile(final_prices, 2.5)
                ci_high = np.percentile(final_prices, 97.5)
                st.metric("95% Confidence Interval", f"{format_currency(ci_low)} to {format_currency(ci_high)}")
            with res_col3:
                prob_profit = np.mean(final_prices > st.session_state.mc_S)
                st.metric("Probability of Profit", f"{prob_profit:.1%}")
            
            # Additional statistics
            with st.expander("Detailed Statistics"):
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Median", format_currency(np.median(final_prices)))
                with stats_col2:
                    st.metric("Std Dev", format_currency(np.std(final_prices)))
                with stats_col3:
                    st.metric("Min", format_currency(np.min(final_prices)))
                with stats_col4:
                    st.metric("Max", format_currency(np.max(final_prices)))

def risk_assessment_page(start_date, end_date):
    st.header("Risk Assessment")
    
    # Introduction
    st.markdown("""
    <div style="background: linear-gradient(135deg, #303030 0%, #2a2a2a 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 20px;">
        <h3 style="color: #66bb6a; margin-top: 0;">Portfolio Risk Analysis</h3>
        <p style="color: #a0a0a0;">Understand downside risk, losses, and diversification of your portfolio.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Portfolio Selection")
    tickers_input = st.text_area(
        "Enter portfolio tickers (comma-separated)", 
        "AAPL, MSFT, GOOGL", 
        key="risk_tickers",
        help="Stock symbols to analyze. Example: AAPL, MSFT, GOOGL, TSLA"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if not validate_tickers(tickers):
        st.stop()

    if st.button("Analyze Portfolio Risk", use_container_width=True):
        with st.spinner("Analyzing risk metrics..."):
            data = fetch_portfolio_data(tickers, start_date, end_date)

        if not data.empty:
            returns = calculate_returns(data)
            portfolio_returns = returns.mean(axis=1)  # Equal-weighted portfolio

            #  VALUE AT RISK SECTION 
            st.markdown("---")
            st.markdown("## Value at Risk (VaR)")
            
            with st.expander("What is VaR?", expanded=True):
                st.markdown("""
                **Value at Risk (VaR)** estimates the maximum loss over a specific time period at a given confidence level.
                
                - **95% VaR:** Maximum loss with 95% confidence (5% chance of worse)
                - **99% VaR:** Maximum loss with 99% confidence (1% chance of worse)
                
                **Three Methods:**
                1. **Historical:** Uses actual past returns distribution
                2. **Parametric:** Assumes normal distribution (faster, more biased)
                3. **Monte Carlo:** Simulates thousands of scenarios (most flexible)
                
                **Interpretation:** If 95% VaR is -5%, there's 95% chance daily loss won't exceed 5%.
                """)

            # Calculate VaR using three methods
            var_95_hist = calculate_historical_var(portfolio_returns, 0.95)
            var_99_hist = calculate_historical_var(portfolio_returns, 0.99)
            var_95_param = calculate_parametric_var(portfolio_returns, 0.95)
            var_99_param = calculate_parametric_var(portfolio_returns, 0.99)
            var_95_mc = calculate_monte_carlo_var(portfolio_returns, 0.95)
            var_99_mc = calculate_monte_carlo_var(portfolio_returns, 0.99)

            # Display VaR results
            st.markdown("### Confidence Level: 95%")
            var_col1, var_col2, var_col3 = st.columns(3)
            with var_col1:
                st.metric(
                    "Historical VaR",
                    format_percentage(var_95_hist * 100) if var_95_hist else "N/A",
                    help="Based on actual historical returns"
                )
            with var_col2:
                st.metric(
                    "Parametric VaR",
                    format_percentage(var_95_param * 100) if var_95_param else "N/A",
                    help="Assumes normal distribution"
                )
            with var_col3:
                st.metric(
                    "Monte Carlo VaR",
                    format_percentage(var_95_mc * 100) if var_95_mc else "N/A",
                    help="Based on 10K simulations"
                )

            st.markdown("### Confidence Level: 99%")
            var99_col1, var99_col2, var99_col3 = st.columns(3)
            with var99_col1:
                st.metric(
                    "Historical VaR",
                    format_percentage(var_99_hist * 100) if var_99_hist else "N/A"
                )
            with var99_col2:
                st.metric(
                    "Parametric VaR",
                    format_percentage(var_99_param * 100) if var_99_param else "N/A"
                )
            with var99_col3:
                st.metric(
                    "Monte Carlo VaR",
                    format_percentage(var_99_mc * 100) if var_99_mc else "N/A"
                )

            #  EXPECTED SHORTFALL 
            st.markdown("---")
            st.markdown("## Expected Shortfall (CVaR)")
            
            with st.expander("What is Expected Shortfall?", expanded=False):
                st.markdown("""
                **Expected Shortfall (CVaR)** is the average loss when VaR is exceeded.
                
                - More conservative than VaR
                - Captures tail risk (extreme scenarios)
                - Example: If 95% ES is -8%, average loss in worst 5% is 8%
                
                **VaR vs ES:**
                - VaR: "We won't lose more than X with 95% confidence"
                - ES: "On average, when we exceed VaR, we lose X"
                """)

            es_95 = calculate_expected_shortfall(portfolio_returns, 0.95)
            es_99 = calculate_expected_shortfall(portfolio_returns, 0.99)
            
            es_col1, es_col2 = st.columns(2)
            with es_col1:
                st.metric(
                    "Expected Shortfall (95%)",
                    format_percentage(es_95 * 100) if es_95 else "N/A",
                    help="Average loss in worst 5% of scenarios"
                )
            with es_col2:
                st.metric(
                    "Expected Shortfall (99%)",
                    format_percentage(es_99 * 100) if es_99 else "N/A",
                    help="Average loss in worst 1% of scenarios"
                )

            #  RISK DISTRIBUTION 
            st.markdown("---")
            st.markdown("## Return Distribution & Risk")
            st.plotly_chart(plot_risk_metrics(portfolio_returns), use_container_width=True)

            #  CORRELATION ANALYSIS 
            if len(tickers) > 1:
                st.markdown("---")
                st.markdown("## Asset Correlations")
                
                with st.expander("What are Correlations?", expanded=False):
                    st.markdown("""
                    **Correlation** measures how assets move together:
                    
                    - **+1.0:** Perfect positive (move together)
                    - **0.0:** No relationship
                    - **-1.0:** Perfect negative (move opposite)
                    
                    **Portfolio Benefit:**
                    - **Low/negative correlations:** Better diversification
                    - **High positive correlations:** Less diversification benefit
                    
                    **Color Coding:**
                    - Green: Positive correlation (assets move together)
                    - White: No correlation
                    - Red: Negative correlation (assets move opposite - better for diversification)
                    """)
                
                corr_matrix = calculate_correlation_matrix(data)
                if not corr_matrix.empty:
                    st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)
                    
                    # Diversification summary
                    st.markdown("### Diversification Summary")
                    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                    
                    if avg_corr > 0.7:
                        diversification_rating = "Poor - Assets highly correlated"
                    elif avg_corr > 0.4:
                        diversification_rating = "Fair - Some diversification benefit"
                    else:
                        diversification_rating = "Good - Effective diversification"
                    
                    st.info(f"**Average Correlation:** {avg_corr:.2f}  \n**Diversification:** {diversification_rating}")

def lstm_prediction_page():
    """Enhanced LSTM Stock Price Prediction Page with improved robustness."""
    st.header("Advanced LSTM Stock Price Prediction")
    
    # Check TensorFlow availability
    if not check_tensorflow():
        st.error("""
        TensorFlow is not installed. To use LSTM predictions, install it with:
        ```
        pip install tensorflow scikit-learn
        ```
        """)
        return
    
    # Introduction
    st.markdown("""
    <div style="background: linear-gradient(135deg, #303030 0%, #2a2a2a 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #22c55e; margin-bottom: 20px;">
        <h3 style="color: #66bb6a; margin-top: 0;">Robust Deep Learning Forecasting</h3>
        <p style="color: #a0a0a0;">Advanced LSTM with bidirectional layers, batch normalization, batch normalization, and Monte Carlo uncertainty quantification.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Stock Ticker", "AAPL", help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)").upper()
    
    with col2:
        historical_days = st.slider("Historical Data (days)", 100, 500, 365, step=50, help="More data = better training")
    
    with col3:
        future_days = st.slider("Forecast Horizon (days)", 5, 60, 30, step=5, help="Days to forecast ahead")
    
    st.markdown("---")
    
    # Model architecture settings
    st.subheader("Advanced Model Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        lookback = st.slider("Lookback Window (days)", 20, 120, 60, step=10, 
                            help="Sequence length for LSTM training")
    
    with config_col2:
        lstm_units = st.slider("LSTM Units", 32, 128, 64, step=8, 
                              help="Number of LSTM neurons (higher = more capacity)")
    
    with config_col3:
        dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.25, step=0.05,
                                help="Regularization to prevent overfitting")
    
    st.markdown("---")
    
    # Train model button
    if st.button("Train Advanced LSTM Model", use_container_width=True):
        try:
            # Techno loading sequence
            loading_placeholder = st.empty()
            
            with loading_placeholder.container():
                st.markdown("""
                <style>
                    @keyframes rotate { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
                    @keyframes dataFlow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
                </style>
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(13, 26, 13, 0.8), rgba(10, 10, 10, 0.9)); border-radius: 12px; border: 2px solid #22c55e; box-shadow: 0 0 30px rgba(34, 197, 94, 0.3); margin: 20px 0;">
                    <div style="position: relative; width: 70px; height: 70px;">
                        <div style="position: absolute; width: 100%; height: 100%; border: 3px solid transparent; border-top: 3px solid #22c55e; border-right: 3px solid #4ade80; border-radius: 50%; animation: rotate 2s linear infinite;"></div>
                        <div style="position: absolute; width: 45px; height: 45px; top: 12.5px; left: 12.5px; border: 2px solid transparent; border-left: 2px solid #16a34a; border-bottom: 2px solid #16a34a; border-radius: 50%; animation: rotate 3s linear reverse infinite;"></div>
                    </div>
                    <div style="color: #22c55e; font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; margin-top: 20px; letter-spacing: 3px; animation: pulse 1.5s ease-in-out infinite; text-shadow: 0 0 10px rgba(34, 197, 94, 0.5);">INITIALIZING NEURAL NETWORK</div>
                    <div style="width: 100%; max-width: 300px; height: 3px; background: rgba(34, 197, 94, 0.1); border-radius: 2px; margin-top: 20px; overflow: hidden;">
                        <div style="height: 100%; background: linear-gradient(90deg, #22c55e, #4ade80, #22c55e); background-size: 200% 100%; border-radius: 2px; animation: dataFlow 2s ease infinite;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Fetch and validate data
            st.info(f"Fetching {historical_days} days of historical data for {ticker}...")
            raw_data = fetch_stock_data_for_lstm(ticker, historical_days)
            
            if raw_data is None:
                st.error(f"Failed to fetch data for {ticker}")
                loading_placeholder.empty()
                return
            
            # Update loading status
            with loading_placeholder.container():
                st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(13, 26, 13, 0.8), rgba(10, 10, 10, 0.9)); border-radius: 12px; border: 2px solid #22c55e; box-shadow: 0 0 30px rgba(34, 197, 94, 0.3); margin: 20px 0;">
                    <div style="position: relative; width: 70px; height: 70px;">
                        <div style="position: absolute; width: 100%; height: 100%; border: 3px solid transparent; border-top: 3px solid #22c55e; border-right: 3px solid #4ade80; border-radius: 50%; animation: rotate 2s linear infinite;"></div>
                        <div style="position: absolute; width: 45px; height: 45px; top: 12.5px; left: 12.5px; border: 2px solid transparent; border-left: 2px solid #16a34a; border-bottom: 2px solid #16a34a; border-radius: 50%; animation: rotate 3s linear reverse infinite;"></div>
                    </div>
                    <div style="color: #4ade80; font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; margin-top: 20px; letter-spacing: 3px; animation: pulse 1.5s ease-in-out infinite; text-shadow: 0 0 10px rgba(74, 222, 128, 0.5);">PREPROCESSING DATA</div>
                    <div style="width: 100%; max-width: 300px; height: 3px; background: rgba(34, 197, 94, 0.1); border-radius: 2px; margin-top: 20px; overflow: hidden;">
                        <div style="height: 100%; background: linear-gradient(90deg, #4ade80, #22c55e, #4ade80); background-size: 200% 100%; border-radius: 2px; animation: dataFlow 2s ease infinite;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Prepare data with 3-way split
            st.info("Preparing data (Train/Validate/Test split with robust scaling)...")
            (X_train, X_validate, X_test, y_train, y_validate, y_test, 
             scaler, _) = prepare_lstm_data(raw_data, lookback=lookback)
            
            # Update loading status
            with loading_placeholder.container():
                st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(13, 26, 13, 0.8), rgba(10, 10, 10, 0.9)); border-radius: 12px; border: 2px solid #22c55e; box-shadow: 0 0 30px rgba(34, 197, 94, 0.3); margin: 20px 0;">
                    <div style="position: relative; width: 70px; height: 70px;">
                        <div style="position: absolute; width: 100%; height: 100%; border: 3px solid transparent; border-top: 3px solid #22c55e; border-right: 3px solid #4ade80; border-radius: 50%; animation: rotate 2s linear infinite;"></div>
                        <div style="position: absolute; width: 45px; height: 45px; top: 12.5px; left: 12.5px; border: 2px solid transparent; border-left: 2px solid #16a34a; border-bottom: 2px solid #16a34a; border-radius: 50%; animation: rotate 3s linear reverse infinite;"></div>
                    </div>
                    <div style="color: #22c55e; font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; margin-top: 20px; letter-spacing: 3px; animation: pulse 1.5s ease-in-out infinite; text-shadow: 0 0 10px rgba(34, 197, 94, 0.5);">BUILDING NEURAL LAYERS</div>
                    <div style="width: 100%; max-width: 300px; height: 3px; background: rgba(34, 197, 94, 0.1); border-radius: 2px; margin-top: 20px; overflow: hidden;">
                        <div style="height: 100%; background: linear-gradient(90deg, #22c55e, #4ade80, #22c55e); background-size: 200% 100%; border-radius: 2px; animation: dataFlow 2s ease infinite;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Build advanced model
            st.info("Building bidirectional LSTM with batch normalization...")
            model = build_lstm_model(lookback, units=lstm_units, dropout_rate=dropout_rate)
            
            # Training with epoch progress callback
            st.info("Training model with adaptive learning rate...")
            
            # Create a variable to track latest epoch info
            epoch_info = {"current": 0, "total": 300}
            
            def update_epoch_display(current_epoch, total_epochs, logs):
                """Callback to update epoch progress in UI."""
                epoch_info["current"] = current_epoch
                epoch_info["total"] = total_epochs
                
                # Calculate progress percentage
                progress_pct = (current_epoch / total_epochs) * 100
                
                with loading_placeholder.container():
                    st.markdown(f"""
                    <style>
                        @keyframes rotate {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
                        @keyframes dataFlow {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
                    </style>
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(13, 26, 13, 0.8), rgba(10, 10, 10, 0.9)); border-radius: 12px; border: 2px solid #22c55e; box-shadow: 0 0 30px rgba(34, 197, 94, 0.3); margin: 20px 0;">
                        <div style="position: relative; width: 70px; height: 70px;">
                            <div style="position: absolute; width: 100%; height: 100%; border: 3px solid transparent; border-top: 3px solid #22c55e; border-right: 3px solid #4ade80; border-radius: 50%; animation: rotate 2s linear infinite;"></div>
                            <div style="position: absolute; width: 45px; height: 45px; top: 12.5px; left: 12.5px; border: 2px solid transparent; border-left: 2px solid #16a34a; border-bottom: 2px solid #16a34a; border-radius: 50%; animation: rotate 3s linear reverse infinite;"></div>
                        </div>
                        <div style="color: #4ade80; font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; margin-top: 20px; letter-spacing: 3px; animation: pulse 1.5s ease-in-out infinite; text-shadow: 0 0 10px rgba(74, 222, 128, 0.5);">TRAINING EPOCH [{current_epoch}/{total_epochs}]</div>
                        <div style="color: #a0a0a0; font-family: 'JetBrains Mono', monospace; font-size: 12px; margin-top: 10px; letter-spacing: 1px;">PROGRESS: {progress_pct:.1f}%</div>
                        <div style="width: 100%; max-width: 300px; height: 4px; background: rgba(34, 197, 94, 0.1); border-radius: 2px; margin-top: 15px; overflow: hidden;">
                            <div style="height: 100%; width: {progress_pct}%; background: linear-gradient(90deg, #22c55e, #4ade80); border-radius: 2px; transition: width 0.3s ease; box-shadow: 0 0 10px rgba(34, 197, 94, 0.5);"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
            model, history = train_lstm_model(model, X_train, y_train, X_validate, y_validate, 
                                             epochs=300, batch_size=32, progress_callback=update_epoch_display)
            
            # Update loading status
            with loading_placeholder.container():
                st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(13, 26, 13, 0.8), rgba(10, 10, 10, 0.9)); border-radius: 12px; border: 2px solid #22c55e; box-shadow: 0 0 30px rgba(34, 197, 94, 0.3); margin: 20px 0;">
                    <div style="position: relative; width: 70px; height: 70px;">
                        <div style="position: absolute; width: 100%; height: 100%; border: 3px solid transparent; border-top: 3px solid #22c55e; border-right: 3px solid #4ade80; border-radius: 50%; animation: rotate 2s linear infinite;"></div>
                        <div style="position: absolute; width: 45px; height: 45px; top: 12.5px; left: 12.5px; border: 2px solid transparent; border-left: 2px solid #16a34a; border-bottom: 2px solid #16a34a; border-radius: 50%; animation: rotate 3s linear reverse infinite;"></div>
                    </div>
                    <div style="color: #22c55e; font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; margin-top: 20px; letter-spacing: 3px; animation: pulse 1.5s ease-in-out infinite; text-shadow: 0 0 10px rgba(34, 197, 94, 0.5);">EVALUATING PREDICTIONS</div>
                    <div style="width: 100%; max-width: 300px; height: 3px; background: rgba(34, 197, 94, 0.1); border-radius: 2px; margin-top: 20px; overflow: hidden;">
                        <div style="height: 100%; background: linear-gradient(90deg, #22c55e, #4ade80, #22c55e); background-size: 200% 100%; border-radius: 2px; animation: dataFlow 2s ease infinite;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Comprehensive evaluation
            st.success("Training complete! Evaluating on all datasets...")
            (metrics, train_pred, validate_pred, test_pred, y_train_actual, 
             y_validate_actual, y_test_actual, residuals) = evaluate_lstm_model(
                model, X_train, X_validate, X_test, y_train, y_validate, y_test, scaler
            )
            
            # Update loading status
            with loading_placeholder.container():
                st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(13, 26, 13, 0.8), rgba(10, 10, 10, 0.9)); border-radius: 12px; border: 2px solid #22c55e; box-shadow: 0 0 30px rgba(34, 197, 94, 0.3); margin: 20px 0;">
                    <div style="position: relative; width: 70px; height: 70px;">
                        <div style="position: absolute; width: 100%; height: 100%; border: 3px solid transparent; border-top: 3px solid #22c55e; border-right: 3px solid #4ade80; border-radius: 50%; animation: rotate 2s linear infinite;"></div>
                        <div style="position: absolute; width: 45px; height: 45px; top: 12.5px; left: 12.5px; border: 2px solid transparent; border-left: 2px solid #16a34a; border-bottom: 2px solid #16a34a; border-radius: 50%; animation: rotate 3s linear reverse infinite;"></div>
                    </div>
                    <div style="color: #4ade80; font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; margin-top: 20px; letter-spacing: 3px; animation: pulse 1.5s ease-in-out infinite; text-shadow: 0 0 10px rgba(74, 222, 128, 0.5);">MONTE CARLO FORECAST</div>
                    <div style="width: 100%; max-width: 300px; height: 3px; background: rgba(34, 197, 94, 0.1); border-radius: 2px; margin-top: 20px; overflow: hidden;">
                        <div style="height: 100%; background: linear-gradient(90deg, #4ade80, #22c55e, #4ade80); background-size: 200% 100%; border-radius: 2px; animation: dataFlow 2s ease infinite;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Generate future predictions with uncertainty
            st.info(f"Generating {future_days}-day forecast with Monte Carlo uncertainty...")
            last_sequence = X_test[-1] if len(X_test) > 0 else X_train[-1]
            (future_mean, future_lower, future_upper, future_std, 
             future_samples) = predict_future_prices_with_uncertainty(
                model, last_sequence, scaler, future_days, num_samples=100
            )
            
            # Clear loading display
            loading_placeholder.empty()
            
            # Store all results
            st.session_state.lstm_model = model
            st.session_state.lstm_history = history
            st.session_state.lstm_metrics = metrics
            st.session_state.lstm_scaler = scaler
            st.session_state.lstm_train_pred = train_pred
            st.session_state.lstm_validate_pred = validate_pred
            st.session_state.lstm_test_pred = test_pred
            st.session_state.lstm_y_train = y_train_actual
            st.session_state.lstm_y_validate = y_validate_actual
            st.session_state.lstm_y_test = y_test_actual
            st.session_state.lstm_residuals = residuals
            st.session_state.lstm_raw_data = raw_data
            st.session_state.lstm_ticker = ticker
            st.session_state.lstm_future_mean = future_mean
            st.session_state.lstm_future_lower = future_lower
            st.session_state.lstm_future_upper = future_upper
            st.session_state.lstm_future_std = future_std
            st.session_state.lstm_future_samples = future_samples
            st.success("Advanced forecast complete with uncertainty quantification!")
        
        except ValueError as e:
            st.error(f"Data Error: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
    
    # Display results if available
    if 'lstm_future_mean' in st.session_state:
        st.markdown("---")
        st.markdown("## Advanced Results & Analysis")
        
        # Training dynamics
        st.subheader("Model Training Dynamics")
        st.plotly_chart(
            plot_lstm_training_history(st.session_state.lstm_history),
            use_container_width=True
        )
        
        # Predictions with uncertainty
        st.subheader("Price Predictions with 95% Confidence Interval")
        st.plotly_chart(
            plot_predictions(
                st.session_state.lstm_raw_data,
                st.session_state.lstm_train_pred,
                st.session_state.lstm_validate_pred,
                st.session_state.lstm_test_pred,
                st.session_state.lstm_future_mean,
                st.session_state.lstm_future_lower,
                st.session_state.lstm_future_upper,
                lookback,
                st.session_state.lstm_future_std
            ),
            use_container_width=True
        )
        
        # Performance metrics
        st.subheader("Comprehensive Performance Metrics")
        st.plotly_chart(
            plot_model_metrics(st.session_state.lstm_metrics),
            use_container_width=True
        )
        
        # Advanced metrics breakdown
        st.markdown("---")
        st.subheader("Advanced Model Diagnostics")
        
        tab1, tab2, tab3 = st.tabs(["Accuracy Metrics", "Risk Metrics", "Residual Analysis"])
        
        with tab1:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Train R² Score", f"{st.session_state.lstm_metrics['Train R²']:.4f}",
                         help="Goodness of fit on training data")
                st.metric("Validate R² Score", f"{st.session_state.lstm_metrics['Validate R²']:.4f}",
                         help="Goodness of fit on validation data")
                st.metric("Test R² Score", f"{st.session_state.lstm_metrics['Test R²']:.4f}",
                         help="Final test set performance")
            
            with metrics_col2:
                st.metric("Train MAE", f"${st.session_state.lstm_metrics['Train MAE']:.2f}")
                st.metric("Validate MAE", f"${st.session_state.lstm_metrics['Validate MAE']:.2f}")
                st.metric("Test MAE", f"${st.session_state.lstm_metrics['Test MAE']:.2f}")
            
            with metrics_col3:
                st.metric("Train RMSE", f"${st.session_state.lstm_metrics['Train RMSE']:.2f}")
                st.metric("Validate RMSE", f"${st.session_state.lstm_metrics['Validate RMSE']:.2f}")
                st.metric("Test RMSE", f"${st.session_state.lstm_metrics['Test RMSE']:.2f}")
        
        with tab2:
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            with risk_col1:
                st.metric("Test MAPE", f"{st.session_state.lstm_metrics['Test MAPE']*100:.2f}%",
                         help="Mean Absolute Percentage Error")
                st.metric("MSE", f"{st.session_state.lstm_metrics['Test MSE']:.6f}",
                         help="Mean Squared Error")
            
            with risk_col2:
                st.metric("Directional Accuracy", f"{st.session_state.lstm_metrics['Directional Accuracy']*100:.1f}%",
                         help="% time model predicted correct direction")
                st.metric("Residual Std Dev", f"${st.session_state.lstm_metrics['Residual Std']:.2f}",
                         help="Prediction error volatility")
            
            with risk_col3:
                acc_interpretation = "Excellent" if st.session_state.lstm_metrics['Test R²'] > 0.8 else "Good" if st.session_state.lstm_metrics['Test R²'] > 0.6 else "Fair"
                st.metric("Model Quality", acc_interpretation,
                         help="Based on R² score")
                st.metric("Residual Skewness", f"{st.session_state.lstm_metrics['Residual Skewness']:.3f}",
                         help="Distribution asymmetry")
        
        with tab3:
            st.write("**Residual Statistics:**")
            residuals_stats_col1, residuals_stats_col2, residuals_stats_col3 = st.columns(3)
            
            residuals = st.session_state.lstm_residuals
            with residuals_stats_col1:
                st.metric("Mean Residual", f"${np.mean(residuals):.4f}",
                         help="Should be close to 0")
                st.metric("Min Residual", f"${np.min(residuals):.2f}")
            
            with residuals_stats_col2:
                st.metric("Median Residual", f"${np.median(residuals):.4f}")
                st.metric("Max Residual", f"${np.max(residuals):.2f}")
            
            with residuals_stats_col3:
                percentile_95 = np.percentile(np.abs(residuals), 95)
                st.metric("95th Percentile Error", f"${percentile_95:.2f}",
                         help="Max typical error")
        
        # Forecast summary
        st.markdown("---")
        st.subheader("Forecast Summary with Uncertainty Quantification")
        
        current_price = st.session_state.lstm_raw_data[-1][0]
        forecast_end_price = st.session_state.lstm_future_mean[-1]
        price_change = forecast_end_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with summary_col2:
            st.metric("Mean Forecast", f"${forecast_end_price:.2f}",
                     delta=f"${price_change:.2f} ({price_change_pct:+.2f}%)")
        
        with summary_col3:
            min_forecast = st.session_state.lstm_future_lower[-1]
            max_forecast = st.session_state.lstm_future_upper[-1]
            st.metric("95% CI Range", f"${max_forecast - min_forecast:.2f}",
                     help=f"${min_forecast:.2f} to ${max_forecast:.2f}")
        
        with summary_col4:
            uncertainty_coeff = (st.session_state.lstm_future_std[-1] / forecast_end_price * 100)
            st.metric("Uncertainty", f"±{uncertainty_coeff:.2f}%",
                     help="1 standard deviation relative to forecast")
        
        # Forecast table with confidence bands
        with st.expander("View Detailed Forecast with Confidence Intervals"):
            forecast_data = pd.DataFrame({
                'Day': range(1, len(st.session_state.lstm_future_mean) + 1),
                'Mean Forecast': st.session_state.lstm_future_mean,
                'Lower 95% CI': st.session_state.lstm_future_lower,
                'Upper 95% CI': st.session_state.lstm_future_upper,
                'Std Dev': st.session_state.lstm_future_std,
                'CI Width': st.session_state.lstm_future_upper - st.session_state.lstm_future_lower
            })
            st.dataframe(forecast_data, use_container_width=True)
        
        # Model information
        with st.expander("Model Architecture & Training Details"):
            st.markdown(f"""
            ### Model Specifications
            - **Architecture:** Bidirectional LSTM with Batch Normalization
            - **LSTM Units:** {lstm_units * 2} (bidirectional) + {lstm_units} (second layer) + {lstm_units // 4} (third layer)
            - **Dropout Rate:** {dropout_rate:.0%}
            - **Lookback Window:** {lookback} trading days
            - **Regularization:** L2 (0.001) on all layers
            
            ### Training Configuration
            - **Optimizer:** Adaptive Adam (learning rate: 0.001, gradient clipping: 1.0)
            - **Early Stopping:** Patience=15 epochs
            - **Learning Rate Reduction:** Factor=0.5 on plateau
            - **Data Split:** 75% train, 10% validation, 15% test
            - **Scaling:** RobustScaler (handles outliers better)
            
            ### Uncertainty Quantification
            - **Method:** Monte Carlo Dropout (100 samples)
            - **Confidence Level:** 95% (±1.96 standard deviations)
            - **Ensemble Size:** 100 forward passes
            
            ### Model Robustness Features
            • Bidirectional processing of sequences
            • Batch normalization for training stability
            • Multiple dense layers for feature extraction
            • Adaptive learning rate scheduling
            • Built-in regularization (L2 + Dropout)
            • 3-way data split (train/validate/test)
            • Residual analysis for error assessment
            • Directional accuracy metrics
            • Monte Carlo uncertainty quantification
            """)
        
        # Disclaimer
        st.markdown("""
        ---
        ### Critical Disclaimer
        
        **This is a research/educational model - NOT financial advice:**
        
        **Limitations:**
        - Neural networks can overfit/underfit
        - Assumes historical patterns continue
        - Vulnerable to black swan events
        - Cannot predict regulatory changes
        - No consideration of sentiment/news
        
        **Error Sources:**
        - Model uncertainty (epistemic)
        - Data uncertainty (aleatoric)
        - Inherent market unpredictability
        - Regime changes and structural breaks
        
        **Best Practices:**
        - Use as ONE of multiple analysis tools
        - Backtest on historical data first
        - Combine with fundamental analysis
        - Risk management is critical
        - Consult financial advisors
        
        **Confidence Decreases With:**
        - Longer forecast horizons
        - High market volatility
        - Low historical data quality
        - Structural market changes
        """)

if __name__ == "__main__":
    main()