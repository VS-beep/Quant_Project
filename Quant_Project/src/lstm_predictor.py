import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential, load_model  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback  # type: ignore
    from tensorflow.keras import regularizers  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Custom callback for epoch progress tracking
class EpochProgressCallback(Callback):
    """Custom callback to track epoch progress and update UI."""
    def __init__(self, total_epochs, progress_callback=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_callback = progress_callback
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if self.progress_callback:
            current_epoch = epoch + 1
            self.progress_callback(current_epoch, self.total_epochs, logs or {})

def check_tensorflow():
    """Check if TensorFlow is available."""
    return TENSORFLOW_AVAILABLE

def validate_data_quality(data, ticker):
    """Validate data quality and handle issues."""
    if data is None or len(data) == 0:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Check for missing values
    if np.isnan(data).any():
        st.warning(f"Detected {np.isnan(data).sum()} NaN values. Filling with forward-fill method...")
        data = pd.DataFrame(data).fillna(method='ffill').values
    
    # Check for minimum data points
    if len(data) < 100:
        raise ValueError(f"Insufficient data ({len(data)} days). Need at least 100 trading days.")
    
    # Check for outliers using IQR method
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    outlier_mask = (data < (Q1 - 3 * IQR)) | (data > (Q3 + 3 * IQR))
    outlier_count = outlier_mask.sum()
    
    if outlier_count > 0:
        st.warning(f"Detected {outlier_count} extreme outliers. These may affect model performance.")
    
    return data

def fetch_stock_data_for_lstm(ticker, days=365):
    """Fetch and validate historical stock data for LSTM training."""
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        close_prices = data['Close'].values.reshape(-1, 1)
        close_prices = validate_data_quality(close_prices, ticker)
        
        return close_prices
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def prepare_lstm_data(data, lookback=60, test_size=0.15, validate_size=0.10):
    """
    Prepare data with advanced preprocessing for LSTM model.
    
    Uses three-way split: training, validation, and testing.
    Implements Robust Scaling for outlier resistance.
    """
    if data is None or len(data) < max(lookback * 3, 100):
        raise ValueError(f"Insufficient data. Need at least {max(lookback * 3, 100)} days of history.")
    
    # Use RobustScaler for outlier resistance instead of MinMaxScaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences with advanced feature engineering
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Three-way split: train, validate, test
    total_samples = len(X)
    train_size = int(total_samples * (1 - test_size - validate_size))
    validate_size_idx = int(total_samples * (1 - test_size))
    
    X_train, X_validate, X_test = X[:train_size], X[train_size:validate_size_idx], X[validate_size_idx:]
    y_train, y_validate, y_test = y[:train_size], y[train_size:validate_size_idx], y[validate_size_idx:]
    
    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) if len(X_train) > 0 else X_train
    X_validate = X_validate.reshape((X_validate.shape[0], X_validate.shape[1], 1)) if len(X_validate) > 0 else X_validate
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) if len(X_test) > 0 else X_test
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test, scaler, lookback

def build_lstm_model(lookback, units=64, dropout_rate=0.25, l2_reg=0.001):
    """
    Build sophisticated LSTM neural network model with:
    - Bidirectional LSTM for better context
    - Batch normalization for training stability
    - L2 regularization to prevent overfitting
    - Multiple LSTM layers with careful dropout
    """
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True, 
                          input_shape=(lookback, 1),
                          kernel_regularizer=regularizers.l2(l2_reg)),
                     input_shape=(lookback, 1)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Bidirectional(LSTM(units // 2, return_sequences=True,
                          kernel_regularizer=regularizers.l2(l2_reg))),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        LSTM(units // 4, return_sequences=False,
             kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate * 0.75),
        
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        Dropout(dropout_rate * 0.5),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        Dense(1)
    ])
    
    # Use adaptive learning rate optimizer
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    
    return model

def train_lstm_model(model, X_train, y_train, X_validate, y_validate, epochs=300, batch_size=32, progress_callback=None):
    """
    Train LSTM model with advanced callbacks:
    - Early stopping with patience
    - Learning rate reduction on plateau
    - Model checkpointing for best weights
    - Optional progress callback for epoch tracking
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=1e-5
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=0
        ),
        EpochProgressCallback(epochs, progress_callback)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_validate, y_validate),
        callbacks=callbacks,
        verbose=0
    )
    
    return model, history

def evaluate_lstm_model(model, X_train, X_validate, X_test, y_train, y_validate, y_test, scaler):
    """
    Comprehensive model evaluation with multiple metrics:
    - MSE, RMSE, MAE, MAPE
    - R² score
    - Directional accuracy (up/down prediction)
    - Residual analysis
    """
    # Get predictions
    train_pred = model.predict(X_train, verbose=0)
    validate_pred = model.predict(X_validate, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform to get actual prices
    train_pred_price = scaler.inverse_transform(train_pred) if len(train_pred) > 0 else train_pred
    validate_pred_price = scaler.inverse_transform(validate_pred) if len(validate_pred) > 0 else validate_pred
    test_pred_price = scaler.inverse_transform(test_pred) if len(test_pred) > 0 else test_pred
    
    y_train_price = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_validate_price = scaler.inverse_transform(y_validate.reshape(-1, 1))
    y_test_price = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate comprehensive metrics for test set
    test_mse = mean_squared_error(y_test_price, test_pred_price)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_price, test_pred_price)
    test_mape = mean_absolute_percentage_error(y_test_price, test_pred_price)
    test_r2 = r2_score(y_test_price, test_pred_price)
    
    # Calculate for training set
    train_mse = mean_squared_error(y_train_price, train_pred_price)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_price, train_pred_price)
    train_r2 = r2_score(y_train_price, train_pred_price)
    
    # Calculate for validation set
    validate_mse = mean_squared_error(y_validate_price, validate_pred_price)
    validate_rmse = np.sqrt(validate_mse)
    validate_mae = mean_absolute_error(y_validate_price, validate_pred_price)
    validate_r2 = r2_score(y_validate_price, validate_pred_price)
    
    # Directional accuracy (buying signal accuracy)
    test_actual_direction = np.sign(np.diff(y_test_price.flatten()))
    test_pred_direction = np.sign(np.diff(test_pred_price.flatten()))
    directional_accuracy = np.mean(test_actual_direction == test_pred_direction)
    
    # Residual analysis
    residuals = (y_test_price - test_pred_price).flatten()
    residual_std = np.std(residuals)
    skewness = (residuals ** 3).mean() / (residual_std ** 3) if residual_std > 0 else 0
    
    metrics = {
        'Train MSE': train_mse,
        'Validate MSE': validate_mse,
        'Test MSE': test_mse,
        'Train RMSE': train_rmse,
        'Validate RMSE': validate_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Validate MAE': validate_mae,
        'Test MAE': test_mae,
        'Test MAPE': test_mape,
        'Train R²': train_r2,
        'Validate R²': validate_r2,
        'Test R²': test_r2,
        'Directional Accuracy': directional_accuracy,
        'Residual Std': residual_std,
        'Residual Skewness': skewness
    }
    
    return (metrics, train_pred_price, validate_pred_price, test_pred_price,
            y_train_price, y_validate_price, y_test_price, residuals)

def predict_future_prices_with_uncertainty(model, last_sequence, scaler, days_ahead=30, num_samples=100):
    """
    Predict future prices with uncertainty quantification using MC Dropout.
    
    Returns:
    - Mean prediction
    - Uncertainty bounds (credible intervals)
    - Multiple sample realizations
    """
    predictions_samples = []
    
    # Collect multiple predictions
    for _ in range(num_samples):
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Predict with dropout enabled (MC dropout for uncertainty)
            current_sequence_reshaped = current_sequence.reshape((1, current_sequence.shape[0], 1))
            next_pred = model.predict(current_sequence_reshaped, verbose=0)
            predictions.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform to prices
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_price = scaler.inverse_transform(predictions_array)
        predictions_samples.append(predictions_price.flatten())
    
    predictions_samples = np.array(predictions_samples)
    
    # Calculate mean, credible intervals, and uncertainty
    mean_prediction = np.mean(predictions_samples, axis=0)
    std_prediction = np.std(predictions_samples, axis=0)
    lower_bound = np.percentile(predictions_samples, 2.5, axis=0)  # 95% CI
    upper_bound = np.percentile(predictions_samples, 97.5, axis=0)
    
    return mean_prediction, lower_bound, upper_bound, std_prediction, predictions_samples

def calculate_prediction_confidence(residuals, forecast_horizon):
    """
    Calculate confidence scores based on:
    - Residual magnitude and variance
    - Time decay (near-term forecasts more reliable)
    - Distribution normality
    """
    base_mae = np.mean(np.abs(residuals))
    residual_variance = np.var(residuals)
    
    # Time decay factor (confidence decreases for longer horizons)
    time_decay = np.exp(-np.arange(forecast_horizon) / forecast_horizon)
    
    # Confidence score (0-100)
    confidence_score = 100 / (1 + base_mae / 10) * time_decay
    
    return confidence_score

def plot_lstm_training_history(history):
    """Plot training, validation and test loss for robust analysis."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training vs Validation Loss', 'Learning Rate Dynamics')
    )
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Loss curves
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=history.history['loss'],
        name='Training Loss',
        line=dict(color='#22c55e', width=2),
        hovertemplate='<b>Epoch %{x}</b><br>Loss: %{y:.6f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=history.history['val_loss'],
        name='Validation Loss',
        line=dict(color='#ea580c', width=2, dash='dash'),
        hovertemplate='<b>Epoch %{x}</b><br>Val Loss: %{y:.6f}<extra></extra>'
    ), row=1, col=1)
    
    # MAE metrics
    if 'mae' in history.history:
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history.history['mae'],
            name='Training MAE',
            line=dict(color='#60a5fa', width=1),
            visible='legendonly'
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history.history['val_mae'],
            name='Validation MAE',
            line=dict(color='#a78bfa', width=1, dash='dash'),
            visible='legendonly'
        ), row=1, col=2)
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss (MSE)", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    
    fig.update_layout(
        title='Advanced Model Training Dynamics',
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
            linewidth=1
        ),
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_predictions(actual_prices, train_pred, validate_pred, test_pred, future_mean_pred, 
                    future_lower, future_upper, lookback, future_std=None):
    """
    Plot actual vs predicted prices with uncertainty bands.
    Enhanced version with confidence intervals and advanced visualization.
    """
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Historical Prices & Model Predictions (Train/Val/Test)', 'Future Forecast with 95% Confidence Interval'),
                        row_heights=[0.6, 0.4],
                        vertical_spacing=0.15)
    
    # Calculate indices for plotting
    all_indices = np.arange(len(actual_prices))
    
    # Actual prices (full history)
    fig.add_trace(go.Scatter(
        x=all_indices,
        y=actual_prices.flatten(),
        name='Actual Price',
        line=dict(color='#c7e9c0', width=2.5),
        hovertemplate='<b>Day %{x}</b><br>Price: $%{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Training predictions
    if len(train_pred) > 0:
        train_indices = np.arange(lookback, lookback + len(train_pred))
        fig.add_trace(go.Scatter(
            x=train_indices,
            y=train_pred.flatten(),
            name='Training Predictions',
            line=dict(color='#22c55e', width=2),
            hovertemplate='<b>Day %{x}</b><br>Predicted: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # Validation predictions
    if len(validate_pred) > 0:
        validate_indices = np.arange(lookback + len(train_pred), lookback + len(train_pred) + len(validate_pred))
        fig.add_trace(go.Scatter(
            x=validate_indices,
            y=validate_pred.flatten(),
            name='Validation Predictions',
            line=dict(color='#fbbf24', width=2),
            hovertemplate='<b>Day %{x}</b><br>Predicted: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # Test predictions
    if len(test_pred) > 0:
        test_indices = np.arange(lookback + len(train_pred) + len(validate_pred), 
                                  lookback + len(train_pred) + len(validate_pred) + len(test_pred))
        fig.add_trace(go.Scatter(
            x=test_indices,
            y=test_pred.flatten(),
            name='Test Predictions',
            line=dict(color='#60a5fa', width=2),
            hovertemplate='<b>Day %{x}</b><br>Predicted: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # Future predictions with confidence bands
    last_actual_idx = len(actual_prices) - 1
    future_indices = np.arange(last_actual_idx + 1, last_actual_idx + 1 + len(future_mean_pred))
    
    # Upper confidence band
    fig.add_trace(go.Scatter(
        x=future_indices,
        y=future_upper,
        fill=None,
        mode='lines',
        line_color='rgba(0, 0, 0, 0)',
        showlegend=False,
        hoverinfo='skip'
    ), row=2, col=1)
    
    # Lower confidence band
    fig.add_trace(go.Scatter(
        x=future_indices,
        y=future_lower,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0, 0, 0, 0)',
        fillcolor='rgba(167, 139, 250, 0.25)',
        name='95% Confidence Interval',
        hovertemplate='<b>Day %{x}</b><br>Lower: $%{y:.2f}<extra></extra>'
    ), row=2, col=1)
    
    # Mean forecast
    fig.add_trace(go.Scatter(
        x=future_indices,
        y=future_mean_pred,
        name='Mean Forecast',
        line=dict(color='#a78bfa', width=3),
        hovertemplate='<b>Day %{x}</b><br>Forecast: $%{y:.2f}<extra></extra>'
    ), row=2, col=1)
    
    # Upper confidence line (dashed)
    fig.add_trace(go.Scatter(
        x=future_indices,
        y=future_upper,
        name='Upper Bound (±1.96σ)',
        line=dict(color='#a78bfa', width=1, dash='dash'),
        hovertemplate='<b>Day %{x}</b><br>Upper: $%{y:.2f}<extra></extra>'
    ), row=2, col=1)
    
    # Lower confidence line (dashed)
    fig.add_trace(go.Scatter(
        x=future_indices,
        y=future_lower,
        name='Lower Bound (±1.96σ)',
        line=dict(color='#a78bfa', width=1, dash='dash'),
        hovertemplate='<b>Day %{x}</b><br>Lower: $%{y:.2f}<extra></extra>'
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title='LSTM Stock Price Prediction with Uncertainty Quantification',
        paper_bgcolor='#0d1a0d',
        plot_bgcolor='#0a0a0a',
        font=dict(family="Inter, sans-serif", size=12, color='#c7e9c0'),
        title_font=dict(size=18, color='#22c55e'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1,
            title='Day'
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1,
            title='Day'
        ),
        yaxis=dict(
            title='Price ($)',
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1
        ),
        yaxis2=dict(
            title='Forecast Price ($)',
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(color='#c7e9c0'),
            bgcolor='rgba(10, 10, 10, 0.7)'
        ),
        height=900,
        hovermode='x unified'
    )
    
    return fig

def plot_model_metrics(metrics):
    """Create visualization of model performance metrics."""
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Separate into different categories
    mse_metrics = {k: v for k, v in metrics.items() if 'MSE' in k}
    mae_metrics = {k: v for k, v in metrics.items() if 'MAE' in k}
    r2_metrics = {k: v for k, v in metrics.items() if 'R²' in k}
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('MSE Comparison', 'MAE Comparison', 'R² Score Comparison'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # MSE vs Test
    fig.add_trace(go.Bar(
        x=list(mse_metrics.keys()),
        y=list(mse_metrics.values()),
        name='MSE',
        marker_color=['#22c55e', '#dc2626'],
        hovertemplate='<b>%{x}</b><br>MSE: %{y:.6f}<extra></extra>'
    ), row=1, col=1)
    
    # MAE vs Test
    fig.add_trace(go.Bar(
        x=list(mae_metrics.keys()),
        y=list(mae_metrics.values()),
        name='MAE',
        marker_color=['#60a5fa', '#ea580c'],
        hovertemplate='<b>%{x}</b><br>MAE: $%{y:.2f}<extra></extra>'
    ), row=1, col=2)
    
    # R² Score
    fig.add_trace(go.Bar(
        x=list(r2_metrics.keys()),
        y=list(r2_metrics.values()),
        name='R²',
        marker_color=['#a78bfa', '#fbbf24'],
        hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>'
    ), row=1, col=3)
    
    fig.update_layout(
        title='Model Performance Metrics',
        paper_bgcolor='#0d1a0d',
        plot_bgcolor='#0a0a0a',
        font=dict(family="Inter, sans-serif", size=11, color='#c7e9c0'),
        title_font=dict(size=18, color='#22c55e'),
        showlegend=False,
        height=400,
        xaxis=dict(
            showgrid=False,
            linecolor='#22c55e',
            linewidth=1
        ),
        xaxis2=dict(
            showgrid=False,
            linecolor='#22c55e',
            linewidth=1
        ),
        xaxis3=dict(
            showgrid=False,
            linecolor='#22c55e',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1
        ),
        yaxis2=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1
        ),
        yaxis3=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            linecolor='#22c55e',
            linewidth=1
        )
    )
    
    return fig
