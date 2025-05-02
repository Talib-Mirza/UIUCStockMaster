import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

# Set page config for better layout
st.set_page_config(layout="wide", page_title="Tesla Stock Predictor")

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def get_latest_tesla_data(days=7):
    data = yf.download("TSLA", period=f"{days}d", interval="1d")
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def get_intraday_data():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Get 3 days of data
        data = yf.download("TSLA", start=start_date, end=end_date, interval="1m")
        if data.empty:
            # If no hourly data, fall back to daily data
            data = yf.download("TSLA", period="3d", interval="1d")
        return data
    except Exception as e:
        # Fallback to daily data if hourly fails
        return yf.download("TSLA", period="3d", interval="1d")

def create_rolling_window(data, window_size=7):
    X = []
    for i in range(window_size):
    # Get all features for the past 7 days
        X.append(data.iloc[i-window_size:i].values)
    return np.array(X)

def scale_data(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    if hasattr(X, 'values'):
        X = X.values
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    X_scaled = scaler.fit_transform(X)
    return X_scaled

model = load_model('models\lstm_tsla.h5')

# Main title
st.title("🚀 Tesla Stock Analysis & Prediction")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Live candlestick chart
    st.subheader("TSLA Price Chart (Last 3 Days)")
    try:
        chart_data = get_intraday_data()
        chart_data.index = pd.to_datetime(chart_data.index)
        if not chart_data.empty:
            fig = go.Figure(data=[go.Candlestick(x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'])])
            fig.update_layout(
                autosize=False,
                xaxis_rangeslider_visible=False,
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Unable to fetch price data. Please try again later.")
    except Exception as e:
        st.error("Error displaying price chart. Please try again later.")

with col2:
    # Key metrics
    st.subheader("Key Metrics")

        # Get 2 days of data to ensure we have previous close
    latest_data = yf.download("TSLA", period="2d", interval="1d")
    if not latest_data.empty and len(latest_data) >= 2:
        current_price = latest_data['Close'].iloc[-1]
        prev_close = latest_data['Close'].iloc[-2]
        price_change = current_price - prev_close
        percent_change = (price_change / prev_close) * 100
        current_price = float(current_price)
        price_change = float(price_change)
        percent_change = float(percent_change)
        
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Current Price</h3>
            <h4>${current_price:.2f}</h2>
            <p style="color: {'green' if price_change >= 0 else 'red'}">
                {price_change:.2f} ({percent_change:.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Volume
        volume = latest_data['Volume'].iloc[-1]
        volume = float(volume)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Volume</h3>
            <h4>{volume:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No data available for metrics calculation")

# Prediction section
st.markdown("---")
st.subheader("Price Direction Prediction")

# Get data for prediction
data = get_latest_tesla_data()

print(data)
input_data = scale_data(data)
input_data = input_data.reshape(1, 7, 5)

# Create prediction button with better styling
if st.button("🔮 Predict Next Day's Direction", use_container_width=True):
    pred = model.predict(input_data)
    direction = "Up" if pred > 0.5 else "Down"
    confidence = pred[0][0] if direction == "Up" else 1 - pred[0][0]
    
    st.markdown(f"""
    <div class="prediction-card">
        <h2>Prediction: {'📈' if direction == 'Up' else '📉'} {direction}</h2>
    </div>
    """, unsafe_allow_html=True)

# Add some technical indicators
st.markdown("---")
st.subheader("Technical Indicators")

# Calculate and display some basic technical indicators
col3, col4, col5 = st.columns(3)

with col3:
    try:
        # Simple Moving Average
        sma_5 = data['Close'].rolling(window=5).mean().iloc[-1]
        st.metric("5-Day SMA", f"${float(sma_5):.2f}")
    except Exception as e:
        st.error("Unable to calculate SMA")

with col4:
    try:
        # RSI (simplified)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        st.metric("RSI (7)", f"{float(rsi.iloc[-1]):.2f}")
    except Exception as e:
        st.error("Unable to calculate RSI")

with col5:
    try:
        # Volume change
        volume_change = ((data['Volume'].iloc[-1] - data['Volume'].iloc[-2]) / data['Volume'].iloc[-2]) * 100
        st.metric("Volume Change", f"{float(volume_change):.2f}%")
    except Exception as e:
        st.error("Unable to calculate volume change")