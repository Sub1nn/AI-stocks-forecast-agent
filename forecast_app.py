import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# App configuration
st.set_page_config(layout="wide")
st.title("AI Time Series Forecast Agent")
st.markdown("Your LSTM-based Stock Forecast AI")

# Sidebar controls
with st.sidebar:
    st.header("Stock Forecast AI Copilot")
    
    uploaded_file = st.file_uploader("Upload Stock Data (CSV)", type=["csv"])
    st.subheader("Model Parameters")
    sequence_length = st.slider("Sequence Length", 10, 90, 60)
    epochs = st.slider("Epochs", 10, 100, 30)
    batch_size = st.slider("Batch Size", 16, 128, 32)
    
    st.subheader("Forecast Parameters")
    forecast_steps = st.slider("Forecast Steps", 5, 90, 30)
    
    use_demo = st.checkbox("Use demo data (AMZN 2006-2018)")

# Load data function
def load_data():
    if use_demo:
        data_path = 'https://raw.githubusercontent.com/Sub1nn/AI-stocks-forecast-agent/main/AMZN_2006-01-01_to_2018-01-01.csv'
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df
    else:
        return None

# Feature engineering
def engineer_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = (df['High'] - df['Low']) / df['Low']
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_5'] = df['Close'].shift(5)
    df['Lag_10'] = df['Close'].shift(10)
    df['Lag_15'] = df['Close'].shift(15)
    df['Lag_20'] = df['Close'].shift(20)
    df['Rolling_Mean_10'] = df['Close'].rolling(window=10).mean()
    df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df

# Model building
def build_model(sequence_length, num_features):
    inputs = Input(shape=(sequence_length, num_features))
    x = Dense(64, activation="relu")(inputs)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Multi-step forecast
def multi_step_forecast(model, X_input, forecast_steps, sequence_length, num_features):
    forecast = []
    current_input = X_input[-1].reshape(1, sequence_length, num_features)
    
    for _ in range(forecast_steps):
        pred = model.predict(current_input, verbose=0)
        forecast.append(pred[0, 0])
        current_input = np.roll(current_input, shift=-1, axis=1)
        current_input[0, -1, 3] = pred  # Assuming 'Close' is at index 3
    return np.array(forecast)

# Denormalization helper
def denormalize(data, scaler, feature_index=3):
    dummy = np.zeros((data.shape[0], scaler.n_features_in_))
    dummy[:, feature_index] = data.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index]

# Cached model training
@st.cache_resource
def train_model(X_train, Y_train, sequence_length, num_features, epochs, batch_size):
    model = build_model(sequence_length, num_features)
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )
    return model, history

# Get data
df = load_data()

# Forecast button
if df is not None:
    st.subheader("📈 Click the button below to start forecasting")
    if st.button("Start Forecasting"):
        df = engineer_features(df)
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility', 
                    'Lag_1', 'Lag_5', 'Lag_10', 'Lag_15', 'Lag_20', 
                    'Rolling_Mean_10', 'Rolling_Mean_20']
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[features])
        
        X, Y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            Y.append(scaled_data[i + sequence_length, 3])
        
        X, Y = np.array(X), np.array(Y)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        with st.spinner("Training LSTM model..."):
            model, history = train_model(X_train, Y_train, sequence_length, len(features), epochs, batch_size)
        
        predictions = model.predict(X_test, verbose=0)
        Y_test_denorm = denormalize(Y_test, scaler)
        predictions_denorm = denormalize(predictions, scaler)

        forecast = multi_step_forecast(model, X_test, forecast_steps, sequence_length, len(features))
        forecast_denorm = denormalize(forecast, scaler)
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='B')[1:]

        st.header("Forecast Results")
        tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Forecast", "Model Metrics"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-len(Y_test):], y=Y_test_denorm, name="Actual", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index[-len(Y_test):], y=predictions_denorm, name="Predicted", line=dict(color='orange')))
            fig.update_layout(title="Actual vs Predicted Closing Price", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-len(Y_test):], y=Y_test_denorm, name="Actual", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index[-len(Y_test):], y=predictions_denorm, name="Predicted", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_denorm, name="Forecast", line=dict(color='green', dash='dot')))
            fig.update_layout(title=f"{forecast_steps}-Day Forecast", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            mse = mean_squared_error(Y_test_denorm, predictions_denorm)
            r2 = r2_score(Y_test_denorm, predictions_denorm)
            mape = mean_absolute_percentage_error(Y_test_denorm, predictions_denorm) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{mse:.2f}")
            col2.metric("R-squared", f"{r2:.4f}")
            col3.metric("MAPE (%)", f"{mape:.2f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['loss'], name="Training Loss", line=dict(color='blue')))
            fig.add_trace(go.Scatter(y=history.history['val_loss'], name="Validation Loss", line=dict(color='orange')))
            fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_denorm})
        csv = forecast_df.to_csv(index=False)
        st.download_button("Download forecast data", data=csv, file_name='stock_forecast.csv', mime='text/csv')
else:
    st.warning("Please upload stock data or use demo data.")
