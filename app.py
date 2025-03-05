import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# Set page title and style
st.set_page_config(page_title="LSTM Stock Price Prediction", layout="wide")
st.title("📈 LSTM Stock Price Prediction")

# Load the pre-trained LSTM model
model = load_model('lstm_model.h5')

# Sidebar for user inputs
st.sidebar.header("Enter Stock Details")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TCS.NS, RELIANCE.NS)").strip().upper()

# Prediction options
option = st.sidebar.radio("What do you want to predict?", 
                          ["Specific Date", "Next 7 Days (1 Week)", "Next 30 Days (1 Month)"])

if not ticker:
    st.warning("⚠️ Please enter a stock ticker to proceed.")
    st.stop()

# Fetch data from yfinance
@st.cache_data
def fetch_stock_data(ticker):
    try:
        data = yf.download(ticker, period="5y", interval="1d")
        if data.empty:
            st.error(f"No data found for ticker: {ticker}")
            st.stop()
        return data['Close'].values.reshape(-1, 1)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

data = fetch_stock_data(ticker)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare sequences for prediction (last 60 days)
def prepare_sequence():
    return scaled_data[-60:]

# Prediction logic
def predict_future_prices(days):
    sequence = prepare_sequence().reshape(1, 60, 1)
    predictions = []
    
    for _ in range(days):
        predicted_price = model.predict(sequence)[0][0]
        predictions.append(predicted_price)
        sequence = np.append(sequence[:, 1:, :], [[[predicted_price]]], axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Main Prediction Flow
predict_button = False  # Track when to show Predict button

if option == "Specific Date":
    future_date = st.sidebar.date_input("Select a future date", datetime.date.today() + datetime.timedelta(days=1))
    if future_date:
        predict_button = st.sidebar.button("🔮 Predict")

elif option in ["Next 7 Days (1 Week)", "Next 30 Days (1 Month)"]:
    predict_button = st.sidebar.button("🔮 Predict")

# Perform Prediction only when Predict button is clicked
if predict_button:
    st.toast("🔍 Fetching data and preparing prediction...", icon="⏳")
    
    if option == "Specific Date":
        today = datetime.date.today()
        days_diff = (future_date - today).days

        if days_diff < 1:
            st.error("Please select a valid future date.")
            st.stop()

        predicted_price = predict_future_prices(days_diff)[-1]
        
        st.success(f"✅ Prediction completed for {future_date}!")

        st.markdown(f"""
        <div style="
            padding: 15px;
            background-color: #f0f9ff;
            border: 2px solid #0077cc;
            border-radius: 10px;
            color: #004466;
            font-size: 20px;
            text-align: center;
        ">
            📅 The approximate predicted stock price for **{ticker}** on **{future_date}** is: <br><br>
            <b style="font-size: 26px;">₹{predicted_price:.2f}</b>
        </div>
        """, unsafe_allow_html=True)

    elif option == "Next 7 Days (1 Week)":
        predictions = predict_future_prices(7)

        st.success("✅ Prediction completed for the next 7 days!")

        st.subheader(f"📊 Approximate Predicted Prices for Next 7 Days ({ticker})")
        future_dates = [(datetime.date.today() + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]

        df = pd.DataFrame({"Date": future_dates, "Predicted Price (₹)": predictions})
        st.table(df)

        plt.figure(figsize=(10, 5))
        plt.plot(future_dates, predictions, color='green', marker='o', linestyle='--', label='Predicted Price')
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.title(f'{ticker} - Next 7 Days Prediction')
        st.pyplot(plt)

    elif option == "Next 30 Days (1 Month)":
        predictions = predict_future_prices(30)

        st.success("✅ Prediction completed for the next 30 days!")

        st.subheader(f"📊 Approximate Predicted Prices for Next 30 Days ({ticker})")
        future_dates = [(datetime.date.today() + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(30)]

        df = pd.DataFrame({"Date": future_dates, "Predicted Price (₹)": predictions})
        st.table(df)

        plt.figure(figsize=(10, 5))
        plt.plot(future_dates, predictions, color='blue', marker='o', linestyle='--', label='Predicted Price')
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.title(f'{ticker} - Next 30 Days Prediction')
        st.pyplot(plt)

    st.toast("📊 Visualization displayed successfully!", icon="📈")

# Footer
st.markdown("---")
st.markdown("""
📌 **Note:** This prediction is only an approximation based on historical data using LSTM models.""")
