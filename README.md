# stock_price_predictor
# 📈 Stock Price Predictor using LSTM

### 🧠 Overview
This project predicts future stock prices using historical data and a Long Short-Term Memory (LSTM) deep learning model.  
The goal is to understand market trends and visualize potential stock movements through an interactive Streamlit web app.

### ⚙️ Tools & Technologies
- **Python**
- **Streamlit** – for building the web interface  
- **Keras / TensorFlow** – for LSTM deep learning model  
- **yFinance** – for fetching live stock data  
- **Pandas, NumPy, Matplotlib** – for data analysis and visualization  
- **scikit-learn** – for data scaling and preprocessing  

### 🚀 Features
- Fetches historical stock data using Yahoo Finance API  
- Predicts stock prices for:
  - A specific future date  
  - Next 7 days  
  - Next 30 days  
- Interactive Streamlit dashboard for data visualization  
- Real-time prediction updates and user-friendly interface  

### 📂 Project Structure
```
│
├── app.py # Streamlit application
├── lstm_model.h5 # Pre-trained LSTM model
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```
### 🔍 How to Run the App
### 1. Clone this repository
   git clone https://github.com/VasundraSri/Stock-Price-Predictor.git
   cd Stock-Price-Predictor
### 2. Install dependencies
pip install -r requirements.txt
### 3. Run the app
streamlit run app.py
### 📊 Output Example
Displays predicted prices for upcoming days

Visualizes the trend using line charts and tables

Provides an easy-to-use sidebar for stock selection and prediction duration

### 📌 Note
This project is for educational and learning purposes only.
Predictions are based on historical data and should not be used for financial decisions.
