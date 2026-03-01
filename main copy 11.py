
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Bidirectional, LSTM, Dropout, Concatenate, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import joblib
from datetime import date, timedelta
import streamlit.components.v1 as components
from news_scraper import get_cafef_news, get_vietstock_news, get_ndh_news

# --- Constants ---
DATA_DIR = "data"
MODELS_DIR = "models"
SCALERS_DIR = "scalers"
FEATURES = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD']
LOOK_BACK = 21

# --- TradingView Widget HTML ---
market_overview_html = '''
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
    "symbols": [
      {"description": "VN-INDEX", "proName": "VCBF:^VNINDEX"},
      {"description": "FPT", "proName": "HOSE:FPT"},
      {"description": "Vietcombank", "proName": "HOSE:VCB"},
      {"description": "Vingroup", "proName": "HOSE:VIC"},
      {"description": "S&P 500", "proName": "OANDA:SPX500USD"},
      {"description": "Nasdaq 100", "proName": "OANDA:NAS100USD"},
      {"description": "Bitcoin", "proName": "BITSTAMP:BTCUSD"},
      {"description": "Ethereum", "proName": "BITSTAMP:ETHUSD"}
    ],
    "showSymbolLogo": true, "colorTheme": "dark", "isTransparent": false,
    "displayMode": "adaptive", "locale": "vi_VN"
  }
  </script>
</div>
<!-- TradingView Widget END -->
'''

# --- Helper Functions ---
def ensure_dir(directory_path):
    os.makedirs(directory_path, exist_ok=True)

# --- Technical Indicator Calculations (Internal) ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    # Avoid division by zero
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast_window=12, slow_window=26):
    ema_fast = data['Close'].ewm(span=fast_window, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_window, adjust=False).mean()
    return ema_fast - ema_slow

def add_technical_indicators(df):
    df_ta = df.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_ta[col] = pd.to_numeric(df_ta[col], errors='coerce')
    df_ta.dropna(inplace=True)
    df_ta['RSI'] = calculate_rsi(df_ta)
    df_ta['MACD'] = calculate_macd(df_ta)
    df_ta.dropna(inplace=True)
    return df_ta

# --- Data Loading and Processing ---
def get_valid_ticker(ticker):
    """Appends .VN to Vietnamese stock codes if they don't have it."""
    ticker = ticker.upper().strip()
    # Simple check for common Vietnamese stock codes (2-3 letters)
    if len(ticker) in [2, 3] and not ticker.endswith(('.VN', '.HNX', '.UPCOM')):
        return f"{ticker}.VN"
    return ticker

@st.cache_data(ttl=3600)
def get_or_download_data(ticker):
    ensure_dir(DATA_DIR)
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    try:
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            raise ValueError(f"Không có dữ liệu cho mã '{ticker}'. Mã có thể bị hủy niêm yết hoặc sai.")
        df.reset_index().to_csv(file_path, index=False)
    except Exception:
        if not os.path.exists(file_path):
            raise ValueError(f"Tải dữ liệu cho '{ticker}' thất bại. Vui lòng kiểm tra lại mã.")
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# --- Model Architecture ---
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn)
    cnn = Flatten()(cnn)
    bilstm = Bidirectional(LSTM(100, return_sequences=True))(inputs)
    bilstm = Dropout(0.2)(bilstm)
    bilstm = Bidirectional(LSTM(100, return_sequences=False))(bilstm)
    merged = Concatenate()([cnn, bilstm])
    outputs = Dense(100, activation='relu')(merged)
    outputs = Dense(1)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Training and Prediction Logic ---
def run_training(ticker):
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(SCALERS_DIR, f"{ticker}_scaler.pkl")
    with st.spinner(f"Đang huấn luyện mô hình cho {ticker}... (việc này có thể mất vài phút)"):
        data = get_or_download_data(ticker)
        data_processed = add_technical_indicators(data)
        if data_processed.empty or len(data_processed) < LOOK_BACK:
            st.error(f"Không đủ dữ liệu cho '{ticker}' để huấn luyện.")
            return None, None, None, None

        data_for_training = data_processed[FEATURES].astype(float)
        # We fit the scaler on feature names to avoid warnings later
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_for_training)

        X, y = create_sequences(scaled_data, LOOK_BACK)
        if len(X) == 0:
            st.error("Không thể tạo chuỗi huấn luyện. Dữ liệu quá ít.")
            return None, None, None, None

        model = build_model(input_shape=(LOOK_BACK, len(FEATURES)))
        model.fit(X, y, batch_size=32, epochs=50, verbose=0)
        
        ensure_dir(MODELS_DIR)
        ensure_dir(SCALERS_DIR)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

    last_sequence = scaled_data[-LOOK_BACK:]
    return model, scaler, last_sequence, data_processed

def predict_future(model, scaler, initial_input, days_to_predict):
    future_predictions = []
    current_input = initial_input.copy()
    num_features = len(FEATURES)
    for _ in range(days_to_predict):
        X_pred = np.reshape(current_input, (1, LOOK_BACK, num_features))
        predicted_price_scaled = model.predict(X_pred, verbose=0)[0, 0]

        dummy_array = np.zeros((1, num_features))
        dummy_array[0, 0] = predicted_price_scaled
        predicted_price_actual = scaler.inverse_transform(dummy_array)[0, 0]
        future_predictions.append(float(predicted_price_actual))

        new_row_scaled = current_input[-1, :].copy()
        new_row_scaled[0] = predicted_price_scaled
        new_row_scaled[1:4] = predicted_price_scaled # Open, High, Low approximation
        current_input = np.vstack([current_input[1:], new_row_scaled])

    return future_predictions

def run_prediction(ticker):
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(SCALERS_DIR, f"{ticker}_scaler.pkl")
    with st.spinner(f"Tải mô hình và dữ liệu cho {ticker}..."):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        data = get_or_download_data(ticker)
        if data.empty or len(data) < LOOK_BACK:
            raise ValueError("Không đủ dữ liệu gần đây để dự đoán.")

        data_processed = add_technical_indicators(data)
        data_filtered = data_processed[FEATURES].astype(float)
        scaled_data = scaler.transform(data_filtered) # Use transform, not fit_transform
        last_sequence = scaled_data[-LOOK_BACK:]

    return model, scaler, last_sequence, data_processed

# --- UI and Main Application ---
def display_news(news_function):
    try:
        news_items = news_function(max_items=7)
        if news_items:
            for item in news_items:
                st.markdown(f"▪️ [{item['title']}]({item['link']})", unsafe_allow_html=True)
        else:
            st.info("Không có tin tức mới hoặc không thể tải từ nguồn này.")
    except Exception as e:
        st.warning(f"Lỗi khi tải tin tức: {e}")

def display_results(df, ticker, future_prices):
    st.success(f"Dự đoán thành công! Giá đóng cửa ngày mai: **${future_prices[0]:,.2f}**")
    tab1, tab2 = st.tabs(['📈 Biểu đồ Dự đoán', '📋 Bảng Dữ liệu'])
    with tab1:
        historical_df = df.tail(90)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_df.index, historical_df['Close'], color='blue', label='Giá thực tế')
        last_actual_date = historical_df.index[-1]
        prediction_dates = pd.date_range(start=last_actual_date + timedelta(days=1), periods=len(future_prices))
        ax.plot(prediction_dates, future_prices, color='red', linestyle='--', label='Giá dự đoán')
        ax.set_title(f'Giá thực tế & Dự đoán cho {ticker}', fontsize=16)
        ax.set_ylabel('Giá Đóng cửa (USD)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with tab2:
        future_dates_table = [df.index[-1] + timedelta(days=i) for i in range(1, len(future_prices) + 1)]
        forecast_df = pd.DataFrame({'Ngày': future_dates_table, 'Giá dự đoán': future_prices})
        st.dataframe(forecast_df.set_index('Ngày').style.format({"Giá dự đoán": "${:,.2f}"}))

st.set_page_config(layout="wide", page_title="Dự đoán Giá Cổ phiếu")
st.markdown("<h1 style='text-align: center;'>💹 MÔ HÌNH DỰ ĐOÁN GIÁ CỔ PHIẾU</h1>", unsafe_allow_html=True)
components.html(market_overview_html, height=72)

col1, col2 = st.columns([1, 2])
with col1:
    st.header("⚙️ Thiết lập")
    ticker_input = st.text_input("Nhập mã cổ phiếu (VD: FPT, AAPL):", "FPT").upper()
    days_to_predict = st.number_input("Số ngày dự đoán (1-30):", 1, 30, 7)
    
    c1, c2 = st.columns(2)
    run_button = c1.button("🚀 Dự đoán", type="primary", use_container_width=True)
    force_train_button = c2.button("🔄 Huấn luyện lại", use_container_width=True)
    
    st.divider()
    st.subheader("📰 Tin tức thị trường")
    tab_news1, tab_news2, tab_news3 = st.tabs(["CafeF", "Vietstock", "NDH.vn"])
    with tab_news1: display_news(get_cafef_news)
    with tab_news2: display_news(get_vietstock_news)
    with tab_news3: display_news(get_ndh_news)

with col2:
    if run_button or force_train_button:
        processed_ticker = get_valid_ticker(ticker_input)
        st.header(f"📈 Kết quả cho: {processed_ticker}")
        model_path = os.path.join(MODELS_DIR, f"{processed_ticker}_model.keras")
        try:
            if force_train_button or not os.path.exists(model_path):
                st.info(f"Không tìm thấy mô hình có sẵn hoặc bạn yêu cầu huấn luyện lại. Bắt đầu quá trình mới...")
                model, scaler, last_sequence, data_for_display = run_training(processed_ticker)
            else:
                st.info("Phát hiện mô hình đã được huấn luyện. Đang tải...")
                model, scaler, last_sequence, data_for_display = run_prediction(processed_ticker)

            if data_for_display is not None and not data_for_display.empty:
                st.subheader("Dữ liệu Gần nhất")
                st.dataframe(data_for_display.tail(5))

            if all(v is not None for v in [model, scaler, last_sequence, data_for_display]):
                with st.spinner("Đang tính toán dự đoán tương lai..."):
                    future_prices = predict_future(model, scaler, last_sequence, days_to_predict)
                display_results(data_for_display, processed_ticker, future_prices)
        
        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"Đã xảy ra một lỗi không mong muốn: {e}")
    else:
        st.info("Nhập mã cổ phiếu và nhấn 'Dự đoán' để xem kết quả.")
