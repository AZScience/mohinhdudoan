
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
from textblob import TextBlob
import traceback

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
      {
        "description": "VN-INDEX",
        "proName": "VCBF:^VNINDEX"
      },
      {
        "description": "FPT",
        "proName": "HOSE:FPT"
      },
      {
        "description": "Vietcombank",
        "proName": "HOSE:VCB"
      },
      {
        "description": "Vingroup",
        "proName": "HOSE:VIC"
      },
      {
        "description": "S&P 500",
        "proName": "OANDA:SPX500USD"
      },
      {
        "description": "Nasdaq 100",
        "proName": "OANDA:NAS100USD"
      },
      {
        "description": "Bitcoin",
        "proName": "BITSTAMP:BTCUSD"
      },
      {
        "description": "Ethereum",
        "proName": "BITSTAMP:ETHUSD"
      }
    ],
    "showSymbolLogo": true,
    "colorTheme": "dark",
    "isTransparent": false,
    "displayMode": "adaptive",
    "locale": "vi_VN"
  }
  </script>
</div>
<!-- TradingView Widget END -->
'''

market_data_html = '''
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-data.js" async>
  {
  "width": "100%",
  "height": "100%",
  "symbolsGroups": [
    {
      "name": "Sàn HOSE",
      "originalName": "Indices",
      "symbols": [
        {
          "name": "HOSE:FPT",
          "displayName": "FPT"
        },
        {
          "name": "HOSE:VCB",
          "displayName": "VCB"
        },
        {
          "name": "HOSE:VIC",
          "displayName": "VIC"
        },
        {
          "name": "HOSE:HPG",
          "displayName": "HPG"
        },
        {
          "name": "HOSE:MSN",
          "displayName": "MSN"
        },
        {
          "name": "HOSE:MWG",
          "displayName": "MWG"
        }
      ]
    }
  ],
  "showSymbolLogo": true,
  "colorTheme": "dark",
  "isTransparent": false,
  "locale": "vi_VN"
}
  </script>
</div>
<!-- TradingView Widget END -->
'''

# --- Helper Functions ---
def ensure_dir(directory_path):
    os.makedirs(directory_path, exist_ok=True)

# --- Sentiment Analysis ---
@st.cache_data(ttl=3600)
def get_and_analyze_sentiment(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    if not news:
        return None, [], "No news found"

    polarity = 0
    titles = []
    processed_count = 0
    for item in news:
        title = item.get('title') # FIX: Use .get() for safety
        if title: # Only process if a title actually exists
            titles.append(title)
            blob = TextBlob(title)
            polarity += blob.sentiment.polarity
            processed_count += 1

    if processed_count == 0:
        return None, [], "No news titles found in the data"

    avg_polarity = polarity / processed_count
    return avg_polarity, titles, None

def display_sentiment_analysis(ticker):
    st.subheader("📰 Phân tích tâm lý từ tin tức gần đây")
    with st.spinner(f"Đang tìm kiếm và phân tích tin tức cho {ticker}..."):
        try:
            sentiment_score, news_titles, error_message = get_and_analyze_sentiment(ticker)
            
            if error_message:
                st.info(f"Không tìm thấy tin tức gần đây cho mã này (nguồn: yfinance). Điều này thường xảy ra với các mã cổ phiếu Việt Nam.")
                st.divider()
                return

            col1, col2 = st.columns([1, 2])

            with col1:
                if sentiment_score > 0.1:
                    st.metric("Tâm lý chung", "TÍCH CỰC", f"{sentiment_score:.2f}")
                    st.markdown("<div style='color: green; font-weight: bold;'>Nhận định: Các tin tức gần đây có xu hướng tích cực, có thể ảnh hưởng tốt đến giá.</div>", unsafe_allow_html=True)
                elif sentiment_score < -0.1:
                    st.metric("Tâm lý chung", "TIÊU CỰC", f"{sentiment_score:.2f}")
                    st.markdown("<div style='color: red; font-weight: bold;'>Nhận định: Các tin tức gần đây có xu hướng tiêu cực, có thể gây áp lực lên giá.</div>", unsafe_allow_html=True)
                else:
                    st.metric("Tâm lý chung", "TRUNG LẬP", f"{sentiment_score:.2f}")
                    st.markdown("<div style='color: gray; font-weight: bold;'>Nhận định: Các tin tức gần đây không có tác động cảm xúc rõ rệt.</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Các tiêu đề tin tức chính:**")
                for i, title in enumerate(news_titles[:5]):
                    st.markdown(f"- {title}")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tải và phân tích tin tức cho {ticker}:", icon="🚨")
            st.exception(e)

    st.divider()

# --- Technical Indicator Calculations ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
@st.cache_data(ttl=3600)
def get_or_download_data(ticker):
    ensure_dir(DATA_DIR)
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    try:
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            raise ValueError(f"Không có dữ liệu cho mã '{ticker}'. Mã có thể bị hủy niêm yết hoặc sai.")
        df.reset_index().to_csv(file_path, index=False)
    except Exception as e:
        if not os.path.exists(file_path):
            raise ValueError(f"Không có dữ liệu cho mã '{ticker}'. Mã có thể bị hủy niêm yết hoặc sai.")
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
    with st.spinner("Tải và xử lý dữ liệu..."):
        data = get_or_download_data(ticker)
        data_processed = add_technical_indicators(data)
    if data_processed.empty or len(data_processed) < LOOK_BACK:
        st.error(f"Không đủ dữ liệu cho '{ticker}' sau khi xử lý.")
        return None, None, None, None
    data_for_training = data_processed[FEATURES].astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_training)
    with st.spinner("Tạo chuỗi và huấn luyện mô hình..."):
        X, y = create_sequences(scaled_data, LOOK_BACK)
        if len(X) == 0:
            st.error("Không thể tạo chuỗi huấn luyện. Dữ liệu quá ít.")
            return None, None, None, None
        model = build_model(input_shape=(LOOK_BACK, len(FEATURES)))
        model.fit(X, y, batch_size=32, epochs=50, verbose=0)
    with st.spinner("Lưu mô hình và scaler..."):
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
        new_row_scaled[1:4] = predicted_price_scaled
        new_row_scaled = new_row_scaled.reshape(1, num_features)
        current_input = np.concatenate([current_input[1:], new_row_scaled], axis=0)
    return future_predictions

def run_prediction(ticker):
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(SCALERS_DIR, f"{ticker}_scaler.pkl")
    with st.spinner(f"Tải mô hình, scaler và dữ liệu gần đây cho {ticker}..."):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        data = get_or_download_data(ticker)
        if data.empty or len(data) < LOOK_BACK:
            raise ValueError("Không đủ dữ liệu gần đây để dự đoán.")
        data_processed = add_technical_indicators(data)
        data_filtered = data_processed[FEATURES].astype(float)
        scaled_data = scaler.transform(data_filtered)
        last_sequence = scaled_data[-LOOK_BACK:]
    return model, scaler, last_sequence, data_processed

# --- UI and Main Application ---
def display_realtime_data(df):
    st.subheader("📊 Dữ liệu thị trường mới nhất")
    if len(df) < 2:
        st.warning("Không đủ dữ liệu để hiển thị thông tin real-time.")
        return
    
    last_row = df.iloc[-1]
    previous_row = df.iloc[-2]
    
    last_price = last_row['Close']
    price_change = last_row['Close'] - previous_row['Close']
    price_delta_percent = (price_change / previous_row['Close']) * 100
    volume = last_row['Volume']
    day_high = last_row['High']
    day_low = last_row['Low']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"Giá đóng cửa ({last_row.name.strftime('%d/%m/%Y')})",
            value=f"${last_price:,.2f}",
            delta=f"{price_change:,.2f} ({price_delta_percent:.2f}%)"
        )
    with col2:
        st.metric(label="Khối lượng giao dịch", value=f"{int(volume):,}")
    with col3:
        st.metric(label="Cao / Thấp trong ngày", value=f"${day_high:,.2f} / ${day_low:,.2f}")
    st.divider()

def display_results(df, ticker, days_to_predict, future_prices):
    if not isinstance(future_prices, list) or not future_prices:
        st.error("Lỗi: Dữ liệu dự đoán không hợp lệ hoặc rỗng.")
        return
    st.success(f"Dự đoán thành công! Giá đóng cửa ngày mai: **${future_prices[0]:.2f}**")

    tab1, tab2 = st.tabs(['📈 Biểu đồ Dự đoán', '📋 Bảng Dữ liệu Dự đoán Tương lai'])

    with tab1:
        try:
            historical_df = df.tail(90)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(historical_df.index, historical_df['Close'], color='blue', label='Giá thực tế')
            last_actual_date = historical_df.index[-1]
            last_actual_price = historical_df['Close'].iloc[-1]
            prediction_dates = [last_actual_date] + [last_actual_date + timedelta(days=i) for i in range(1, len(future_prices) + 1)]
            prediction_values = [last_actual_price] + [float(p) for p in future_prices]
            prediction_series = pd.Series(data=prediction_values, index=pd.to_datetime(prediction_dates))
            ax.plot(prediction_series.index, prediction_series.values, color='red', linestyle='--', label='Giá dự đoán')
            ax.set_title(f'Giá thực tế & Dự đoán cho {ticker}', fontsize=16)
            ax.set_xlabel('Ngày', fontsize=12)
            ax.set_ylabel('Giá Đóng cửa (Tiền)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi vẽ biểu đồ: {e}")

    with tab2:
        try:
            last_actual_price = df['Close'].iloc[-1]
            comparison_prices = [last_actual_price] + future_prices[:-1]
            future_dates_table = [df.index[-1] + timedelta(days=i) for i in range(1, len(future_prices) + 1)]
            
            forecast_df = pd.DataFrame({
                'Ngày': future_dates_table, 
                'Giá đóng cửa dự đoán': future_prices,
                'Giá hôm trước': comparison_prices
            })

            def get_status_with_icon(row):
                if row['Giá đóng cửa dự đoán'] > row['Giá hôm trước']:
                    return '▲ TĂNG'
                else:
                    return '▼ GIẢM'
            
            forecast_df['Trạng thái'] = forecast_df.apply(get_status_with_icon, axis=1)

            display_df = forecast_df[['Ngày', 'Giá đóng cửa dự đoán', 'Trạng thái']].copy()
            display_df['Ngày'] = pd.to_datetime(display_df['Ngày']).dt.strftime('%d/%m/%Y')
            
            def color_status(val):
                color = 'green' if 'TĂNG' in val else 'red'
                return f'color: {color}; font-weight: bold;'

            styled_df = display_df.set_index('Ngày').style.applymap(
                color_status, subset=['Trạng thái']
            ).format({
                "Giá đóng cửa dự đoán": "${:,.2f}"
            })
            
            st.dataframe(styled_df, use_container_width=True, height=300)

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tạo bảng dữ liệu: {e}")
            import traceback
            st.code(traceback.format_exc())

# --- Streamlit App Main ---
st.set_page_config(layout="wide", page_title="MÔ HÌNH HỌC SÂU DỰ ĐOÁN GIÁ CỔ PHIẾU")

# Inject custom CSS for fixed header/footer and no scroll
st.markdown("""<style>
    /* Main page adjustments */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem; /* Make space for footer */
    }
    /* Header margins */
    h1[style*='text-align: center'] { margin-bottom: 0 !important; }
    p[style*='text-align: right'] { margin-top: 0 !important; margin-bottom: 0 !important; }
    hr { margin-top: 0.5rem !important; margin-bottom: 1rem !important; }
    /* Fixed Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #888;
        text-align: center;
        padding: 5px 0;
        font-size: 0.8rem;
        z-index: 999;
    }
    .footer p { margin: 0; }
</style>""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #0072B2;'>💹 MÔ HÌNH HỌC SÂU DỰ ĐOÁN GIÁ CỔ PHIẾU </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right; font-style: bold;'>Học viên: LÊ HỒNG PHONG - Mã số: 8480201</p>", unsafe_allow_html=True)
st.markdown("---")

components.html(market_overview_html, height=72)

col1, col2 = st.columns([1, 2])
with col1:
    st.header("⚙️ Nhập mã cổ phiếu")
    ticker_input = st.text_input("📈Nhập mã chứng khoán cần dự đoán (VD: FPT.VN, NVDA, AAPL...):", "AAPL").upper()
    days_to_predict = st.number_input(
        "📅 Số ngày dự đoán (1-30):",
        min_value=1,
        max_value=30,
        value=7,
        help="Mô hình đáng tin cậy nhất cho dự đoán ngắn hạn (1-7 ngày)."
    )
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        run_button = st.button("🚀 Dự đoán", type="primary", use_container_width=True)
    with col1_2:
        force_train_button = st.button("🔄 Huấn luyện", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True) # Add some space
    components.html(market_data_html, height=450)

with col2:
    st.header(f"📈 Kết quả dự đoán cho cổ phiếu: {ticker_input}")
    action = None
    if run_button:
        action = 'predict_or_train'
    elif force_train_button:
        action = 'force_train'
    
    if action:
        processed_ticker = ticker_input.split(',')[0].strip()
        if not processed_ticker:
            st.error("Mã chứng khoán không được để trống.")
        else:
            # Display sentiment analysis right after getting the ticker
            display_sentiment_analysis(processed_ticker)
            model_path = os.path.join(MODELS_DIR, f"{processed_ticker}_model.keras")
            try:
                if action == 'force_train' or not os.path.exists(model_path):
                    st.warning(f"Bắt đầu quá trình huấn luyện mới cho {processed_ticker}.")
                    model, scaler, last_sequence, data_for_display = run_training(processed_ticker)
                else:
                    model, scaler, last_sequence, data_for_display = run_prediction(processed_ticker)
                
                if data_for_display is not None:
                    display_realtime_data(data_for_display)

                if all(v is not None for v in [model, scaler, last_sequence, data_for_display]):
                    with st.spinner("Đang dự đoán giá tương lai..."):
                        future_prices = predict_future(model, scaler, last_sequence, days_to_predict)
                    display_results(data_for_display, processed_ticker, days_to_predict, future_prices)
            
            except ValueError as e:
                st.error(e)
                st.warning("Gợi ý: Mã CK Việt Nam thường có đuôi '.VN' (VCB.VN). Chỉ số có ký hiệu đặc biệt ('^VNINDEX').")
            except Exception as e:
                st.error(f"Đã xảy ra một lỗi nghiêm trọng: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.write("Kết quả sẽ hiện thị sau khi nhấn nút để bắt đầu.")

# Footer call
st.markdown("<div class=\"footer\"><p><h5>© 2026 Lê Hồng Phong. Bảo lưu mọi quyền. Email: ngviphuc@gmail.com. SĐT: 0937 382 399</h5></p></div>", unsafe_allow_html=True)
