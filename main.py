
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Bidirectional, LSTM, Dropout, Concatenate, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from datetime import date, timedelta
import streamlit.components.v1 as components
from textblob import TextBlob
import traceback
from gnews import GNews
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

# --- Constants ---
DATA_DIR = "data"
MODELS_DIR = "models"
SCALERS_DIR = "scalers"
FEATURES = ['Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MACD', 'Sentiment']
LOOK_BACK = 21
CLOSE_INDEX = FEATURES.index('Close')
HIGH_INDEX = FEATURES.index('High')
LOW_INDEX = FEATURES.index('Low')
TARGET_INDICES = [CLOSE_INDEX, HIGH_INDEX, LOW_INDEX]

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

# --- Sentiment Analysis ---
@st.cache_data(ttl=3600)
def analyze_sentiment_over_time(ticker, days=30):
    search_term = ticker.split('.')[0]
    google_news = GNews(language='en', country='US')
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    google_news.start_date = (start_date.year, start_date.month, start_date.day)
    google_news.end_date = (end_date.year, end_date.month, end_date.day)
    
    try:
        news_items = google_news.get_news(f'{search_term} stock')
        if not news_items: return pd.DataFrame(), [], f"No news found for '{search_term}' in the last {days} days."
    except Exception:
        return pd.DataFrame(), [], "Failed to fetch news."

    df = pd.DataFrame(news_items)
    df['published date'] = pd.to_datetime(df['published date'], errors='coerce')
    df.dropna(subset=['published date'], inplace=True)
    df['sentiment'] = df['title'].apply(lambda title: TextBlob(title).sentiment.polarity)
    
    df['date'] = df['published date'].dt.date
    sentiment_trend = df.groupby('date')['sentiment'].mean()
    volume_trend = df.groupby('date').size()
    
    trend_df = pd.DataFrame({'sentiment': sentiment_trend, 'volume': volume_trend}).reindex(pd.date_range(start=start_date, end=end_date, freq='D')).fillna(0)
    
    latest_titles = df.sort_values(by='published date', ascending=False)['title'].tolist()
    return trend_df, latest_titles, None

def display_sentiment_chart(trend_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=trend_df.index, y=trend_df['sentiment'], name="Chỉ số Tâm lý", line=dict(color='#FF6F00', width=2)), secondary_y=False)
    fig.add_trace(go.Bar(x=trend_df.index, y=trend_df['volume'], name="Lượng tin tức", marker_color='#00B0F6', opacity=0.6), secondary_y=True)

    fig.update_layout(title_text="Xu Hướng Tâm Lý & Khối Lượng Tin Tức (30 Ngày)", template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=80, b=20))
    fig.update_yaxes(title_text="Chỉ số Tâm lý", secondary_y=False)
    fig.update_yaxes(title_text="Lượng tin tức", secondary_y=True)
    fig.update_xaxes(tickformat="%d/%m/%Y") # UI Improvement: Standardize date format
    st.plotly_chart(fig, use_container_width=True)

def display_sentiment_analysis(ticker):
    st.subheader("📰 Phân tích tâm lý từ Google News")
    with st.spinner(f"Đang tìm kiếm và phân tích tin tức cho {ticker}..."):
        try:
            sentiment_df, news_titles, error_message = analyze_sentiment_over_time(ticker)

            # MAJOR BUG FIX: Explicitly check if any news was found
            if error_message or sentiment_df['volume'].sum() == 0:
                st.warning(f"⚠️ Không tìm thấy tin tức nào cho '{ticker}' trong 30 ngày qua.", icon="📡")
                st.info("Tính năng có thể tạm gián đoạn hoặc mã cổ phiếu của bạn không có tin tức quốc tế gần đây.")
                return

            last_day_with_news_df = sentiment_df[sentiment_df['volume'] > 0]

            if last_day_with_news_df.empty:
                 st.warning(f"⚠️ Không tìm thấy tin tức nào cho '{ticker}' trong 30 ngày qua.", icon="📡")
                 return
            
            last_meaningful_day = last_day_with_news_df.index[-1]
            last_score = last_day_with_news_df['sentiment'].iloc[-1]
            
            if last_meaningful_day.date() == date.today():
                message_prefix = "Tâm lý gần đây"
            else:
                last_date_str = last_meaningful_day.strftime('%d/%m/%Y')
                message_prefix = f"Tâm lý gần nhất ({last_date_str})"
            
            if last_score > 0.15:
                st.success(f"**{message_prefix}: TÍCH CỰC ({last_score:.2f})**")
            elif last_score < -0.15:
                st.error(f"**{message_prefix}: TIÊU CỰC ({last_score:.2f})**")
            else:
                st.info(f"**{message_prefix}: TRUNG LẬP ({last_score:.2f})**")

            display_sentiment_chart(sentiment_df)

            with st.expander("🔍 Diễn giải & Phương pháp tính"):
                avg_sentiment = sentiment_df[sentiment_df['sentiment'] > 0]['sentiment'].mean() if not last_day_with_news_df.empty else 0
                total_articles = sentiment_df['volume'].sum()
                if not last_day_with_news_df.empty:
                    max_sentiment_day = sentiment_df['sentiment'].idxmax()
                    min_sentiment_day = sentiment_df['sentiment'].idxmin()
                else:
                    max_sentiment_day = None
                    min_sentiment_day = None

                st.markdown("##### Các chỉ số chính trong 30 ngày")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Điểm TB (những ngày có tin)", f"{avg_sentiment:.3f}")
                    if max_sentiment_day: st.metric("Ngày tích cực nhất", max_sentiment_day.strftime('%d/%m/%Y'))
                with col2:
                    st.metric("Tổng số tin tức", f"{int(total_articles)}")
                    if min_sentiment_day: st.metric("Ngày tiêu cực nhất", min_sentiment_day.strftime('%d/%m/%Y'))
                st.markdown("<hr>", unsafe_allow_html=True)
                
                st.markdown("##### Phương pháp tính điểm Tâm lý")
                st.markdown('''
                - **Nguồn dữ liệu**: Tiêu đề các bài báo tiếng Anh từ **Google News** có chứa tên mã cổ phiếu.
                - **Công cụ**: Thư viện xử lý ngôn ngữ tự nhiên `TextBlob` của Python được sử dụng để phân tích.
                - **Điểm số**: Mỗi tiêu đề được gán một "Điểm Phân cực" (Polarity Score) dao động từ **-1.0** (rất tiêu cực) đến **+1.0** (rất tích cực). Điểm 0 thể hiện sự trung lập.
                ''')

                st.markdown("##### Cách diễn giải biểu đồ")
                st.markdown('''
                - **Đường màu cam (Chỉ số Tâm lý)**: Là điểm tâm lý trung bình của tất cả các tin tức trong một ngày. Một xu hướng tăng cho thấy nhận thức của truyền thông đang tích cực hơn và ngược lại. 
                - **Cột màu xanh (Lượng tin tức)**: Thể hiện số lượng bài báo được tìm thấy mỗi ngày. Sự đột biến về số lượng, đặc biệt khi đi kèm với điểm tâm lý cao hoặc thấp, thường báo hiệu một sự kiện quan trọng.
                ''')

            with st.expander("Xem các tiêu đề tin tức mới nhất"):
                if not news_titles:
                    st.write("Không có tiêu đề nào.")
                else:
                    for title in news_titles[:7]: st.markdown(f"- {title}")
        except Exception as e:
            st.warning("⚠️ Đã xảy ra lỗi trong quá trình phân tích tâm lý.", icon="📡")
            st.code(f"Chi tiết lỗi: {e}")
    st.divider()

# --- Data Integration & TA ---
def find_pivot_levels(df, lookback_period=250, peak_prominence=0.05):
    data = df.tail(lookback_period)
    if data.empty or len(data) < 20: return [], []
    price_range = data['High'].max() - data['Low'].min()
    if price_range == 0: return [], []
    
    actual_prominence = price_range * peak_prominence
    resistance_indices, _ = find_peaks(data['High'], prominence=actual_prominence)
    support_indices, _ = find_peaks(-data['Low'], prominence=actual_prominence)
    
    resistance_levels = data['High'].iloc[resistance_indices]
    support_levels = data['Low'].iloc[support_indices]
    
    last_price = data['Close'].iloc[-1]
    
    resistances = sorted([r for r in resistance_levels if r > last_price])
    supports = sorted([s for s in support_levels if s < last_price], reverse=True)
    
    return supports[:1], resistances[:1]

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    middle_band = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band, middle_band, lower_band

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast_window=12, slow_window=26):
    ema_fast = data['Close'].ewm(span=fast_window, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_window, adjust=False).mean()
    return ema_fast - ema_slow

def add_technical_indicators(df):
    df_ta = df.copy()
    df_ta['RSI'] = calculate_rsi(df_ta)
    df_ta['MACD'] = calculate_macd(df_ta)
    df_ta['Bollinger_Upper'], df_ta['Bollinger_Middle'], df_ta['Bollinger_Lower'] = calculate_bollinger_bands(df_ta)
    return df_ta

# --- Data Loading and Processing ---
@st.cache_data(ttl=3600)
def get_or_download_data(ticker):
    ensure_dir(DATA_DIR)
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    try:
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty: raise ValueError(f"Không có dữ liệu cho mã '{ticker}'.")
        df.index.name = 'Date'
        df.to_csv(file_path)
    except Exception as e:
        if not os.path.exists(file_path):
            raise ValueError(f"Lỗi tải dữ liệu cho '{ticker}' và không có file dự phòng: {e}")
        st.warning(f"Không thể tải dữ liệu mới. Sử dụng dữ liệu đã lưu. Lỗi: {e}")

    df = pd.read_csv(file_path, parse_dates=['Date']).set_index('Date')
    df = df[~df.index.duplicated(keep='last')]
    df = df[df.index <= pd.to_datetime('today').normalize()]
    return df

# UPGRADE: Create sequences for a multi-output model
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, TARGET_INDICES])
    return np.array(X), np.array(y)

# --- Model Architecture ---
# UPGRADE: Build a multi-output model
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
    outputs = Dense(3)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Training and Prediction Logic ---
def run_training(ticker):
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(SCALERS_DIR, f"{ticker}_scaler.pkl")
    
    with st.spinner("Bước 1/4: Tải dữ liệu giá lịch sử..."):
        price_data = get_or_download_data(ticker)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
        price_data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    with st.spinner("Bước 2/4: Thêm các chỉ báo kỹ thuật..."):
        data_with_ta = add_technical_indicators(price_data)

    with st.spinner("Bước 3/4: Tải và tích hợp dữ liệu tâm lý..."):
        data_with_ta['Sentiment'] = 0.0
        
    data_processed = data_with_ta.dropna()
    if data_processed.empty or len(data_processed) < LOOK_BACK:
        st.error(f"Không đủ dữ liệu cho '{ticker}' để huấn luyện.")
        return None, None, None, None
    
    data_for_training = data_processed[FEATURES].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_training)
    
    with st.spinner("Bước 4/4: Huấn luyện mô hình Deep Learning đa đầu ra..."):
        X, y = create_sequences(scaled_data, LOOK_BACK)
        if len(X) == 0: st.error("Không thể tạo chuỗi huấn luyện. Dữ liệu quá ít."); return None, None, None, None
        
        model = build_model(input_shape=(LOOK_BACK, len(FEATURES)))
        model.fit(X, y, batch_size=32, epochs=50, verbose=0)
        
    with st.spinner("Hoàn tất! Đang lưu mô hình và scaler..."):
        ensure_dir(MODELS_DIR); ensure_dir(SCALERS_DIR)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
    st.success(f"Đã huấn luyện và lưu thành công mô hình mới cho {ticker}!")
    return model, scaler, scaled_data[-LOOK_BACK:], data_processed

def predict_future_range(model, scaler, initial_input, historical_df, days_to_predict):
    future_rows = []
    current_input = initial_input.copy()
    temp_df = historical_df.tail(100).copy()
    num_features = len(FEATURES)

    for _ in range(days_to_predict):
        X_pred = np.reshape(current_input, (1, LOOK_BACK, num_features))
        predicted_scaled_vector = model.predict(X_pred, verbose=0)[0]

        dummy_array = np.zeros((1, num_features))
        dummy_array[0, TARGET_INDICES] = predicted_scaled_vector
        inversed_prices = scaler.inverse_transform(dummy_array)[0]
        
        predicted_close = inversed_prices[CLOSE_INDEX]
        predicted_high = inversed_prices[HIGH_INDEX]
        predicted_low = inversed_prices[LOW_INDEX]
        
        predicted_high = max(predicted_high, predicted_close, temp_df['Close'].iloc[-1])
        predicted_low = min(predicted_low, predicted_close, temp_df['Close'].iloc[-1])

        last_date = temp_df.index[-1]
        next_date = last_date + timedelta(days=1)
        
        new_row_data = {
            'Open': temp_df['Close'].iloc[-1], 
            'High': predicted_high,
            'Low': predicted_low,
            'Close': predicted_close,
            'Volume': temp_df['Volume'].iloc[-1]
        }
        new_row_df = pd.DataFrame([new_row_data], index=[next_date])
        temp_df = pd.concat([temp_df, new_row_df])

        temp_df_with_ta = add_technical_indicators(temp_df)
        last_calculated_row = temp_df_with_ta.iloc[-1]
        future_rows.append(last_calculated_row)

        new_feature_row_dict = {
            'Close': last_calculated_row['Close'],
            'High': last_calculated_row['High'],
            'Low': last_calculated_row['Low'],
            'Open': last_calculated_row['Open'],
            'Volume': last_calculated_row['Volume'],
            'RSI': last_calculated_row['RSI'],
            'MACD': last_calculated_row['MACD'],
            'Sentiment': 0.0,
        }
        ordered_feature_row = [new_feature_row_dict.get(f, 0.0) for f in FEATURES]
        scaled_new_row = scaler.transform([ordered_feature_row])
        
        current_input = np.concatenate([current_input[1:], scaled_new_row], axis=0)

    return pd.DataFrame(future_rows)

def run_prediction(ticker):
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(SCALERS_DIR, f"{ticker}_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
         raise FileNotFoundError(f"Chưa có mô hình cho mã '{ticker}'.")

    with st.spinner(f"Tải mô hình, scaler và dữ liệu gần đây cho {ticker}..."): 
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error')
        scaler = joblib.load(scaler_path)
        
        data = get_or_download_data(ticker)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        data_processed = add_technical_indicators(data)
        data_processed['Sentiment'] = 0.0
        data_processed.dropna(inplace=True)
        
        if data_processed.empty or len(data_processed) < LOOK_BACK:
            raise ValueError("Không đủ dữ liệu gần đây để dự đoán.")

        data_for_scaling = data_processed[FEATURES].astype(float)
        scaled_data = scaler.transform(data_for_scaling)
        
    return model, scaler, scaled_data[-LOOK_BACK:], data_processed

# --- UI and Main Application ---
def display_realtime_data(df):
    st.subheader("📊 Dữ liệu thị trường mới nhất")
    if len(df) < 2: return

    last_row = df.iloc[-1]
    prev_row_price = df.iloc[-2]['Close']
    price_change = last_row['Close'] - prev_row_price
    price_delta_percent = (price_change / prev_row_price) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Giá đóng cửa ({last_row.name.strftime('%d/%m/%Y')})", f"{last_row['Close']:,.2f}", f"{price_change:,.2f} ({price_delta_percent:.2f}%)")
    col2.metric("Khối lượng giao dịch", f"{int(last_row['Volume']):,}")
    col3.metric("Cao / Thấp trong ngày", f"${last_row['High']:,.2f} / ${last_row['Low']:,.2f}")
    st.divider()

def display_results(df, future_df, ticker):
    if future_df.empty: st.error("Lỗi: Dữ liệu dự đoán không hợp lệ."); return
        
    first_prediction = future_df.iloc[0]
    st.success(f"Dự đoán thành công! Giá đóng cửa ngày mai: **${first_prediction['Close']:,.2f}**")
    
    st.subheader("🎯 Dự đoán Phiên giao dịch Ngày mai")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dự đoán Cao nhất", f"${first_prediction['High']:,.2f}")
    col2.metric("Dự đoán Đóng cửa", f"${first_prediction['Close']:,.2f}")
    col3.metric("Dự đoán Thấp nhất", f"${first_prediction['Low']:,.2f}")
    st.divider()

    tab1, tab2 = st.tabs(['📈 Biểu đồ Nến Dự đoán', '📋 Bảng Dữ liệu Dự đoán'])
    with tab1:
        chart_df_hist = df.tail(120)
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=chart_df_hist.index, y=chart_df_hist['Bollinger_Upper'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=chart_df_hist.index, y=chart_df_hist['Bollinger_Lower'], mode='lines', line=dict(width=0), name='Dải Bollinger Lịch sử', fill='tonexty', fillcolor='rgba(0, 176, 246, 0.1)'))
        fig.add_trace(go.Scatter(x=chart_df_hist.index, y=chart_df_hist['Bollinger_Middle'], mode='lines', line=dict(color='rgba(0, 176, 246, 0.6)', width=1.5, dash='dash'), name='Trung bình 20 ngày'))

        fig.add_trace(go.Candlestick(x=chart_df_hist.index, open=chart_df_hist['Open'], high=chart_df_hist['High'], low=chart_df_hist['Low'], close=chart_df_hist['Close'], name='Giá Lịch sử'))
        
        last_hist_date = chart_df_hist.index[-1]
        future_dates = pd.Index([last_hist_date]).append(future_df.index)

        connect_point_high = pd.Series(chart_df_hist['High'].iloc[-1], index=[last_hist_date])
        connect_high = pd.concat([connect_point_high, future_df['High']])

        connect_point_low = pd.Series(chart_df_hist['Low'].iloc[-1], index=[last_hist_date])
        connect_low = pd.concat([connect_point_low, future_df['Low']])

        fig.add_trace(go.Scatter(x=future_dates, y=connect_high, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_dates, y=connect_low, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 111, 0, 0.2)', name='Vùng giá dự đoán'))

        fig.add_trace(go.Candlestick(x=future_df.index, open=future_df['Open'], high=future_df['High'], low=future_df['Low'], close=future_df['Close'], name='Dự đoán (Dạng nến)', increasing=dict(line=dict(color='#00CF83')), decreasing=dict(line=dict(color='#FF5A5A'))))

        static_supports, static_resistances = find_pivot_levels(df.dropna(subset=['High', 'Low', 'Close']))
        for r_level in static_resistances:
            fig.add_hline(y=r_level, line_dash="dash", line_color="#EF5350", annotation_text=f"Kháng cự tĩnh {r_level:,.2f}", annotation_position="bottom right", annotation_font_color="#EF5350")
        for s_level in static_supports:
            fig.add_hline(y=s_level, line_dash="dash", line_color="#26A69A", annotation_text=f"Hỗ trợ tĩnh {s_level:,.2f}", annotation_position="top left", annotation_font_color="#26A69A")

        fig.update_layout(title=f'Phân Tích & Dự Đoán Dạng Nến cho {ticker}', yaxis_title='Giá (Tiền)', xaxis_title='Ngày', xaxis_rangeslider_visible=False, template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(tickformat="%m/%Y") # UI Improvement: Standardize date format
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        display_df = future_df[['Open', 'High', 'Low', 'Close']].copy()
        
        last_hist_close = df['Close'].iloc[-1]
        previous_closes = [last_hist_close] + future_df['Close'].iloc[:-1].tolist()
        
        display_df['Giá Hôm Trước'] = previous_closes
        display_df['Trạng thái'] = display_df.apply(lambda row: '▲ TĂNG' if row['Close'] > row['Giá Hôm Trước'] else '▼ GIẢM', axis=1)
        
        display_df.index = display_df.index.strftime('%d/%m/%Y')
        
        final_cols = ['Open', 'High', 'Low', 'Close', 'Trạng thái']
        display_df_styled = display_df[final_cols].style.format({'Open': '${:,.2f}', 'High': '${:,.2f}', 'Low': '${:,.2f}', 'Close': '${:,.2f}'}).applymap(lambda v: f'color: {"#00CF83" if "TĂNG" in v else "#FF5A5A"}; font-weight: bold;', subset=['Trạng thái'])
        
        st.dataframe(display_df_styled, use_container_width=True)

# --- Streamlit App Main ---
st.set_page_config(layout="wide", page_title="MÔ HÌNH HỌC SÂU DỰ ĐOÁN GIÁ CỔ PHIẾU")
st.markdown("<h1 style='text-align:center;color:#0072B2'>💹 MÔ HÌNH HỌC SÂU DỰ ĐOÁN GIÁ CỔ PHIẾU</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:right;font-style:bold'>Học viên: LÊ HỒNG PHONG - Mã số: 8480201</p><hr>", unsafe_allow_html=True)

components.html(market_overview_html, height=72)
col1, col2 = st.columns([1, 2])
action = None
with col1:
    st.header("⚙️ Nhập mã cổ phiếu")
    ticker_input = st.text_input("📈Nhập mã chứng khoán (VD: FPT.VN, NVDA, AAPL...):", "AAPL").upper()
    days_to_predict = st.number_input("📅 Số ngày dự đoán (1-30):", 1, 30, 7)
    c1, c2 = st.columns(2)
    if c1.button("🚀 Dự đoán", type="primary", use_container_width=True): action = 'predict_or_train'
    if c2.button("🔄 Huấn luyện lại", use_container_width=True): action = 'force_train'
    
    if action and (processed_ticker := ticker_input.split(',')[0].strip()):
        display_sentiment_analysis(processed_ticker)

with col2:
    st.header(f"📈 Kết quả dự đoán cho cổ phiếu: {ticker_input}")
    if action and (processed_ticker := ticker_input.split(',')[0].strip()):
        try:
            model_path = os.path.join(MODELS_DIR, f"{processed_ticker}_model.keras")
            needs_training = (action == 'force_train') or not os.path.exists(model_path)

            if needs_training:
                st.info(f"Bắt đầu quá trình huấn luyện mô hình đa đầu ra cho {processed_ticker}...", icon="✨")
                model, scaler, last_sequence, data_for_display = run_training(processed_ticker)
            else:
                model, scaler, last_sequence, data_for_display = run_prediction(processed_ticker)
            
            if data_for_display is not None: display_realtime_data(data_for_display)

            if all(v is not None for v in [model, scaler, last_sequence, data_for_display]):
                with st.spinner("Mô phỏng và dự đoán các phiên giao dịch tương lai..."):
                    future_df = predict_future_range(model, scaler, last_sequence, data_for_display, days_to_predict)
                
                display_results(data_for_display, future_df, processed_ticker)
        
        except (FileNotFoundError, ValueError) as e:
            st.error(f"Lỗi: {e}")
            st.info("Gợi ý: Nếu đây là lần đầu tiên, hãy thử nhấn 'Huấn luyện lại'.")
        except Exception as e:
            st.error(f"Đã xảy ra một lỗi nghiêm trọng: {e}")
            st.code(traceback.format_exc())
    else:
        st.info("Kết quả sẽ hiện thị sau khi nhấn nút để bắt đầu.")

# FINAL UI IMPROVEMENT: Add a dynamic, transparent, fixed, styled footer
current_date_str = date.today().strftime("%m/%Y")
footer_style = f'''
<style>
.footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #888; /* A more subtle text color */
    text-align: center;
    padding: 10px;
}}
</style>
<div class="footer">
    <p>© Lê Hồng Phong. Bảo lưu mọi quyền. Email: ngviphuc@gmail.com. Điện thoại: 0937 382 399</p>
</div>
'''
st.markdown(footer_style, unsafe_allow_html=True)
