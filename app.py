
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta

# --- Basic Configuration ---
st.set_page_config(layout="wide")
st.title("Ứng dụng Phân tích Cổ phiếu với Chỉ báo Kỹ thuật")

# --- Sidebar for User Input ---
st.sidebar.header("Tùy chọn")
TICKER = st.sidebar.text_input("Nhập mã cổ phiếu:", "AAPL").upper()

st.info(f"Phân tích mã cổ phiếu: {TICKER}")

# --- Data Loading ---
try:
    st.write(f"Đang tải dữ liệu cho {TICKER}...")
    data = yf.download(TICKER, period="1y", auto_adjust=True)
    
    if data.empty:
        st.error(f"Không thể tải dữ liệu cho {TICKER}. Vui lòng kiểm tra lại mã cổ phiếu.")
    else:
        st.success(f"Đã tải thành công {len(data)} dòng dữ liệu.")

        # --- Calculate Technical Indicators ---
        st.write("Đang tính toán các chỉ báo kỹ thuật (RSI, MACD)...")
        # Calculate RSI
        data.ta.rsi(append=True)
        # Calculate MACD
        data.ta.macd(append=True)
        
        st.success("Tính toán chỉ báo thành công.")
        st.dataframe(data.tail())

        # --- Chart Display ---
        st.write("Đang vẽ biểu đồ...")

        # Create Figure with subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=(f'Giá {TICKER}', 'Chỉ số sức mạnh tương đối (RSI)', 'MACD'),
                            row_heights=[0.6, 0.2, 0.2])

        # Plot Close Price
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Giá đóng cửa',
            line=dict(color='blue')
        ), row=1, col=1)

        # Plot RSI
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI_14'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        # Add RSI Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Plot MACD
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD_12_26_9'],
            mode='lines',
            name='MACD',
            line=dict(color='orange')
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACDs_12_26_9'],
            mode='lines',
            name='Signal Line',
            line=dict(color='cyan')
        ), row=3, col=1)
        fig.add_bar(
            x=data.index,
            y=data['MACDh_12_26_9'],
            name='Histogram',
            marker_color='grey',
        , row=3, col=1)

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success("Hiển thị biểu đồ thành công!")

except Exception as e:
    st.error(f"Đã xảy ra một lỗi nghiêm trọng:")
    st.exception(e)
