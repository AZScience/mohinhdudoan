
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# --- Basic Configuration ---
st.set_page_config(layout="wide")
st.title("Kiểm tra hiển thị biểu đồ - Lối thoát")

st.info("Ứng dụng này chỉ tải và hiển thị biểu đồ giá đóng cửa của mã AAPL để kiểm tra chức năng cốt lõi.")

TICKER = "AAPL"

# --- Data Loading ---
try:
    st.write(f"Đang tải dữ liệu cho {TICKER}...")
    data = yf.download(TICKER, period="1y")
    
    if data.empty:
        st.error(f"Không thể tải dữ liệu cho {TICKER}. Dữ liệu trống.")
    else:
        st.success(f"Đã tải thành công {len(data)} dòng dữ liệu.")
        st.dataframe(data.tail())

        # --- Chart Display ---
        st.write("Đang cố gắng vẽ biểu đồ...")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Giá đóng cửa thực tế',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title=f"Biểu đồ giá đóng cửa cho {TICKER}",
            xaxis_title="Ngày",
            yaxis_title="Giá (USD)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success("Nếu bạn thấy biểu đồ này, chức năng cốt lõi đã hoạt động!")

except Exception as e:
    st.error(f"Đã xảy ra một lỗi nghiêm trọng:")
    st.exception(e)
