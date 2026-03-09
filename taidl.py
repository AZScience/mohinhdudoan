
import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import sys

# --- CÁC HÀM TÍNH TOÁN CHỈ BÁO KỸ THUẬT ---

def calculate_rsi(data, length=14):
    """Tính toán Chỉ số Sức mạnh Tương đối (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Tính toán Trung bình động Hội tụ Phân kỳ (MACD)."""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, length=20, std=2):
    """Tính toán Dải Bollinger."""
    middle_band = data['Close'].rolling(window=length).mean()
    std_dev = data['Close'].rolling(window=length).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    return lower_band, middle_band, upper_band

# --- HÀM TẢI DỮ LIỆU CHÍNH ---

def download_stock_data(ticker, start_date="2019-01-01"):
    """
    Tải dữ liệu lịch sử cổ phiếu, tính toán các chỉ báo kỹ thuật và lưu vào file CSV.
    """
    data_dir = "data"
    try:
        os.makedirs(data_dir, exist_ok=True)
        print(f"Thư mục '{data_dir}' đã sẵn sàng.")
    except OSError as e:
        print(f"Lỗi khi tạo thư mục '{data_dir}': {e}")
        return

    file_path = os.path.join(data_dir, f"{ticker}.csv")
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\nĐang chuẩn bị tải dữ liệu cho mã: {ticker}")
    print(f"Thời gian: từ {start_date} đến {end_date}")
    print(f"Lưu tại: {file_path}")

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=True,
            auto_adjust=False
        )

        if data.empty:
            print(f"Lỗi: Không tìm thấy dữ liệu cho mã {ticker}. Mã có thể không hợp lệ.")
            return

        print("\nĐang tính toán các chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands)...")
        
        # Tính toán và thêm các chỉ báo vào DataFrame
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = calculate_macd(data)
        data['Bollinger_Lower'], data['Bollinger_Middle'], data['Bollinger_Upper'] = calculate_bollinger_bands(data)

        # Reset index để 'Date' trở thành một cột
        data = data.reset_index()

        if "Adj Close" not in data.columns and "Close" in data.columns:
            data["Adj Close"] = data["Close"]

        # Xác định các cột sẽ được giữ lại
        cols_to_keep = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal',
            'Bollinger_Lower', 'Bollinger_Middle', 'Bollinger_Upper'
        ]
        
        # Chỉ giữ lại các cột thực sự tồn tại trong DataFrame
        existing_cols = [c for c in cols_to_keep if c in data.columns]
        data = data[existing_cols]
        
        # Loại bỏ các hàng có giá trị NaN (xuất hiện ở đầu do các phép tính chỉ báo)
        data.dropna(inplace=True)

        data.to_csv(file_path, index=False, encoding="utf-8-sig")

        print(f"\n>> THÀNH CÔNG! Đã lưu {len(data)} dòng dữ liệu vào '{file_path}'")
        print("5 dòng dữ liệu cuối (đã bao gồm các chỉ báo kỹ thuật):")
        print(data.tail().to_string())

    except Exception as e:
        print(f"\n>> ĐÃ XẢY RA LỖI khi tải {ticker}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tickers_to_download = [ticker.upper() for ticker in sys.argv[1:]]
        print(f"Sẽ tải dữ liệu cho các mã: {', '.join(tickers_to_download)}")
        for ticker in tickers_to_download:
            download_stock_data(ticker)
    else:
        print("--- CÔNG CỤ TẢI DỮ LIỆU CHỨNG KHOÁN (NÂNG CAO) ---")
        print("Sử dụng: python taidl.py <MÃ_1> <MÃ_2> ...")
        print("Ví dụ:   python taidl.py ORCL AAPL MSFT")
        print("\nKhông có mã nào được cung cấp. Kết thúc chương trình.")
