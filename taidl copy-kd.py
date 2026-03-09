
import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import sys

def download_stock_data(ticker, start_date="2019-01-01"):
    """
    Tải dữ liệu lịch sử cổ phiếu và lưu vào file CSV trong thư mục 'data'.
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

        # Reset index để 'Date' trở thành một cột, định dạng chuẩn cho CSV
        data = data.reset_index()

        if "Adj Close" not in data.columns and "Close" in data.columns:
            data["Adj Close"] = data["Close"]

        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        existing_cols = [c for c in cols if c in data.columns]
        data = data[existing_cols]

        data.to_csv(file_path, index=False, encoding="utf-8-sig")

        print(f"\n>> THÀNH CÔNG! Đã lưu {len(data)} dòng dữ liệu vào '{file_path}'")
        print("5 dòng dữ liệu cuối:")
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
        print("--- CÔNG CỤ TẢI DỮ LIỆU CHỨNG KHOÁN ---")
        print("Sử dụng: python taidl.py <MÃ_1> <MÃ_2> ...")
        print("Ví dụ:   python taidl.py ORCL AAPL MSFT")
        print("\nKhông có mã nào được cung cấp. Kết thúc chương trình.")

