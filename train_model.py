
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    AveragePooling1D,
    Bidirectional,
    LSTM,
    Dropout,
    Flatten,
    Dense,
    Concatenate,
    MultiHeadAttention,
)
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import os

def load_data_from_csv(ticker):
    """Tải dữ liệu đã được tiền xử lý từ file CSV trong thư mục 'data'."""
    file_path = os.path.join("data", f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {file_path}. "
            f"Vui lòng chạy lệnh 'python taidl.py {ticker}' trước."
        )
    # SỬA LỖI: Chỉ định header ở hàng 0 và bỏ qua hàng 1 có thể chứa dữ liệu rác
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date', header=0, skiprows=[1])
    # Chuyển đổi tất cả các cột số sang kiểu float để đảm bảo
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    print(f"Đã tải và dọn dẹp thành công dữ liệu từ {file_path}")
    return df

def create_sequences(data, sequence_length):
    """Tạo sequences cho mô hình time series."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i : (i + sequence_length)]
        y = data[i + sequence_length][0]  # Dự đoán giá Close (cột đầu tiên)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_parallel_model(
    sequence_length,
    n_features,
    lstm_units=100,
    dropout_rate=0.2,
    num_heads=8,
    key_dim=128,
):
    """Tạo mô hình song song CNN và BiLSTM với MultiHeadAttention."""
    input_layer = Input(shape=(sequence_length, n_features))
    cnn_branch = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(
        input_layer
    )
    cnn_branch = AveragePooling1D(pool_size=2)(cnn_branch)
    cnn_branch = Flatten()(cnn_branch)
    bilstm_branch = Bidirectional(
        LSTM(lstm_units, return_sequences=True, activation="relu")
    )(input_layer)
    bilstm_branch = Dropout(dropout_rate)(bilstm_branch)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(query=bilstm_branch, value=bilstm_branch, key=bilstm_branch)
    attention_output = Flatten()(attention_output)
    combined = Concatenate()([cnn_branch, attention_output])
    dense = Dense(128, activation="relu")(combined)
    dense = Dropout(dropout_rate)(dense)
    output = Dense(1, activation="linear")(dense)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def main():
    """Hàm chính để chạy toàn bộ quy trình."""
    # --- 1. Tải dữ liệu từ CSV ---
    ticker = "ORCL"
    df = load_data_from_csv(ticker)

    # --- CẬP NHẬT DANH SÁCH TÍNH NĂNG ---
    features = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal',
        'Bollinger_Lower', 'Bollinger_Middle', 'Bollinger_Upper'
    ]
    # Đảm bảo chỉ chọn các cột tồn tại trong DataFrame
    features = [f for f in features if f in df.columns]
    df_features = df[features]

    print(f"\nSử dụng {len(features)} tính năng để huấn luyện mô hình:")
    print(features)

    # --- 2. Phân chia và chuẩn hóa dữ liệu ---
    train_size = int(len(df_features) * 0.8)
    val_size = int(len(df_features) * 0.1)
    train_data = df_features[:train_size]
    val_data = df_features[train_size : train_size + val_size]
    test_data = df_features[train_size + val_size :]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    sequence_length = 21
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_val, y_val = create_sequences(val_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)
    n_features = X_train.shape[2]

    # --- 3. Tìm kiếm tham số tối ưu (Grid Search) ---
    param_grid = {
        "batch_size": [32, 64],
        "epochs": [50],
        "model__lstm_units": [100],
        "model__dropout_rate": [0.2],
        "model__num_heads": [8],
    }

    kr = KerasRegressor(
        build_fn=create_parallel_model,
        sequence_length=sequence_length,
        n_features=n_features,
        verbose=0,
    )

    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=kr,
        param_grid=param_grid,
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        scoring="neg_mean_squared_error",
    )

    print("\nBắt đầu Grid Search (với bộ dữ liệu và features đã được nâng cấp)...")
    start_time = time.time()
    grid_search.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    )
    end_time = time.time()
    print(f"Grid Search hoàn tất trong {end_time - start_time:.2f} giây.")
    print("Tham số tốt nhất:", grid_search.best_params_)

    best_model = grid_search.best_estimator_.model

    # --- 4. Đánh giá mô hình tốt nhất ---
    predictions_scaled = best_model.predict(X_test)
    dummy_predictions = np.zeros((len(predictions_scaled), n_features))
    dummy_predictions[:, 0] = predictions_scaled.ravel()
    predictions = scaler.inverse_transform(dummy_predictions)[:, 0]

    dummy_y_test = np.zeros((len(y_test), n_features))
    dummy_y_test[:, 0] = y_test.ravel()
    actual_prices = scaler.inverse_transform(dummy_y_test)[:, 0]

    rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
    mae = mean_absolute_error(actual_prices, predictions)
    mape = mean_absolute_percentage_error(actual_prices, predictions)
    r2 = r2_score(actual_prices, predictions)

    print("\n--- Kết quả đánh giá trên tập test ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R^2: {r2:.4f}")

    # --- 5. Vẽ biểu đồ kết quả ---
    plt.figure(figsize=(15, 7))
    plt.plot(test_data.index[sequence_length:], actual_prices, color="blue", label="Giá Thực tế")
    plt.plot(test_data.index[sequence_length:], predictions, color="red", alpha=0.7, label="Giá Dự đoán")
    plt.title(f"Dự đoán giá cổ phiếu {ticker} (Model đã nâng cấp)")
    plt.xlabel("Ngày")
    plt.ylabel("Giá Close")
    plt.legend()
    plt.grid(True)
    chart_filename = f'{ticker}_prediction_chart.png'
    plt.savefig(chart_filename)
    print(f"\nĐã lưu biểu đồ dự đoán vào file: {chart_filename}")

if __name__ == "__main__":
    main()
