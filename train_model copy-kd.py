
import time
import yfinance as yf
import pandas as pd
import pandas_ta as ta
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


def load_and_preprocess_data(ticker, period, interval):
    """Tải và tiền xử lý dữ liệu từ yfinance."""
    df = yf.download(tickers=ticker, period=period, interval=interval)
    df.drop("Adj Close", axis=1, inplace=True)

    # Thêm các chỉ báo kỹ thuật
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.dropna(inplace=True)

    return df


def create_sequences(data, sequence_length):
    """Tạo sequences cho mô hình time series."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i : (i + sequence_length)]
        y = data[i + sequence_length][0]  # Dự đoán giá Close
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
    # Input Layer
    input_layer = Input(shape=(sequence_length, n_features))

    # --- Nhánh 1: CNN ---
    cnn_branch = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(
        input_layer
    )
    cnn_branch = AveragePooling1D(pool_size=2)(cnn_branch)
    cnn_branch = Flatten()(cnn_branch)

    # --- Nhánh 2: BiLSTM với Attention ---
    bilstm_branch = Bidirectional(
        LSTM(lstm_units, return_sequences=True, activation="relu")
    )(input_layer)
    bilstm_branch = Dropout(dropout_rate)(bilstm_branch)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(query=bilstm_branch, value=bilstm_branch, key=bilstm_branch)
    attention_output = Flatten()(attention_output)

    # --- Kết hợp hai nhánh ---
    combined = Concatenate()([cnn_branch, attention_output])

    # --- Các lớp Fully Connected ---
    dense = Dense(128, activation="relu")(combined)
    dense = Dropout(dropout_rate)(dense)
    output = Dense(1, activation="linear")(dense)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def main():
    """Hàm chính để chạy toàn bộ quy trình."""
    # --- 1. Tải và xử lý dữ liệu ---
    ticker = "ORCL"
    df = load_and_preprocess_data(ticker, period="5y", interval="1d")
    features = ["Close", "Open", "High", "Low", "Volume", "RSI_14", "MACD_12_26_9"]
    df_features = df[features]

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

    sequence_length = 21  # Sử dụng 21 ngày để dự đoán ngày tiếp theo
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_val, y_val = create_sequences(val_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)

    # --- 3. Tìm kiếm tham số tối ưu (Grid Search) ---
    param_grid = {
        "batch_size": [32, 64],
        "epochs": [50, 100],
        "model__lstm_units": [100, 150],
        "model__dropout_rate": [0.2, 0.3],
        "model__num_heads": [4, 8],
    }

    # Sử dụng KerasRegressor để bọc mô hình
    kr = KerasRegressor(
        build_fn=create_parallel_model,
        sequence_length=sequence_length,
        n_features=len(features),
        verbose=0,
    )

    # TimeSeriesSplit cho cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=kr,
        param_grid=param_grid,
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        scoring="neg_mean_squared_error",
    )

    print("Bắt đầu Grid Search...")
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

    # Tạo một mảng rỗng với cùng số feature như scaler đã fit
    dummy_predictions = np.zeros((len(predictions_scaled), len(features)))
    dummy_predictions[:, 0] = predictions_scaled.ravel()

    # Áp dụng inverse_transform
    predictions = scaler.inverse_transform(dummy_predictions)[:, 0]

    # Chuẩn bị y_test thực tế để so sánh
    dummy_y_test = np.zeros((len(y_test), len(features)))
    dummy_y_test[:, 0] = y_test.ravel()
    actual_prices = scaler.inverse_transform(dummy_y_test)[:, 0]

    # Tính toán các chỉ số lỗi
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
    plt.title(f"Dự đoán giá cổ phiếu {ticker}")
    plt.xlabel("Ngày")
    plt.ylabel("Giá Close")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

