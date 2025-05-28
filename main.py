import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.LSTM import train_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType
import torch
#Load Data from CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['dteday'] = pd.to_datetime(data['dteday'])
    data = data.sort_values('dteday')
    print(data.head())
    return data

# Chuẩn hóa dữ liệu
def create_sequences(data, seq_length,cnt_index=4):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # Chuỗi đầu vào dài seq_length
        y.append(data[i + seq_length, cnt_index])  # Giá trị 'cnt' tại bước tiếp theo
    return np.array(X), np.array(y)

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_test_tensor).numpy()
    return y_pred


@pandas_udf(FloatType())
def predict_batch_udf(*cols: pd.Series) -> pd.Series:
    # Danh sách các cột
    features = ['temp', 'atemp', 'hum', 'windspeed','cnt', 'hr' ]
    seq_len =24

    # Tạo DataFrame từ các cột đầu vào
    pdf = pd.DataFrame({feat: col for feat, col in zip(features, cols)})
    
    # Load mô hình và scaler
    model, scaler_X, scaler_y = load_model_and_scaler("/content/lstm_model.pth", "/content/scaler_X.pkl", "/content/scaler_y.pkl")

    # Chuyển dữ liệu thành numpy
    data = pdf[features].astype('float32').values
    data_scaled = scaler_X.transform(data)

    # Tạo sequence
    X = []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i + seq_len])
    if not X:
        return pd.Series([None] * len(pdf))  # Không đủ dữ liệu

    X_tensor = torch.tensor(np.array(X))  # [batch, seq, features]
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()

    # Đảo chuẩn hóa
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
   

    # Pad để khớp độ dài gốc
    padding = [None] * seq_len
    return pd.Series(padding + y_pred.tolist())

if __name__ == '__main__':
    file_path = "./dataset/hour.csv"

    # Load data
    data = load_data(file_path)

    features = ['temp', 'atemp', 'hum', 'windspeed', 'cnt', 'hr']
    data_multivariate = data[features].values
    
    scaler_X = MinMaxScaler()
    data_multivariate_scaled = scaler_X.fit_transform(data_multivariate)
    
     # Tạo sequence từ dữ liệu đã chuẩn hóa
    X, y = create_sequences(data_multivariate_scaled, 24, 4)
    y = y.reshape(-1, 1)

    joblib.dump(scaler_X, "scaler_X.pkl")
    # variable for LSTM model
    input_size = X.shape[2]
    hidden_size = 64
    num_layers = 1
    epochs = 10
    batch_size = 32
    lr = 0.0005
    
    model = train_model(X, y, input_size, hidden_size, num_layers, epochs, batch_size, lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()
    # Đưa về giá trị gốc
    # Tạo mảng tạm để inverse transform
    feature_idx=4
    temp_shape = scaler_X.inverse_transform(np.zeros((len(y), len(features))))
    temp_shape[:, feature_idx] = y.flatten()  # Gán y vào cột 'cnt'
    y_true_unscaled = scaler_X.inverse_transform(temp_shape)[:, feature_idx]
    temp_shape[:, feature_idx] = y_pred.flatten()
    y_pred_unscaled = scaler_X.inverse_transform(temp_shape)[:, feature_idx]

    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_unscaled, label='Giá trị thực tế', color='blue')
    plt.plot(y_pred_unscaled, label='Giá trị dự đoán', color='red', linestyle='--')
    plt.xlabel('Thời gian')
    plt.ylabel('Số lượng (cnt)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    


