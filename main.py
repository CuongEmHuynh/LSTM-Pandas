import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.LSTM import train_model
from sklearn.preprocessing import MinMaxScaler
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

    # Chuẩn hóa y (cột cnt) riêng biệt
    scaler_y = MinMaxScaler()
    y = y.reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y)

    # variable for LSTM model
    input_size = X.shape[2]
    hidden_size = 64
    num_layers = 1
    epochs = 10
    batch_size = 32
    lr = 0.0005
    
    model = train_model(X, y, input_size, hidden_size, num_layers, epochs, batch_size, lr)


