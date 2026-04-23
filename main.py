import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)

class TempDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]


def create_dataframe(df_raw):
    cols = {"DATETIME": "Datetime", "T": "Temperature", "P": "Air Pressure", "U": "Humidity", "Ff": "Wind speed",
            "N": "Cloudiness", "RRR": "Precipitation"}

    df = pd.DataFrame()
    for old, new in cols.items():
        df[new] = df_raw[old]

    df["Precipitation"] = df["Precipitation"].replace({"Осадков нет", "Следы осадков"}, 0.0).astype(np.float64)
    clouds = {'40%.': 0.4, '100%.': 1, 'Облаков нет.': 0, '20–30%.': 0.25, '10%  или менее, но не 0': 0.05,
              '90  или более, но не 100%': 0.95, '70 – 80%.': 0.75}
    for old, new in clouds.items():
        df["Cloudiness"] = df["Cloudiness"].replace(old, new)
    df["Cloudiness"] = df["Cloudiness"].astype(np.float64)

    precip = 0
    for i in range(len(df)):
        if np.isnan(df.loc[i, "Precipitation"]):
            df.loc[i, "Precipitation"] = precip
        else:
            precip = df.loc[i, "Precipitation"]

    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True)
    return df


def create_features(df):
    data = df.copy().loc[::-1]

    data["Day"] = pd.to_datetime(data["Datetime"]).dt.dayofyear
    data["Hour"] = pd.to_datetime(data["Datetime"]).dt.hour
    data["Hour_sin"] = np.sin(2 * np.pi * data["Hour"] / 24.0)
    data["Hour_cos"] = np.cos(2 * np.pi * data["Hour"] / 24.0)

    return data


def lag_features(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()

    lags = [1, 4]
    windows = [4]

    for lag in lags:
        train[f'temp_lag_{lag}'] = train['Temperature'].shift(lag)

    for window in windows:
        train[f'rolling_mean_{window}'] = train['Temperature'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        train[f'rolling_std_{window}'] = train['Temperature'].transform(
            lambda x: x.rolling(window, min_periods=1).std().shift(1)
        )

    train['temp_diff'] = train['Temperature'].diff(1)

    if "Temperature" not in test.columns:
        test["Temperature"] = 0

    combined = pd.concat([train, test]).sort_values(["Datetime"])

    for lag in lags:
        combined[f'temp_lag_{lag}'] = combined['Temperature'].shift(lag)

    for window in windows:
        combined[f'rolling_mean_{window}'] = combined['Temperature'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        combined[f'rolling_std_{window}'] = combined['Temperature'].transform(
            lambda x: x.rolling(window, min_periods=1).std().shift(1)
        )

    combined['temp_diff'] = combined['Temperature'].diff(1)

    train_mod = combined.iloc[:len(train)].dropna().reset_index(drop=True)
    test_mod = combined.iloc[len(train):].reset_index(drop=True)

    for col in test_mod.select_dtypes(include=[np.number]).columns:
        test_mod[col] = test_mod[col].fillna(0)

    if 'Temperature' not in test_df.columns:
        test_mod = test_mod.drop('Temperature', axis=1)

    return train_mod, test_mod


def prep_dataloaders(train_df, test_df, seq_len=4, train_ratio=0.8, batch_size=32):
    train_ft = create_features(train_df)
    test_ft = create_features(test_df)

    train_ft, test_ft = lag_features(train_ft, test_ft)

    feature_cols = [
        "Hour_sin", "Hour_cos",
        "temp_lag_1", "temp_lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "temp_diff", "Day", "Hour", "Air Pressure",
        "Wind speed", "Precipitation", "Cloudiness", "Humidity"
    ]

    feature_cols = [col for col in feature_cols if col in train_ft.columns]

    train_size = int(len(train_ft) * train_ratio)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_train_raw = scaler_x.fit_transform(train_ft[feature_cols].iloc[:train_size])
    y_train_raw = scaler_y.fit_transform(train_ft[["Temperature"]].iloc[:train_size])

    x_val_raw = scaler_x.transform(train_ft[feature_cols].iloc[train_size:])
    y_val_raw = scaler_y.transform(train_ft[["Temperature"]].iloc[train_size:])

    x_test = scaler_x.transform(test_ft[feature_cols])

    def create_sequences(x, y=None, seq_length=4):
        x_seq, y_seq = [], []
        for i in range(len(x) - seq_length):
            x_seq.append(x[i:i+seq_length])
            if y is not None:
                y_seq.append(y[i+seq_length])
        return np.array(x_seq), np.array(y_seq) if y is not None else None

    x_train_seq, y_train_seq = create_sequences(x_train_raw, y_train_raw, seq_len)
    x_val_seq, y_val_seq = create_sequences(x_val_raw, y_val_raw, seq_len)
    x_test_seq, _ = create_sequences(x_test, None, seq_len)

    train_dataset = TempDataset(x_train_seq, y_train_seq)
    val_dataset = TempDataset(x_val_seq, y_val_seq)
    test_dataset = TempDataset(x_test_seq, None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {x_train_seq.shape}, Val: {x_val_seq.shape}, Test: {x_test_seq.shape}")

    return train_loader, val_loader, test_loader, scaler_y, test_ft.iloc[seq_len:].reset_index(drop=True)[['Datetime']]


class TempLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.3):
        super(TempLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm1(x)
        last_hidden = lstm_out[:, -1, :]

        x = self.dropout(last_hidden)
        x = self.tanh(x)
        x, (hidden, cell) = self.lstm2(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validation(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = criterion(model(x_batch), y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch in loader:
            preds.append(model(x_batch.to(device)).cpu().numpy())
    return np.concatenate(preds)


if __name__ == "__main__":
    # df_raw = pd.read_csv("feb.csv", delimiter=";")
    #
    # df = create_dataframe(df_raw)
    # df.to_csv("df1.csv", sep=";")
    df = pd.read_csv("df1.csv", delimiter=";")
    train_df = df.iloc[:200]
    test_df = df.iloc[200:]

    train_loader, val_loader, test_loader, scaler_y, test_metadata = prep_dataloaders(
        train_df, test_df
    )

    # ds = train_loader.dataset
    # train_ratio = 0.8
    # train_size = int(train_ratio * len(ds))
    # val_size = len(ds) - train_size
    #
    # train_indices = list(range(train_size))
    # val_indices = list(range(train_size, len(ds)))
    #
    # train_ds = torch.utils.data.Subset(ds, train_indices)
    # val_ds = torch.utils.data.Subset(ds, val_indices)
    # # train_ds, val_ds = random_split(ds, [train_size, val_size])
    #
    # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 100
    lr = 1e-3

    sample_batch = next(iter(train_loader))[0]
    input_dim = sample_batch.shape[2]

    model = TempLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=3,
        dropout=0.3
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\nStarting learning")
    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validation(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            print("Best model!")

    # filename = f"models/model_{int(best_val_loss*1e3)}_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.pth"
    # print(filename)
    # torch.save(best_model, filename)

    print(f"\nFinished learning\nBest validation loss: {best_val_loss: .4f}")
    predictions = scaler_y.inverse_transform(predict(model, test_loader, device))
    preds = pd.DataFrame()
#    preds["real_value"] = test_df.loc["Temperature"]
#    preds["pred_value"] = predictions.reshape(20)
    print(test_df["Temperature"], predictions)
