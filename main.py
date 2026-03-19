import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_csv("feb.csv", delimiter=";")

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
    data["Day"] = data["Datetime"].dt.dayofyear
    data["Hour"] = data["Datetime"].dt.hour

    data["Hour_sin"] = np.sin(2 * np.pi * data["Hour"] / 24.0)
    data["Hour_cos"] = np.sin(2 * np.pi * data["Hour"] / 24.0)

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
            lambda x: x.rolling(window, mean_periods=1).mean().shift(1)
        )
        train[f'rolling_std_{window}'] = train['Temperature'].transform(
            lambda x: x.rolling(window, mean_periods=1).std().shift(1)
        )

    train['temp_diff'] = train['Temperature'].diff(1)

    if "Temperature" not in test.columns:
        test["Temperature"] = 0

    combined = pd.concat([train, test]).sort_values(["Datetime"])

    for lag in lags:
        combined[f'temp_lag_{lag}'] = combined['Temperature'].shift(lag)

    for window in windows:
        combined[f'rolling_mean_{window}'] = combined['Temperature'].transform(
            lambda x: x.rolling(window, mean_periods=1).mean().shift(1)
        )
        combined[f'rolling_std_{window}'] = combined['Temperature'].transform(
            lambda x: x.rolling(window, mean_periods=1).std().shift(1)
        )

    combined['temp_diff'] = combined['Temperature'].diff(1)

    train_mod = combined.iloc[:len(train)].dropna().reset_index(drop=True)
    test_mod = combined.iloc[len(train):].reset_index(drop=True)

    for col in test_mod.select_dtypes(include=[np.number]).columns:
        test_mod[col] = test_mod[col].fillna(0)

    if 'Temperature' not in test_df.columns:
        test_mod = test_mod.drop('Temperature', axis=1)

    return train_mod, test_mod


def prep_dataloaders(train_df, test_df, seq_len=4, batch_size=32):
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

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = scaler_x.fit_transform(train_ft[feature_cols])
    y_train = scaler_y.fit_transform(train_ft[["Temperature"]])
    x_test = scaler_x.transform(test_ft[feature_cols])

    def create_sequences(x, y=None, seq_length=4):
        x_seq, y_seq = [], []
        for i in range(len(x) - seq_length):
            x_seq.append(x[i:i+seq_length])
            if y is not None:
                y_seq.append(y[i+seq_length])
        return np.array(x_seq), np.array(y_seq) if y is not None else None

    x_train_seq, y_train_seq = create_sequences(x_train, y_train, seq_len)
    x_test_seq, _ = create_sequences(x_test, None, seq_len)

    train_dataset = TempDataset(x_train_seq, y_train_seq)
    test_dataset = TempDataset(x_test_seq, None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {x_train_seq.shape}, Test: {x_test_seq.shape}")

    return train_loader, test_loader, scaler_y, test_ft.iloc[seq_len:].reset_index(drop=True)[['Date']]


df = create_dataframe(df_raw)
train_df = df.iloc[:20]
test_df = df.iloc[20:]

train_loader, test_loader, scaler_y, test_metadata = prep_dataloaders(
    train_df, test_df
)


class TempLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(TempLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        x = self.dropout(last_hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "mps")
n_epochs = 20
lr = 5e-2

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

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch in loader:
            preds.append(model(x_batch.to(device)).cpu().numpy())
    return np.concatenate(preds)

sample_batch = next(iter(train_loader))[0]
input_dim = sample_batch.shape[2]

model = TempLSTM(
    input_dim=input_dim,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("\nStarting learning")
for epoch in range(n_epochs):
    loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f}")

print("\nFinished learning")
predictions = scaler_y.inverse_transform(predict(model, test_loader, device))
