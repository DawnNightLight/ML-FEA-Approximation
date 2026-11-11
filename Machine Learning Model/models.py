import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Linear regression

def train_linear_regression(X_train, y_train):

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model


# Random forest

def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# xgboost
def train_xgboost(X_train, y_train):

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# neural network
class NeuralNetwork(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


def train_neural_network(
    X_train,
    y_train,
    epochs=1000
):


    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_scaled = input_scaler.fit_transform(X_train)

    y_scaled = output_scaler.fit_transform(
        np.asarray(y_train).reshape(-1, 1)
    )



    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_scaled,
        dtype=torch.float32
    )


    model = NeuralNetwork(
        input_size=X_train.shape[1]
    )



    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    model.train()

    for epoch in range(epochs):

        predictions = model(X_tensor)

        loss = loss_function(
            predictions,
            y_tensor
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    return model, input_scaler, output_scaler