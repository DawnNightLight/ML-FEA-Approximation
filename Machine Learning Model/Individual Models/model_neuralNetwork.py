import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("../../../Misc/Run Results/FinrayTestRun.csv")

print("Dataset:")
print(data.head())

print("\nNumber of simulations:", len(data))


# ============================================================
# 2. DEFINE INPUTS AND OUTPUT
# ============================================================


input_columns = [
    "baseSize",
    "externalBeamWidth",
    "internalBeamSpacing",
    "internalBeamSize",
    "load"
]

output_columns = [
    "max_Displacement"
]

X = data[input_columns].values
y = data[output_columns].values


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. NORMALISE DATA
# ============================================================

input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train = input_scaler.fit_transform(X_train)
X_test = input_scaler.transform(X_test)

y_train = output_scaler.fit_transform(y_train)
y_test = output_scaler.transform(y_test)


# Convert to PyTorch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# ============================================================
# 5. DEFINE NEURAL NETWORK
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(len(input_columns), 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, len(output_columns))
        )

    def forward(self, x):

        return self.network(x)


model = NeuralNetwork()


# ============================================================
# 6. LOSS FUNCTION AND OPTIMISER
# ============================================================

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 7. TRAIN
# ============================================================

epochs = 1000

for epoch in range(epochs):

    # Make predictions
    predictions = model(X_train)

    # Calculate error
    loss = loss_function(
        predictions,
        y_train
    )

    # Clear previous gradients
    optimizer.zero_grad()

    # Calculate gradients
    loss.backward()

    # Update network
    optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# 8. TEST MODEL
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(X_test)


# Convert predictions back to original units

predictions = output_scaler.inverse_transform(
    predictions.numpy()
)

actual = output_scaler.inverse_transform(
    y_test.numpy()
)


# ============================================================
# 9. CALCULATE ERROR
# ============================================================

mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

print("\n============================")
print("MODEL PERFORMANCE")
print("============================")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# ============================================================
# 10. DISPLAY SOME PREDICTIONS
# ============================================================

print("\nExample predictions:")

for i in range(min(10, len(actual))):

    print(
        f"FEA: {actual[i][0]:.3f} "
        f"| ML: {predictions[i][0]:.3f}"
    )


# ============================================================
# 11. PLOT FEA VS ML
# ============================================================

plt.scatter(
    actual,
    predictions
)

# Perfect prediction line

minimum = min(actual.min(), predictions.min())
maximum = max(actual.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("FEA Max Displacement")
plt.ylabel("ML Predicted Max Displacement")

plt.title("FEA vs Machine Learning")

plt.show()