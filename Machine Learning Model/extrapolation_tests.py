import pandas as pd
import numpy as np
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_neural_network
)


# ============================================================
# LOAD DATA
# ============================================================

normal_data = pd.read_csv("../../Misc/Run Results/FinrayManyVariants.csv")
extreme_data = pd.read_csv("../../Misc/Run Results/FinrayTestRunExtreme.csv")


input_columns = [
    "baseSize",
    "externalBeamWidth",
    "internalBeamSpacing",
    "internalBeamSize",
    "load"
]

output_column = "max_Displacement"


X_normal = normal_data[input_columns]
y_normal = normal_data[output_column]

X_extreme = extreme_data[input_columns]
y_extreme = extreme_data[output_column]


# ============================================================
# TRAIN USING ALL NORMAL DATA
# ============================================================

models = {}

models["Linear Regression"] = train_linear_regression(
    X_normal,
    y_normal
)

models["Random Forest"] = train_random_forest(
    X_normal,
    y_normal
)

models["XGBoost"] = train_xgboost(
    X_normal,
    y_normal
)

nn_model, input_scaler, output_scaler = train_neural_network(
    X_normal,
    y_normal
)

models["Neural Network"] = (
    nn_model,
    input_scaler,
    output_scaler
)


# ============================================================
# TEST EXTREME CASES
# ============================================================

print("\n========================================")
print("EXTRAPOLATION RESULTS")
print("========================================")


for name, model in models.items():

    # --------------------------------------------------------
    # Neural Network
    # --------------------------------------------------------

    if name == "Neural Network":

        nn_model, input_scaler, output_scaler = model

        X_scaled = input_scaler.transform(
            X_extreme
        )

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        nn_model.eval()

        with torch.no_grad():

            predictions_scaled = nn_model(
                X_tensor
            ).numpy()

        predictions = output_scaler.inverse_transform(
            predictions_scaled
        ).flatten()


    # --------------------------------------------------------
    # Other models
    # --------------------------------------------------------

    else:

        predictions = model.predict(
            X_extreme
        )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_extreme,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_extreme,
            predictions
        )
    )

    r2 = r2_score(
        y_extreme,
        predictions
    )


    print(f"\n{name}")

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")