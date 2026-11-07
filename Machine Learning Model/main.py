import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
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

trained_models = {}

# ============================================================
# 1. LOAD DATA
# ============================================================

# Read both CSV files into DataFrames
data1 = pd.read_csv("../../Misc/Run Results/FinrayTestRun.csv")
data2 = pd.read_csv("../../Misc/Run Results/FinrayManyVariants.csv")

# Combine them (headers will automatically align)
data = pd.concat([data1, data2], ignore_index=True)

input_columns = [
    "baseSize",
    "externalBeamWidth",
    "internalBeamSpacing",
    "internalBeamSize",
    "load"
]

output_column = "max_Displacement"

X = data[input_columns]
y = data[output_column]


# ============================================================
# 2. PERMANENT TRAIN / TEST SPLIT
# ============================================================

X_pool, X_test, y_pool, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Total simulations:", len(data))
print("Training pool:", len(X_pool))
print("Test set:", len(X_test))


# ============================================================
# 3. TRAINING SIZES
# ============================================================

training_sizes = [
    25,
    50,
    100,
    150,
    len(X_pool)
]

# Remove sizes larger than the training pool
training_sizes = [
    size for size in training_sizes
    if size <= len(X_pool)
]


# ============================================================
# 4. MODELS
# ============================================================

models = {

    "Linear Regression":
        train_linear_regression,

    "Random Forest":
        train_random_forest,

    "XGBoost":
        train_xgboost,

    "Neural Network":
        train_neural_network
}


# ============================================================
# 5. STORE RESULTS
# ============================================================

results = []


# ============================================================
# 6. TRAIN EACH MODEL AT EACH TRAINING SIZE
# ============================================================

for training_size in training_sizes:

    print("\n")
    print("========================================")
    print(f"TRAINING SIZE: {training_size}")
    print("========================================")


    # --------------------------------------------------------
    # Randomly select training cases
    # --------------------------------------------------------

    X_train = X_pool.sample(
        n=training_size,
        random_state=42
    )

    y_train = y_pool.loc[
        X_train.index
    ]


    # --------------------------------------------------------
    # Train every model
    # --------------------------------------------------------

    for name, training_function in models.items():

        print(f"\nTraining {name}...")


        # ====================================================
        # NEURAL NETWORK
        # ====================================================

        if name == "Neural Network":

            model, input_scaler, output_scaler = \
                training_function(
                    X_train,
                    y_train
                )

            # Scale test inputs
            X_test_scaled = input_scaler.transform(
                X_test
            )

            X_test_tensor = torch.tensor(
                X_test_scaled,
                dtype=torch.float32
            )

            # Predict
            model.eval()

            with torch.no_grad():

                predictions_scaled = model(
                    X_test_tensor
                ).numpy()

            # Convert back to original units
            predictions = output_scaler.inverse_transform(
                predictions_scaled
            ).flatten()


        # ====================================================
        # OTHER MODELS
        # ====================================================

        else:

            model = training_function(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )


        # ====================================================
        # CALCULATE METRICS
        # ====================================================

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        results.append({

            "Training Size": training_size,

            "Model": name,

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2
        })


        print(
            f"MAE = {mae:.4f} | "
            f"RMSE = {rmse:.4f} | "
            f"R² = {r2:.4f}"
        )

        if training_size == len(X_pool):

            if name == "Neural Network":
                trained_models[name] = (
                    model,
                    input_scaler,
                    output_scaler
                )

            else:
                trained_models[name] = model


# ============================================================
# 7. CONVERT RESULTS TO DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


print("\n\n")
print("========================================")
print("FINAL RESULTS")
print("========================================")

print(results_df.to_string(index=False))


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "training_size_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "training_size_results.csv"
)


# ============================================================
# 9. PLOT MAE
# ============================================================

plt.figure()

for model_name in models.keys():

    model_results = results_df[
        results_df["Model"] == model_name
    ]

    plt.plot(
        model_results["Training Size"],
        model_results["MAE"],
        marker="o",
        label=model_name
    )

plt.xlabel("Number of Training FEA Simulations")

plt.ylabel("Mean Absolute Error")

plt.title("Model Error vs Training Data")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 10. PLOT R²
# ============================================================

plt.figure()

for model_name in models.keys():

    model_results = results_df[
        results_df["Model"] == model_name
    ]

    plt.plot(
        model_results["Training Size"],
        model_results["R2"],
        marker="o",
        label=model_name
    )

plt.xlabel("Number of Training FEA Simulations")

plt.ylabel("R²")

plt.title("Model R² vs Training Data")

plt.legend()

plt.grid()

plt.show()