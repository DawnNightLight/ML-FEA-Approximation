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

data1 = pd.read_csv("../../Misc/Run Results/FinrayTestRun.csv")
data2 = pd.read_csv("../../Misc/Run Results/FinrayManyVariants.csv")

data = pd.concat([data1, data2], ignore_index=True)

data = pd.read_csv("../../Misc/Run Results/FinrayTestRun.csv")

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


X_pool, X_test, y_pool, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Total simulations:", len(data))
print("Training pool:", len(X_pool))
print("Test set:", len(X_test))


training_sizes = [
    25,
    50,
    100,
    200,
    500,
    1000,
    2000,
    3000,
    4000,
    len(X_pool)
]

training_sizes = [
    size for size in training_sizes
    if size <= len(X_pool)
]

number_of_trials = 10


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


results = []


for training_size in training_sizes:

    print("\n")
    print("========================================")
    print(f"TRAINING SIZE: {training_size}")
    print("========================================")


    for trial in range(number_of_trials):

        print(
            f"\nTrial {trial + 1}/{number_of_trials}"
        )


        X_train = X_pool.sample(
            n=training_size,
            random_state=trial
        )

        y_train = y_pool.loc[
            X_train.index
        ]


   
        for name, training_function in models.items():

            if name == "Neural Network":

                model, input_scaler, output_scaler = \
                    training_function(
                        X_train,
                        y_train
                    )

                X_test_scaled = input_scaler.transform(
                    X_test
                )

                X_test_tensor = torch.tensor(
                    X_test_scaled,
                    dtype=torch.float32
                )

                model.eval()

                with torch.no_grad():

                    predictions_scaled = model(
                        X_test_tensor
                    ).numpy()

                predictions = output_scaler.inverse_transform(
                    predictions_scaled
                ).flatten()


            else:

                model = training_function(
                    X_train,
                    y_train
                )

                predictions = model.predict(
                    X_test
                )


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


            results.append({

                "Training Size": training_size,

                "Trial": trial + 1,

                "Model": name,

                "MAE": mae,

                "RMSE": rmse,

                "R2": r2
            })


            print(
                f"{name}: "
                f"R²={r2:.4f}"
            )


results_df = pd.DataFrame(results)


print("\n\n")
print("========================================")
print("ALL RESULTS")
print("========================================")

print(
    results_df.to_string(index=False)
)


results_df.to_csv(
    "learning_curve_raw_results.csv",
    index=False
)


summary = (
    results_df
    .groupby(
        ["Training Size", "Model"]
    )
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),

        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),

        R2_mean=("R2", "mean"),
        R2_std=("R2", "std")
    )
    .reset_index()
)


print("\n\n")
print("========================================")
print("AVERAGE RESULTS")
print("========================================")

print(
    summary.to_string(index=False)
)


summary.to_csv(
    "learning_curve_summary.csv",
    index=False
)


plt.figure()

for model_name in models.keys():

    model_results = summary[
        summary["Model"] == model_name
    ]

    plt.errorbar(
        model_results["Training Size"],
        model_results["R2_mean"],
        yerr=model_results["R2_std"],
        marker="o",
        capsize=5,
        label=model_name
    )


plt.xlabel(
    "Number of Training FEA Simulations"
)

plt.ylabel(
    "R²"
)

plt.title(
    "Model Performance vs Training Data"
)

plt.legend()

plt.grid()

plt.show()


plt.figure()

for model_name in models.keys():

    model_results = summary[
        summary["Model"] == model_name
    ]

    plt.errorbar(
        model_results["Training Size"],
        model_results["MAE_mean"],
        yerr=model_results["MAE_std"],
        marker="o",
        capsize=5,
        label=model_name
    )


plt.xlabel(
    "Number of Training FEA Simulations"
)

plt.ylabel(
    "Mean Absolute Error"
)

plt.title(
    "Prediction Error vs Training Data"
)

plt.legend()

plt.grid()

plt.show()