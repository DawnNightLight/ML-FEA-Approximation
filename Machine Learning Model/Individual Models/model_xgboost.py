import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


data = pd.read_csv("../../../Misc/Run Results/FinrayTestRun.csv")

print("\nNumber of simulations:", len(data))


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


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


predictions = model.predict(X_test)


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


print("\n============================")
print("XGBOOST PERFORMANCE")
print("============================")

print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.6f}")


print("\nExample predictions:")

for actual, predicted in zip(
    y_test.values[:10],
    predictions[:10]
):

    print(
        f"FEA: {actual:.4f} "
        f"| XGBoost: {predicted:.4f}"
    )


plt.scatter(
    y_test,
    predictions
)

minimum = min(
    y_test.min(),
    predictions.min()
)

maximum = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("FEA Tip Displacement")
plt.ylabel("XGBoost Prediction")

plt.title("FEA vs XGBoost")

plt.show()


importance = model.feature_importances_

print("\nFeature importance:")

for feature, value in zip(
    input_columns,
    importance
):

    print(
        f"{feature}: {value:.4f}"
    )